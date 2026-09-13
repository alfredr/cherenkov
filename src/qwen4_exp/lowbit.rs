//! Building the low-bit expert stores from the 4-bit one, in process.
//! `experts2.bin` and `experts3.bin` coexist beside the source store;
//! choosing or rebuilding one precision leaves the other file intact.
//!
//! `experts.bin` holds each expert's three projections as MLX affine
//! 4-bit codes (`w = q * scale + bias`, groups of 64). A low-bit store
//! keeps the same scales and biases and drops the bottom bits of the
//! codes, reconstructing at the midpoint of the range each new code
//! covers: `q >> 1` with `(2q + 0.5)` at three bits, `q >> 2` with
//! `(4q + 1.5)` at two.
//!
//! The layout is the part that matters. A kernel that peels codes one at
//! a time costs 2.3x the 4-bit kernel's time even though it reads fewer
//! bytes, which makes the whole exercise pointless; these layouts put a
//! word's codes where one masked cast (`& 0x03030303`, `& 0x01010101`)
//! yields four codes that are CONSECUTIVE in the deinterleaved input
//! stream, exactly as the 4-bit nibble trick does. See
//! `fn_q2_rows_h` and `fn_q3_rows_h` in kernels/qwen4_exp/experts.metal,
//! which must agree with the packing here bit for bit.

use super::ExpertLayout;
use crate::units::BYTES_PER_GB;
use anyhow::{Context, Result};
use std::io::Write as _;
use std::os::unix::fs::FileExt as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Expert record layout shared by the packer, prefill, and decode.
/// Low-bit records keep the base order: three weight matrices followed
/// by six scale/bias blocks copied verbatim from the 4-bit record.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub bits: u32,
    pub stride: usize,
    pub mat: usize,
    pub scale_bytes: usize,
    pub gate_w: usize,
    pub up_w: usize,
    pub down_w: usize,
    pub gate_s: usize,
    pub gate_b: usize,
    pub up_s: usize,
    pub up_b: usize,
    pub down_s: usize,
    pub down_b: usize,
}

const PAGE: usize = 16384;

impl Layout {
    /// Preserve the base store's offsets verbatim; prefill and decode use
    /// this same description rather than reconstructing the record layout.
    pub fn four_bit(e: &ExpertLayout) -> Self {
        Self {
            bits: 4,
            stride: e.record_stride as usize,
            mat: e.inter * e.hidden / 2,
            scale_bytes: (e.gate_b - e.gate_s) as usize,
            gate_w: e.gate_w as usize,
            up_w: e.up_w as usize,
            down_w: e.down_w as usize,
            gate_s: e.gate_s as usize,
            gate_b: e.gate_b as usize,
            up_s: e.up_s as usize,
            up_b: e.up_b as usize,
            down_s: e.down_s as usize,
            down_b: e.down_b as usize,
        }
    }

    /// Address-table tag shared with the decode shaders.
    pub fn kind(&self) -> u8 {
        match self.bits {
            3 => 1,
            2 => 2,
            _ => 0,
        }
    }

    pub fn new(e: &ExpertLayout, bits: u32) -> Result<Self> {
        anyhow::ensure!(
            bits == 2 || bits == 3,
            "low-bit store must be 2 or 3 bits, not {bits}"
        );

        let codes = e.inter * e.hidden;

        anyhow::ensure!(
            codes.is_multiple_of(32),
            "matrices must be a whole number of 32-code chunks"
        );

        let mat = codes * bits as usize / 8;
        // One scale and one bias per group of `group` codes, bf16.
        let scale_bytes = (e.gate_b - e.gate_s) as usize;
        let body = 3 * mat + 6 * scale_bytes;
        let stride = body.div_ceil(PAGE) * PAGE;

        Ok(Layout {
            bits,
            stride,
            mat,
            scale_bytes,
            gate_w: 0,
            up_w: mat,
            down_w: 2 * mat,
            gate_s: 3 * mat,
            gate_b: 3 * mat + scale_bytes,
            up_s: 3 * mat + 2 * scale_bytes,
            up_b: 3 * mat + 3 * scale_bytes,
            down_s: 3 * mat + 4 * scale_bytes,
            down_b: 3 * mat + 5 * scale_bytes,
        })
    }

    fn manifest_json(&self, records: usize) -> String {
        format!(
            "{{\n \"bits\": {},\n \"layout\": \"vectorized\",\n \"stride\": {},\n \"records\": {},\n \
             \"w\": {{ \"gate\": {}, \"up\": {}, \"down\": {} }},\n \
             \"s\": {{ \"gate_s\": {}, \"gate_b\": {}, \"up_s\": {}, \"up_b\": {}, \"down_s\": {}, \"down_b\": {} }},\n \
             \"source\": \"experts.bin, built in process by src/qwen4_exp/lowbit.rs\"\n}}\n",
            self.bits,
            self.stride,
            records,
            self.gate_w,
            self.up_w,
            self.down_w,
            self.gate_s,
            self.gate_b,
            self.up_s,
            self.up_b,
            self.down_s,
            self.down_b
        )
    }
}

/// Bit position of each of a 32-code chunk's codes in the 2-bit layout:
/// word index (0 or 1) and the offset within it. One `& 0x03030303` at
/// shift `2p` then yields the four codes of group `p`, and those four are
/// consecutive in `xe` (groups 0 and 1) or `xo` (groups 2 and 3).
fn q2_slots() -> [(usize, u32); 32] {
    let mut out = [(0usize, 0u32); 32];

    for (c, slot) in out.iter_mut().enumerate() {
        let (t, r) = (c / 16, c % 16);
        let (p, b) = match (r % 2 == 0, r < 8) {
            (true, true) => (0, r / 2),
            (true, false) => (1, (r - 8) / 2),
            (false, true) => (2, (r - 1) / 2),
            (false, false) => (3, (r - 9) / 2),
        };
        *slot = (t, (8 * b + 2 * p) as u32);
    }

    out
}

/// Three-bit codes split into their upper two bits and lowest bit.
/// The upper pair uses the 2-bit layout; the lowest bit goes into a third
/// word at `8 * byte + 4 * word + group`. The shader reconstructs
/// `code = 2 * upper + lowest` from the corresponding masked casts.
fn q3_slots() -> [(usize, u32, u32); 32] {
    let mut out = [(0usize, 0u32, 0u32); 32];

    for (c, slot) in out.iter_mut().enumerate() {
        let (t, r) = (c / 16, c % 16);
        let (j, b) = match (r % 2 == 0, r < 8) {
            (true, true) => (0, r / 2),
            (true, false) => (1, (r - 8) / 2),
            (false, true) => (2, (r - 1) / 2),
            (false, false) => (3, (r - 9) / 2),
        };
        *slot = (t, (8 * b + 2 * j) as u32, (8 * b + 4 * t + j) as u32);
    }

    out
}

/// Repack one matrix of `codes` 4-bit codes (as u32 words of 8 nibbles)
/// into `dst`.
fn pack_matrix(src: &[u8], codes: usize, bits: u32, dst: &mut [u8]) {
    let read_word = |i: usize| -> u32 {
        u32::from_le_bytes([src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]])
    };
    let mut q4_codes = [0u8; 32];
    let chunks = codes / 32;

    if bits == 2 {
        let slots = q2_slots();

        for chunk in 0..chunks {
            for w in 0..4 {
                let word = read_word(chunk * 4 + w);

                for j in 0..8 {
                    q4_codes[w * 8 + j] = ((word >> (4 * j)) & 0xF) as u8;
                }
            }

            let mut packed_words = [0u32; 2];

            for (c, &(t, off)) in slots.iter().enumerate() {
                packed_words[t] |= ((q4_codes[c] >> 2) as u32) << off;
            }

            let offset = chunk * 8;

            dst[offset..offset + 4].copy_from_slice(&packed_words[0].to_le_bytes());
            dst[offset + 4..offset + 8].copy_from_slice(&packed_words[1].to_le_bytes());
        }
    } else {
        let slots = q3_slots();

        for chunk in 0..chunks {
            for w in 0..4 {
                let word = read_word(chunk * 4 + w);

                for j in 0..8 {
                    q4_codes[w * 8 + j] = ((word >> (4 * j)) & 0xF) as u8;
                }
            }

            let mut upper_words = [0u32; 2];
            let mut lowest_bits = 0u32;

            for (c, &(t, upper_offset, low_offset)) in slots.iter().enumerate() {
                let v = q4_codes[c] >> 1;
                upper_words[t] |= ((v >> 1) as u32) << upper_offset;
                lowest_bits |= ((v & 1) as u32) << low_offset;
            }

            let offset = chunk * 12;

            dst[offset..offset + 4].copy_from_slice(&upper_words[0].to_le_bytes());
            dst[offset + 4..offset + 8].copy_from_slice(&upper_words[1].to_le_bytes());
            dst[offset + 8..offset + 12].copy_from_slice(&lowest_bits.to_le_bytes());
        }
    }
}

fn pack_record(e: &ExpertLayout, l: &Layout, src: &[u8], dst: &mut [u8]) {
    let gate_codes = e.inter * e.hidden;
    let down_codes = e.hidden * e.inter;

    for (src_off, dst_off, codes) in [
        (e.gate_w as usize, l.gate_w, gate_codes),
        (e.up_w as usize, l.up_w, gate_codes),
        (e.down_w as usize, l.down_w, down_codes),
    ] {
        let bytes = codes / 2;

        pack_matrix(
            &src[src_off..src_off + bytes],
            codes,
            l.bits,
            &mut dst[dst_off..dst_off + l.mat],
        );
    }

    for (s, d) in [
        (e.gate_s, l.gate_s),
        (e.gate_b, l.gate_b),
        (e.up_s, l.up_s),
        (e.up_b, l.up_b),
        (e.down_s, l.down_s),
        (e.down_b, l.down_b),
    ] {
        dst[d..d + l.scale_bytes].copy_from_slice(&src[s as usize..s as usize + l.scale_bytes]);
    }
}

fn paths(dir: &Path, bits: u32) -> (PathBuf, PathBuf) {
    (
        dir.join(format!("experts{bits}.bin")),
        dir.join(format!("manifest{bits}.json")),
    )
}

/// Whether a usable store of this layout is already on disk. The
/// manifest is only a claim, so a few records are re-packed and compared
/// byte for byte: that is what catches a store written by an older
/// packing, which would otherwise be read as noise by the kernel.
fn usable(dir: &Path, e: &ExpertLayout, l: &Layout, records: usize) -> bool {
    let (bin, man) = paths(dir, l.bits);
    let Ok(meta) = std::fs::metadata(&bin) else {
        return false;
    };

    if meta.len() != (records * l.stride) as u64 {
        return false;
    }

    let Ok(bytes) = std::fs::read(&man) else {
        return false;
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };

    if v["layout"].as_str() != Some("vectorized")
        || v["stride"].as_u64() != Some(l.stride as u64)
        || v["records"].as_u64() != Some(records as u64)
    {
        return false;
    }

    spot_check(dir, e, l, records).unwrap_or(false)
}

/// Read-only readiness check shared by indexed preparation and the runtime.
pub(crate) fn is_usable(dir: &Path, e: &ExpertLayout, bits: u32) -> Result<bool> {
    let layout = Layout::new(e, bits)?;

    Ok(usable(dir, e, &layout, e.layers * e.experts))
}

/// Re-pack a handful of records spread through the file and compare with
/// what is stored. Reads about 15 MB, so it costs milliseconds.
fn spot_check(dir: &Path, e: &ExpertLayout, l: &Layout, records: usize) -> Result<bool> {
    let (bin, _) = paths(dir, l.bits);
    let src = std::fs::File::open(dir.join("experts.bin"))?;
    let dst = std::fs::File::open(&bin)?;
    let stride4 = e.record_stride as usize;
    let mut inbuf = vec![0u8; stride4];
    let mut want = vec![0u8; l.stride];
    let mut got = vec![0u8; l.stride];

    for r in [0, records / 3, 2 * records / 3, records - 1] {
        src.read_exact_at(&mut inbuf, (r * stride4) as u64)?;
        dst.read_exact_at(&mut got, (r * l.stride) as u64)?;
        want.fill(0);
        pack_record(e, l, &inbuf, &mut want);

        if want != got {
            eprintln!(
                "the {}-bit store does not match the current packing at record {r}",
                l.bits
            );

            return Ok(false);
        }
    }

    Ok(true)
}

/// Make sure the `bits`-bit store exists next to `experts.bin`, building
/// it from the 4-bit records if it is missing, the wrong size, or in the
/// older code-at-a-time layout. Returns the layout either way.
///
/// `force` rebuilds an existing store (--repack).
pub fn ensure(dir: &Path, e: &ExpertLayout, bits: u32, force: bool) -> Result<Layout> {
    ensure_with_policy(dir, e, bits, force, true)
}

pub(crate) fn ensure_with_policy(
    dir: &Path,
    e: &ExpertLayout,
    bits: u32,
    force: bool,
    allow_build: bool,
) -> Result<Layout> {
    let layouts = ensure_selected(dir, e, &[bits], force, allow_build)?;

    Ok(layouts[0])
}

/// Build missing targets together, reading each Q4 source record once.
/// Duplicate precisions are ignored; valid cached stores are left untouched.
pub fn ensure_many(dir: &Path, e: &ExpertLayout, bits: &[u32]) -> Result<Vec<Layout>> {
    ensure_selected(dir, e, bits, false, true)
}

fn ensure_selected(
    dir: &Path,
    e: &ExpertLayout,
    bits: &[u32],
    force: bool,
    allow_build: bool,
) -> Result<Vec<Layout>> {
    let mut selected = bits.to_vec();

    selected.sort_unstable();
    selected.dedup();

    let layouts = selected
        .iter()
        .map(|&b| Layout::new(e, b))
        .collect::<Result<Vec<_>>>()?;
    let records = e.layers * e.experts;
    let mut pending = Vec::new();

    for l in &layouts {
        if !force && usable(dir, e, l, records) {
            continue;
        }

        anyhow::ensure!(
            allow_build,
            "the {}-bit expert store is missing or invalid; server policy forbids building it",
            l.bits
        );
        pending.push(*l);
    }

    if pending.is_empty() {
        return Ok(layouts);
    }

    // Check the combined requirement before invalidating or writing any target.
    let additional_bytes = pending
        .iter()
        .map(|l| announce_build(dir, l, records))
        .sum();

    crate::storage::require_space(dir, additional_bytes)?;

    for l in &pending {
        let (_, man) = paths(dir, l.bits);

        // Failed rebuilds must not leave manifests claiming partial files are ready.
        if man.exists() {
            std::fs::remove_file(man)?;
        }
    }

    let t0 = std::time::Instant::now();

    build(dir, e, &pending, records)?;

    for l in &pending {
        let (_, man) = paths(dir, l.bits);

        std::fs::write(man, l.manifest_json(records))?;
    }

    let targets = pending
        .iter()
        .map(|l| l.bits.to_string())
        .collect::<Vec<_>>()
        .join("+");

    eprintln!(
        "built the {targets}-bit store in {:.0}s",
        t0.elapsed().as_secs_f64()
    );

    Ok(layouts)
}

fn announce_build(dir: &Path, l: &Layout, records: usize) -> u64 {
    let bits = l.bits;
    let (bin, _) = paths(dir, bits);
    let need = (records * l.stride) as u64;

    eprintln!(
        "{} the {bits}-bit expert store at {} ({:.1} GB; measured build about 76s, hardware/cache dependent)",
        if bin.exists() {
            "rebuilding"
        } else {
            "building"
        },
        bin.display(),
        need as f64 / BYTES_PER_GB as f64
    );

    let have = std::fs::metadata(&bin).map(|m| m.len()).unwrap_or(0);

    need.saturating_sub(have)
}

fn create_output(dir: &Path, l: &Layout, records: usize) -> Result<std::fs::File> {
    let (bin, _) = paths(dir, l.bits);
    let out = std::fs::OpenOptions::new()
        .create(true)
        // Preserve existing allocation; set_len sizes the store before every record is rewritten.
        .truncate(false)
        .read(true)
        .write(true)
        .open(&bin)
        .with_context(|| format!("creating {}", bin.display()))?;

    out.set_len((records * l.stride) as u64)?;

    Ok(out)
}

fn build(dir: &Path, e: &ExpertLayout, layouts: &[Layout], records: usize) -> Result<()> {
    let src_path = dir.join("experts.bin");
    let src = std::fs::File::open(&src_path)
        .with_context(|| format!("opening {}", src_path.display()))?;
    let outputs = layouts
        .iter()
        .map(|l| Ok((*l, create_output(dir, l, records)?)))
        .collect::<Result<Vec<_>>>()?;
    let stride4 = e.record_stride as usize;
    // One input and one reusable output buffer per worker, even for two targets.
    let output_stride = layouts.iter().map(|l| l.stride).max().unwrap_or(0);
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(6, |n| n.get().min(8));

    const BATCH: usize = 16;

    std::thread::scope(|s| -> Result<()> {
        let mut handles = Vec::new();

        for w in 0..workers {
            let (src, outputs, next, done) = (&src, &outputs, &next, &done);

            handles.push(s.spawn(move || -> Result<()> {
                let mut inbuf = vec![0u8; stride4];
                let mut outbuf = vec![0u8; output_stride];

                loop {
                    let lo = next.fetch_add(BATCH, Ordering::Relaxed);

                    if lo >= records {
                        return Ok(());
                    }

                    pack_records(
                        e,
                        src,
                        outputs,
                        lo..(lo + BATCH).min(records),
                        &mut inbuf,
                        &mut outbuf,
                    )?;

                    let n = done.fetch_add(BATCH, Ordering::Relaxed) + BATCH;

                    if w == 0 && n % 2048 < BATCH {
                        eprint!("\r  {}/{records} records", n.min(records));

                        let _ = std::io::stderr().flush();
                    }
                }
            }));
        }

        for h in handles {
            h.join()
                .map_err(|_| anyhow::anyhow!("packer thread panicked"))??;
        }

        Ok(())
    })?;
    eprintln!("\r  {records}/{records} records");

    for (_, out) in &outputs {
        out.sync_all()?;
    }

    Ok(())
}

fn pack_records(
    e: &ExpertLayout,
    src: &std::fs::File,
    outputs: &[(Layout, std::fs::File)],
    records: std::ops::Range<usize>,
    inbuf: &mut [u8],
    outbuf: &mut [u8],
) -> Result<()> {
    for r in records {
        src.read_exact_at(inbuf, r as u64 * e.record_stride)
            .with_context(|| format!("reading record {r}"))?;

        for (l, out) in outputs {
            let record = &mut outbuf[..l.stride];

            record.fill(0);
            pack_record(e, l, inbuf, record);
            out.write_all_at(record, (r * l.stride) as u64)
                .with_context(|| format!("writing {}-bit record {r}", l.bits))?;
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4_exp/lowbit.rs"]
mod tests;
