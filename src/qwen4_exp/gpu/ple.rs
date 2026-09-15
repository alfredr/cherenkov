//! N-gram prefetch/gather and PLE gating/convolution dispatch.

use super::*;

impl Gpu<'_> {
    /// The 16 n-gram row ids of the token at index `t` of `self.tokens`.
    pub(super) fn ngram_ids_at(&self, pl: &Ple, t: usize) -> Vec<u64> {
        CpuModel::ngram_ids_from(
            &self.p.cfg,
            &pl.multipliers,
            &pl.head_offsets,
            &pl.head_sizes,
            &self.tokens[..=t],
        )
    }

    /// Start pulling the n-gram rows of tokens t0..t0+n into the page
    /// cache on a thread pool, so the block's gather (which faults the
    /// mapping one row at a time) finds them warm. The ids only depend
    /// on the tokens, which are known when a step or chunk starts.
    pub(super) fn ngram_prefetch_start(&self, t0: usize, n: usize) {
        let Some(pl) = self.layers.iter().find_map(|l| l.ple.as_ref()) else {
            return;
        };

        self.ngram_prefetch_join();

        let mut ids: Vec<u64> = (t0..t0 + n)
            .flat_map(|t| self.ngram_ids_at(pl, t))
            .collect();

        ids.sort_unstable();
        ids.dedup();

        let row_bytes = self.p.manifest.ngram.row_bytes;
        let file = self.ngram_file.try_clone().expect("dup ngram fd");
        let h = std::thread::spawn(move || prefetch_ngram_rows(&file, &ids, row_bytes));
        *self.ngram_prefetch.borrow_mut() = Some(h);
    }

    pub(super) fn ngram_prefetch_join(&self) {
        if let Some(h) = self.ngram_prefetch.borrow_mut().take() {
            let _ = h.join();
        }
    }

    /// PLE block over `nb` rows at positions base_pos..: n-gram rows are
    /// gathered on the CPU from `self.tokens` (which must hold the batch).
    pub(super) fn ple_b(&self, enc: &Enc, pl: &Ple, nb: usize, base_pos: usize) -> Result<()> {
        let c = &self.p.cfg;
        let s = &self.scratch;
        let h = c.hidden_size as u32;
        let hh = c.hc_hidden();
        let dim = self.p.manifest.ngram.dim;
        // `nb` never exceeds the trunk row cap the buffer is sized for.
        let e = unsafe {
            std::slice::from_raw_parts_mut(
                pl.e.contents().cast::<f32>().as_ptr(),
                self.trunk_rows * c.ple_embed_dim,
            )
        };
        let t_gather = std::time::Instant::now();

        self.ngram_prefetch_join();

        for b in 0..nb {
            let ids = self.ngram_ids_at(pl, base_pos + b);

            anyhow::ensure!(
                ids.len() * dim == c.ple_embed_dim,
                "n-gram head layout mismatch"
            );

            let eb = &mut e[b * c.ple_embed_dim..(b + 1) * c.ple_embed_dim];

            for (hi, &id) in ids.iter().enumerate() {
                self.p.ngram_row(id, &mut eb[hi * dim..(hi + 1) * dim]);
            }
        }

        self.ngram_gather_s
            .set(self.ngram_gather_s.get() + t_gather.elapsed().as_secs_f64());
        self.prep_h(enc, &pl.e, 0, c.ple_embed_dim as u32, nb, &s.hc.h1);
        self.qmv_h(enc, &pl.key, &s.ple_key, nb, &s.hc.h1);
        self.qmv_h(enc, &pl.value, &s.ple_value, nb, &s.hc.h1);

        let groups = c.hc_count as u32;

        self.group_norm_b(
            enc,
            &s.ple_key,
            0,
            pl.norm_key,
            &s.ple_keyn,
            h,
            groups,
            0.0,
            nb,
        );
        self.group_norm_b(
            enc,
            &s.hyper,
            0,
            pl.norm_query,
            &s.ple_query,
            h,
            groups,
            0.0,
            nb,
        );

        let gp = self.group_params(true, 0.0);

        self.dispatch(
            enc,
            &self.pipes.ple_gate_b,
            |e| {
                self.bind(e, 0, &s.ple_keyn, 0);
                self.bind(e, 1, &s.ple_query, 0);
                self.bind(e, 2, &s.ple_value, 0);
                self.bind(e, 3, &s.ple_gated, 0);
                set_bytes(e, 4, &gp);
            },
            nb * c.hc_count,
            256,
            true,
        );
        self.group_norm_b(
            enc,
            &s.ple_gated,
            0,
            pl.norm_conv,
            &s.ple_gvn,
            h,
            groups,
            0.0,
            nb,
        );

        let cp = PleConvParams {
            channels: hh as u32,
            ksize: pl.kernel,
            dilation: pl.dilation,
            span: pl.span,
            filled: base_pos as u32,
            nb: nb as u32,
        };

        self.dispatch(
            enc,
            &self.pipes.ple_conv_b,
            |e| {
                self.bind(e, 0, &s.ple_gated, 0);
                self.bind(e, 1, &s.ple_gvn, 0);
                self.bind(e, 2, &self.dense, pl.conv.0);
                self.bind(e, 3, &pl.hist, 0);
                self.bind(e, 4, &s.ple_out, 0);
                set_bytes(e, 5, &cp);
            },
            hh,
            256,
            false,
        );
        self.add(enc, &s.hyper, &s.ple_out, nb * hh);

        Ok(())
    }
}

/// Touch each selected row once; workers claim disjoint indices from the queue.
fn prefetch_ngram_rows(file: &std::fs::File, ids: &[u64], row_bytes: u64) {
    use std::os::unix::fs::FileExt as _;

    use std::sync::atomic::{AtomicUsize, Ordering};

    let next = AtomicUsize::new(0);
    let workers = 16.min(ids.len().max(1));

    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                let mut buf = [0u8; 256];

                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);

                    if i >= ids.len() {
                        break;
                    }

                    let _ = file.read_at(&mut buf[..row_bytes as usize], ids[i] * row_bytes);
                }
            });
        }
    });
}
