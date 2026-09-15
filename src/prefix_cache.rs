//! Bounded LRU checkpoints of complete hybrid-model sequence state.

use crate::{
    options::Options,
    qwen4_exp::gpu::{Gpu, MAX_NB, PrefixState},
    runner::PrefillResume,
};
use anyhow::{Result, ensure};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

struct Entry {
    tokens: Vec<u32>,
    following: Option<u32>,
    complete: bool,
    next: u32,
    drafts: Vec<u32>,
    logits: Vec<f32>,
    drafting: bool,
    state: PrefixState,
    bytes: usize,
    touched: Instant,
}

pub(crate) struct PrefixCache {
    entries: VecDeque<Entry>,
    capacity: usize,
    used: usize,
    evictions: u64,
    max_entries: usize,
    idle: Option<Duration>,
}

fn matches(tokens: &[u32], following: Option<u32>, complete: bool, request: &[u32]) -> bool {
    request.starts_with(tokens)
        && if request.len() == tokens.len() {
            complete
        } else {
            // The last MTP cache row depends on the token AFTER the trunk prefix.
            // Never reuse it for a different continuation.
            following.is_none_or(|next| request[tokens.len()] == next)
        }
}

impl PrefixCache {
    pub(crate) fn new(capacity: usize, max_entries: usize, idle_seconds: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            used: 0,
            evictions: 0,
            max_entries,
            idle: (idle_seconds > 0).then(|| Duration::from_secs(idle_seconds)),
        }
    }

    fn evict_oldest(&mut self) {
        if let Some(entry) = self.entries.pop_front() {
            self.used -= entry.bytes;
            self.evictions += 1;
        }
    }

    pub(crate) fn expire(&mut self) {
        if let Some(idle) = self.idle {
            while self
                .entries
                .front()
                .is_some_and(|e| e.touched.elapsed() >= idle)
            {
                self.evict_oldest();
            }
        }
    }

    pub(crate) fn stats(&self) -> (usize, usize, u64) {
        (self.entries.len(), self.used, self.evictions)
    }

    fn store(&mut self, gpu: &Gpu<'_>, ids: &[u32], seed: &PrefillResume, drafting: bool) {
        let pos = gpu.pos;
        let bytes = gpu.prefix_state_bytes()
            + pos * 4
            + std::mem::size_of::<Entry>()
            + seed.drafts.len() * 4
            + seed.logits.as_ref().map_or(0, |l| l.len() * 4);

        if bytes > self.capacity || pos == 0 {
            return;
        }

        if let Some(i) = self
            .entries
            .iter()
            .position(|e| e.tokens == ids[..pos] && e.drafting == drafting)
        {
            self.used -= self.entries.remove(i).unwrap().bytes;
        }

        // Evict BEFORE copying state: allocation peaks also stay within the cache budget.
        while self.used + bytes > self.capacity || self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }

        let following = drafting.then(|| ids.get(pos).copied().unwrap_or(seed.next));

        self.entries.push_back(Entry {
            tokens: ids[..pos].to_vec(),
            following,
            complete: pos == ids.len(),
            next: seed.next,
            drafts: seed.drafts.clone(),
            logits: seed.logits.clone().unwrap_or_default(),
            drafting,
            state: gpu.save_prefix(),
            bytes,
            touched: Instant::now(),
        });

        self.used += bytes;
    }

    pub(crate) fn begin(
        &mut self,
        gpu: &mut Gpu<'_>,
        ids: &[u32],
        stable_boundaries: &[usize],
        options: &Options,
    ) -> Result<Prefill> {
        self.expire();

        let drafting = options.effective_drafts() > 0;
        let mut cached = 0;
        let mut seed = PrefillResume::default();

        if let Some(i) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.drafting == drafting && matches(&e.tokens, e.following, e.complete, ids)
            })
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)
        {
            let mut entry = self.entries.remove(i).unwrap();
            entry.touched = Instant::now();

            gpu.restore_prefix(&entry.state)?;

            cached = entry.tokens.len();
            seed = PrefillResume {
                next: entry.next,
                drafts: entry.drafts.clone(),
                logits: Some(entry.logits.clone()),
            };

            self.entries.push_back(entry);
        }

        let mut boundaries = vec![ids.len()];

        if self.capacity > 0 {
            boundaries.extend(
                stable_boundaries
                    .iter()
                    .copied()
                    .filter(|&b| b > cached && b < ids.len()),
            );

            if ids.len() > 1 {
                boundaries.push(ids.len() - 1);
            }
        }

        boundaries.sort_unstable();
        boundaries.dedup();

        Ok(Prefill {
            cached,
            boundaries,
            seed,
        })
    }
}

/// The cursor stays with its request; one advance processes at most one chunk.
pub(crate) struct Prefill {
    pub cached: usize,
    boundaries: Vec<usize>,
    pub seed: PrefillResume,
}

pub(crate) struct PrefillProgress {
    pub done: bool,
    /// Successful engine rows suitable for pacing; excludes boundary tails,
    /// memory-limited chunks, and the small-row execution path.
    pub pacing_tokens: Option<usize>,
}

impl Prefill {
    pub(crate) fn advance(
        &mut self,
        gpu: &mut Gpu<'_>,
        cache: &mut PrefixCache,
        ids: &[u32],
        options: &Options,
        quantum: usize,
    ) -> Result<PrefillProgress> {
        let Some(end) = self.boundaries.iter().copied().find(|&end| end > gpu.pos) else {
            return Ok(PrefillProgress {
                done: true,
                pacing_tokens: None,
            });
        };
        let pf_min = std::env::var("CHERENKOV_PREFILL_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let remaining = end - gpu.pos;
        let engine = remaining >= pf_min;
        let capacity = if engine {
            let fit = gpu.prefill_rows_fit(false)?;

            std::env::var("CHERENKOV_PREFILL_CHUNK")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(fit)
                .min(fit)
        } else {
            std::env::var("CHERENKOV_ROWS_MAX")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(MAX_NB)
                .clamp(1, MAX_NB)
                // The short-row path runs the shared trunk scratch, which is
                // sized for one committed token plus the drafts.
                .min(gpu.trunk_rows())
        };
        let n = chunk_len(remaining, capacity, quantum)?;
        let pos = gpu.pos;
        self.seed = prefill_rows(
            gpu,
            &ids[pos..pos + n],
            ids.get(pos + n).copied(),
            options.effective_drafts(),
            engine,
        )?;

        if gpu.pos == end || engine {
            cache.store(gpu, ids, &self.seed, options.effective_drafts() > 0);
        }

        gpu.prefill_release();

        Ok(PrefillProgress {
            done: gpu.pos == ids.len(),
            pacing_tokens: pacing_sample(remaining, capacity, quantum, engine).then_some(n),
        })
    }
}

fn pacing_sample(remaining: usize, capacity: usize, quantum: usize, engine: bool) -> bool {
    engine && remaining >= quantum && capacity >= quantum
}

/// Balance a boundary's chunks without exceeding memory or scheduling limits.
fn chunk_len(remaining: usize, capacity: usize, quantum: usize) -> Result<usize> {
    let size = capacity.min(quantum);

    ensure!(
        remaining > 0 && size > 0,
        "prefill chunk requires tokens and capacity"
    );

    Ok(remaining.div_ceil(remaining.div_ceil(size)))
}

/// Advance one prompt chunk and retain the predictions needed to resume decode.
fn prefill_rows(
    gpu: &mut Gpu<'_>,
    rows: &[u32],
    following: Option<u32>,
    drafts: usize,
    engine: bool,
) -> Result<PrefillResume> {
    let last = following.is_none();

    if engine {
        let (next, draft) = gpu.prefill_chunk(rows, following, false)?;
        let mut seed = PrefillResume {
            next,
            drafts: Vec::new(),
            logits: Some(gpu.logits().to_vec()),
        };

        if drafts == 0 || !last {
            return Ok(seed);
        }

        seed.drafts.push(draft);

        if drafts >= 2 {
            seed.drafts.push(gpu.mtp_chain(draft)?);
        }

        return Ok(seed);
    }

    let result = gpu.step_rows(rows, false, false)?;

    gpu.commit(rows.len())?;

    let mut seed = PrefillResume {
        next: result[rows.len() - 1],
        drafts: Vec::new(),
        logits: Some(gpu.logits_row(rows.len() - 1).to_vec()),
    };

    if drafts == 0 {
        return Ok(seed);
    }

    let mut next = rows[1..].to_vec();

    next.push(following.unwrap_or(seed.next));

    seed.drafts = gpu.mtp_draft(&next, if last { drafts } else { 1 })?;

    Ok(seed)
}

#[cfg(test)]
#[path = "../tests/unit/prefix_cache.rs"]
mod tests;
