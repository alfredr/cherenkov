//! Commit/rollback, sequence reset, and prefix checkpoints.

use super::*;

impl Gpu<'_> {
    /// Keep the first `n` rows of the step in flight: roll the recurrent
    /// state back to after row n-1 if rows were rejected, advance the
    /// position. KV rows and the PLE ring are positional and get
    /// overwritten by the next step.
    pub fn commit(&mut self, n: usize) -> Result<()> {
        anyhow::ensure!(
            n >= 1 && n <= self.batch_nb,
            "commit {n} of {} rows",
            self.batch_nb
        );

        if n < self.batch_nb {
            anyhow::ensure!(self.batch_snap, "rollback needs a snapshotting step");

            let plane = n - 1;
            let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
            let enc = cb.computeCommandEncoder().context("encoder")?;

            for l in &self.layers {
                let Mix::Delta(d) = &l.mix else {
                    continue;
                };

                for (src, dst) in [(&d.mid, &d.state), (&d.mid_hist, &d.hist)] {
                    let len = (dst.length() / 4) as u32;

                    self.dispatch(
                        &enc,
                        &self.pipes.copy_f32,
                        |e| {
                            self.bind(e, 0, src, plane * dst.length());
                            self.bind(e, 1, dst, 0);
                            set_bytes(e, 2, &len);
                        },
                        len as usize,
                        256,
                        false,
                    );
                }
            }

            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        }

        self.pos = self.batch_pos + n;

        self.tokens.truncate(self.pos);

        Ok(())
    }

    /// Forget the sequence: recurrent states, conv histories and the PLE
    /// ring are zeroed; KV rows are overwritten positionally.
    pub fn reset(&mut self) {
        let zero = |b: &Buf| unsafe {
            std::ptr::write_bytes(b.contents().cast::<u8>().as_ptr(), 0, b.length());
        };

        for l in &self.layers {
            if let Mix::Delta(d) = &l.mix {
                zero(&d.state);
                zero(&d.hist);
            }

            if let Some(p) = &l.ple {
                zero(&p.hist);
            }
        }

        self.pos = 0;

        self.tokens.clear();

        self.batch_nb = 0;
        self.mtp_len = 0;
    }

    pub(super) fn prefix_regions(&self, pos: usize, mtp_len: usize) -> Vec<(&Buf, usize, usize)> {
        let c = &self.p.cfg;
        let kv_row = c.num_key_value_heads * c.head_dim;
        let scales = kv_q8_side(self.max_t, kv_row).0;
        let mut regions = Vec::new();

        for (l, n) in self
            .layers
            .iter()
            .map(|l| (l, pos))
            .chain(self.mtp.iter().map(|m| (&m.layer, mtp_len)))
        {
            match &l.mix {
                Mix::Delta(d) => {
                    regions.push((&d.state, 0, d.state.length()));
                    regions.push((&d.hist, 0, d.hist.length()));
                }
                Mix::Attn(a) => {
                    for b in [&a.kc, &a.vc] {
                        regions.push((b, 0, n * kv_row));
                        regions.push((b, scales, n * kv_row / 32 * 2));
                    }

                    // The QSA index caches are half precision (2 bytes each).
                    regions.push((&a.ikc, 0, n * c.indexer_head_dim * 2));
                    regions.push((
                        &a.blk,
                        0,
                        n.div_ceil(c.indexer_compress_ratio) * c.indexer_head_dim * 2,
                    ));
                }
            }

            if let Some(ple) = &l.ple {
                regions.push((&ple.hist, 0, ple.hist.length()));
            }
        }

        regions
    }

    pub(crate) fn prefix_state_bytes(&self) -> usize {
        self.state_bytes_at(self.pos, self.mtp_len)
    }

    pub(crate) fn state_bytes_at(&self, pos: usize, mtp_len: usize) -> usize {
        self.prefix_regions(pos, mtp_len)
            .iter()
            .map(|(_, _, n)| n + size_of::<Vec<u8>>())
            .sum::<usize>()
            + pos * size_of::<u32>()
            + size_of::<PrefixState>()
    }

    pub(crate) fn save_prefix(&self) -> PrefixState {
        let data = self
            .prefix_regions(self.pos, self.mtp_len)
            .into_iter()
            .map(|(b, off, n)| {
                assert!(off + n <= b.length());

                // Prefill has waited for all GPU work before checkpointing.
                unsafe {
                    std::slice::from_raw_parts(b.contents().cast::<u8>().as_ptr().add(off), n)
                }
                .to_vec()
            })
            .collect();

        PrefixState {
            pos: self.pos,
            mtp_len: self.mtp_len,
            tokens: self.tokens[..self.pos].to_vec(),
            data,
        }
    }

    /// Reuse suspended-request buffers; only initialized prefixes are copied.
    pub(crate) fn save_into(&self, checkpoint: &mut Option<PrefixState>) {
        let Some(state) = checkpoint else {
            *checkpoint = Some(self.save_prefix());

            return;
        };

        for ((buffer, offset, len), data) in self
            .prefix_regions(self.pos, self.mtp_len)
            .into_iter()
            .zip(&mut state.data)
        {
            data.resize(len, 0);

            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents().cast::<u8>().as_ptr().add(offset),
                    data.as_mut_ptr(),
                    len,
                );
            }
        }

        state.pos = self.pos;
        state.mtp_len = self.mtp_len;

        state.tokens.clone_from(&self.tokens);
    }

    pub(crate) fn restore_prefix(&mut self, state: &PrefixState) -> Result<()> {
        ensure!(
            state.pos <= self.max_t && state.mtp_len <= self.max_t,
            "prefix exceeds context"
        );

        let regions = self.prefix_regions(state.pos, state.mtp_len);

        ensure!(
            regions.len() == state.data.len(),
            "prefix state layout mismatch"
        );

        for ((b, off, n), data) in regions.into_iter().zip(&state.data) {
            ensure!(
                n == data.len() && off + n <= b.length(),
                "prefix region mismatch"
            );

            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    b.contents().cast::<u8>().as_ptr().add(off),
                    n,
                );
            }
        }

        self.pos = state.pos;
        self.mtp_len = state.mtp_len;

        self.tokens.clone_from(&state.tokens);

        self.batch_pos = self.pos;
        self.batch_nb = 0;
        self.folded_mtp = None;

        Ok(())
    }

    /// Start another request without retaining unbounded profiling history.
    /// Weights and expert residency remain loaded; sequence state is independent.
    pub(crate) fn reset_request(&mut self) {
        self.reset();
        self.clear_profile();
    }

    /// Server scheduling publishes counters each turn instead of retaining a log.
    pub(crate) fn clear_profile(&mut self) {
        self.expert_history.clear();
        self.route_history.clear();
        self.state_history.clear();
        self.ngram_ms.clear();
        self.step_ms.clear();
        self.gpu_ms.clear();
        self.io_ms.clear();
        self.misses.clear();
        self.miss_bytes.clear();
        self.warm.clear();
        self.set_ms.clear();
        self.read_ms.clear();
        self.lookahead_hit.clear();
        self.lookahead_issued.clear();
        self.la_log.clear();
        self.la_pending.clear();
        self.cut.clear();
        self.dispatches.clear();
        self.gpu_idle_ms.clear();
        self.rows.clear();
        self.mtp_ms.clear();
        self.prefill_stats.clear();
    }
}
