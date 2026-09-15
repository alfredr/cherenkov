//! Trunk row-batch execution and single-token stepping.

use super::*;

impl Gpu<'_> {
    /// Run `tokens` at positions pos.. as one batch; returns the greedy
    /// argmax after each row. Nothing is committed until `commit`.
    /// `snap` keeps rollback snapshots after each row but the last.
    /// With `fold_mtp`, the MTP head's first pass runs in the same
    /// command buffer over all `nb` rows, pairing row b with the trunk's
    /// own prediction for it (the token that follows any accepted row),
    /// so `mtp_draft` needs no round trip for the first draft.
    pub fn step_rows(&mut self, tokens: &[u32], snap: bool, fold_mtp: bool) -> Result<Vec<u32>> {
        let nb = tokens.len();

        anyhow::ensure!(
            !fold_mtp || self.mtp.is_some(),
            "folding the MTP pass needs the MTP head"
        );

        self.folded_mtp = None;

        // The shared scratch spans `trunk_rows` rows: one committed token
        // plus the drafts. The prefill row path is clamped to the same cap.
        anyhow::ensure!(
            (1..=self.trunk_rows).contains(&nb),
            "rows per step must be 1..={} (set --drafts to raise it)",
            self.trunk_rows
        );
        anyhow::ensure!(self.pos + nb <= self.max_t, "context capacity exceeded");
        anyhow::ensure!(
            !snap || nb - 1 <= MAX_SNAP,
            "at most {MAX_SNAP} draft rows per verify step"
        );

        let t0 = std::time::Instant::now();
        let ngram0 = self.ngram_gather_s.get();
        let c = &self.p.cfg;
        let h = c.hidden_size as u32;
        let hh = c.hc_hidden();

        self.tokens.truncate(self.pos);
        self.tokens.extend_from_slice(tokens);
        self.ngram_prefetch_start(self.pos, nb);

        self.batch_pos = self.pos;
        self.batch_nb = nb;
        self.batch_snap = snap;

        unsafe {
            let ids = self.scratch.ids.contents().cast::<u32>().as_ptr();

            for (i, &t) in tokens.iter().enumerate() {
                ids.add(IDS_IN + i).write(t);
            }
        }

        self.last_experts.clear();
        self.last_routes.clear();
        self.last_states.clear();
        self.last_route_w.clear();
        self.last_miss.clear();
        self.dispatch_count.set(0);

        self.step_no += 1;
        self.step_misses = 0;
        self.step_miss_bytes = 0;
        self.step_set_s = 0.0;
        self.step_read_s = 0.0;
        self.step_warm = 0;
        self.step_cut = 0;
        let snap_after = if snap { nb - 1 } else { 0 };
        let pos = self.pos;
        let n_layers = self.layers.len().min(layer_cap());
        let base = self.event_base;
        self.event_base += 4 * (n_layers as u64 + 1);

        let phase_clock = self.phase_clock();
        let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
        let mut enc = self.phase_encoder(&cb, None, 0)?;

        {
            let s = &self.scratch;
            let (i0, nbu) = (IDS_IN as u32, nb as u32);

            self.dispatch(
                &enc,
                &self.pipes.embed_rows,
                |e| {
                    self.bind(e, 0, &s.ids, 0);
                    self.bind(e, 1, &self.dense, self.embed.w);
                    self.bind(e, 2, &self.dense, self.embed.s);
                    self.bind(e, 3, &self.dense, self.embed.b);
                    self.bind(e, 4, &s.e, 0);
                    set_bytes(e, 5, &h);
                    set_bytes(e, 6, &i0);
                    set_bytes(e, 7, &nbu);
                },
                nb * h as usize,
                256,
                false,
            );

            let gp = self.group_params(false, 0.0);

            self.dispatch(
                &enc,
                &self.pipes.replicate_b,
                |e| {
                    self.bind(e, 0, &s.e, 0);
                    self.bind(e, 1, &s.hyper, 0);
                    set_bytes(e, 2, &gp);
                    set_bytes(e, 3, &nbu);
                },
                nb * hh,
                256,
                false,
            );
        }

        // A block's MoE output is injected by the next fused norm
        // (`pending`); only the PLE block needs it applied up front.
        let mut pending: Option<&Buf> = None;
        let mtp_fold = if fold_mtp { self.mtp.as_ref() } else { None };

        for li in 0..n_layers {
            let layer = &self.layers[li];

            self.encode_ple_before_block(&enc, layer, nb, pos, &mut pending)?;

            // The last trunk layer looks ahead into the MTP block (its
            // stream differs by the fold, an approximation like the rest).
            let la = self.lookahead_layer(li, n_layers, fold_mtp);

            self.encode_block(
                &cb,
                &mut enc,
                layer,
                li,
                pos,
                nb,
                snap_after,
                &self.scratch.hyper,
                pending.take(),
                la,
                base + 4 * li as u64 + 1,
            )?;

            pending = Some(&self.scratch.moe_out);
        }

        self.head_b(
            &enc,
            &self.final_mixer,
            nb,
            &self.scratch.hyper,
            0,
            pending.take(),
            &self.scratch.logits,
            IDS_OUT,
        );

        if let Some(mtp) = mtp_fold {
            // Row b of the head pairs the trunk's residual at pos + b with
            // the trunk's argmax for it (written by the head just above).
            self.encode_mtp_prelude(&enc, mtp, nb, IDS_OUT, &self.scratch.hyper, 0);

            let slot_row = self.layers.len();

            self.encode_block(
                &cb,
                &mut enc,
                &mtp.layer,
                slot_row,
                pos,
                nb,
                0,
                &self.scratch.mtp_hyper,
                None,
                None,
                base + 4 * n_layers as u64 + 1,
            )?;
            self.head_b(
                &enc,
                &mtp.mixer,
                nb,
                &self.scratch.mtp_hyper,
                0,
                Some(&self.scratch.moe_out),
                &self.scratch.mtp_logits,
                IDS_MTP_OUT,
            );
        }

        enc.endEncoding();
        cb.commit();

        // Service the layers in order while the GPU runs.
        let mut predicted: std::collections::VecDeque<(usize, Vec<u32>)> =
            std::collections::VecDeque::new();
        let (mut la_hits, mut la_total, mut la_issued) = (0usize, 0usize, 0usize);
        let mut io_s = 0.0f64;
        let mut turn_s = 0.0f64;
        let mtp_record = if fold_mtp {
            self.mtp.as_ref().map(|m| m.layer.moe.record_layer)
        } else {
            None
        };

        for li in 0..n_layers {
            let record_layer = self.layers[li].moe.record_layer;
            let next = self
                .lookahead_layer(li, n_layers, fold_mtp)
                .map(|layer| layer.moe.record_layer);
            let (hits, total, issued) = self.service_block(
                record_layer,
                li,
                nb,
                base + 4 * li as u64 + 1,
                &mut predicted,
                next,
                &mut io_s,
                &mut turn_s,
            )?;
            la_hits += hits;
            la_total += total;
            la_issued += issued;
        }

        if let Some(record_layer) = mtp_record {
            let slot_row = self.layers.len();
            let (hits, total, _) = self.service_block(
                record_layer,
                slot_row,
                nb,
                base + 4 * n_layers as u64 + 1,
                &mut predicted,
                None,
                &mut io_s,
                &mut turn_s,
            )?;
            la_hits += hits;
            la_total += total;
        }

        cb.waitUntilCompleted();

        self.collect_trunk_phases(n_layers, mtp_record, phase_clock);

        let gpu_s = cb.GPUEndTime() - cb.GPUStartTime();

        self.join_pending()?;
        self.join_inflight()?;
        self.gpu_ms.push(gpu_s * 1e3);
        self.gpu_idle_ms.push(turn_s * 1e3);
        self.dispatches.push(self.dispatch_count.get());
        self.io_ms.push(io_s * 1e3);
        self.set_ms.push(self.step_set_s * 1e3);
        self.warm.push(self.step_warm);
        self.read_ms.push(self.step_read_s * 1e3);
        self.misses.push(self.step_misses);
        self.cut.push(self.step_cut);
        self.miss_bytes.push(self.step_miss_bytes);
        self.lookahead_hit.push(if la_total > 0 {
            la_hits as f64 / la_total as f64
        } else {
            f64::NAN
        });
        self.lookahead_issued.push(la_issued);
        self.expert_history.push(self.last_experts.clone());
        self.route_history.push((
            tokens.to_vec(),
            std::mem::take(&mut self.last_routes),
            std::mem::take(&mut self.last_route_w),
            std::mem::take(&mut self.last_miss),
        ));

        if self.dump_states {
            self.state_history
                .push(std::mem::take(&mut self.last_states));
        }

        self.rows.push(nb);
        self.step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        self.ngram_ms
            .push((self.ngram_gather_s.get() - ngram0) * 1e3);

        if fold_mtp {
            self.folded_mtp =
                Some(self.read_u32(&self.scratch.ids, IDS_MTP_OUT + nb)[IDS_MTP_OUT..].to_vec());
        }

        Ok(self.read_u32(&self.scratch.ids, IDS_OUT + nb)[IDS_OUT..].to_vec())
    }

    /// PLE needs the pending expert output injected before its convolution.
    fn encode_ple_before_block(
        &self,
        enc: &Enc,
        layer: &GLayer,
        nb: usize,
        pos: usize,
        pending: &mut Option<&Buf>,
    ) -> Result<()> {
        let Some(pl) = &layer.ple else {
            return Ok(());
        };

        if let Some(out) = pending.take() {
            self.inject_b(enc, &self.scratch.hyper, out, nb);
        }

        self.ple_b(enc, pl, nb, pos)
    }

    /// One-block lookahead; deeper routing raised misses about 40%.
    /// The last trunk layer predicts the folded MTP block when enabled.
    fn lookahead_layer(&self, index: usize, count: usize, fold_mtp: bool) -> Option<&GLayer> {
        if !self.lookahead {
            return None;
        }

        if index + 1 < count {
            return Some(&self.layers[index + 1]);
        }

        if fold_mtp {
            return self.mtp.as_ref().map(|mtp| &mtp.layer);
        }

        None
    }

    /// One token, committed: the greedy next token.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        let out = self.step_rows(&[token], false, false)?;

        self.commit(1)?;

        Ok(out[0])
    }
}
