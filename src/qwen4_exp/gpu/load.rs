//! Model loading, pipeline creation, and fixed/pool allocation.

use super::*;
use crate::units::BYTES_PER_GB;
use objc2_metal::MTLDevice;

impl<'a> Gpu<'a> {
    pub fn load(p: &'a Packed, max_t: usize, options: &Options) -> Result<Self> {
        Self::load_bounded(p, max_t, options, 0, None, prefill::MAX_PREFILL_ROWS)
    }

    pub(crate) fn load_bounded(
        p: &'a Packed,
        max_t: usize,
        options: &Options,
        reserve: usize,
        memory_bytes: Option<usize>,
        prefill_rows: usize,
    ) -> Result<Self> {
        options.validate()?;
        p.manifest.experts.validate_dimensions()?;

        let c = &p.cfg;

        anyhow::ensure!(
            c.hidden_size.is_multiple_of(256),
            "fn_norm_prep_b needs hidden % 256 == 0"
        );
        anyhow::ensure!(
            c.indexer_head_dim <= 256 && c.indexer_head_dim.is_multiple_of(32),
            "indexer head dim must be a multiple of 32 up to 256"
        );
        anyhow::ensure!(
            c.indexer_n_heads * c.indexer_head_dim <= 1024,
            "indexer query heads too wide for fn_index_score"
        );
        anyhow::ensure!(
            c.indexer_budget.is_multiple_of(c.indexer_compress_ratio),
            "indexer budget must be a multiple of the block size"
        );
        anyhow::ensure!(
            c.moe_intermediate_size.is_multiple_of(64),
            "expert width must be a multiple of the 64-code quantization group"
        );
        anyhow::ensure!(
            c.num_experts_per_tok * MAX_NB < SLOT_STRIDE,
            "slot table too narrow"
        );
        anyhow::ensure!(
            c.output_gate_type == "sigmoid",
            "only the sigmoid DeltaNet output gate is implemented"
        );

        let ctx = MetalContext::new()?;
        let allocation_limit = memory_bytes
            .map(|bytes| {
                bytes
                    .checked_sub(reserve)
                    .context("cache exceeds memory budget")
            })
            .transpose()?;

        ctx.allocation_limit.set(allocation_limit);

        let lib = ctx.compile_library(FORWARD_MSL)?;
        let blib = ctx.compile_library(BATCH_MSL)?;
        let per_nb = |base: &str| -> Result<[Pso; MAX_NB]> {
            Ok([
                ctx.pipeline(&blib, &format!("{base}1"))?,
                ctx.pipeline(&blib, &format!("{base}2"))?,
                ctx.pipeline(&blib, &format!("{base}3"))?,
                ctx.pipeline(&blib, &format!("{base}4"))?,
            ])
        };
        let expert_qmm = |bits| -> Result<[Pso; 3]> {
            Ok([
                ctx.pipeline(&blib, &format!("fn_expert_qmm_q{bits}_n8"))?,
                ctx.pipeline(&blib, &format!("fn_expert_qmm_q{bits}_n16"))?,
                ctx.pipeline(&blib, &format!("fn_expert_qmm_q{bits}_n32"))?,
            ])
        };
        let pipes = Pipes {
            expert_qmm: [expert_qmm(2)?, expert_qmm(3)?],
            qmv_h: ctx.pipeline(&lib, "qmv_multi_h")?,
            qmv_hn: ctx.pipeline(&lib, "qmv_multi_hn")?,
            prep_h: ctx.pipeline(&lib, "deinterleave_bh")?,
            embed_rows: ctx.pipeline(&lib, "embed_rows")?,
            qk_norm_rope_b: ctx.pipeline(&lib, "qk_norm_rope_b")?,
            conv_b: ctx.pipeline(&lib, "conv_b")?,
            delta_norms: ctx.pipeline(&lib, "delta_norms")?,
            delta_gates: ctx.pipeline(&lib, "delta_gates")?,
            delta_scan2: ctx.pipeline(&lib, "delta_scan2")?,
            kv_append_q8: ctx.pipeline(&lib, "kv_append_q8")?,
            attn_part2_q8: ctx.pipeline(&lib, "attn_part2_q8")?,
            attn_combine: ctx.pipeline(&lib, "attn_combine")?,
            argmax_partial: ctx.pipeline(&lib, "argmax_partial")?,
            argmax_final: ctx.pipeline(&lib, "argmax_final")?,
            add: ctx.pipeline(&lib, "add_inplace")?,
            copy_f32: ctx.pipeline(&lib, "copy_f32")?,
            zero: ctx.pipeline(&blib, "fn_zero")?,
            replicate_b: ctx.pipeline(&blib, "fn_replicate_b")?,
            group_norm_b: ctx.pipeline(&blib, "fn_group_norm_b")?,
            norm_prep_b: ctx.pipeline(&blib, "fn_norm_prep_b")?,
            qmv_silu_b: per_nb("fn_qmv_silu_b")?,
            hc_mix_b: ctx.pipeline(&blib, "fn_hc_mix_b")?,
            inject_b: ctx.pipeline(&blib, "fn_inject_b")?,
            bf16_matvec_b: ctx.pipeline(&blib, "fn_bf16_matvec_b")?,
            topk_softmax_b: ctx.pipeline(&blib, "fn_topk_softmax_b")?,
            moe_gate_up_b: per_nb("fn_moe_gate_up_b")?,
            moe_act_b: ctx.pipeline(&blib, "fn_moe_act_b")?,
            moe_down_b: per_nb("fn_moe_down_b")?,
            moe_combine_b: ctx.pipeline(&blib, "fn_moe_combine_b")?,
            ple_gate_b: ctx.pipeline(&blib, "fn_ple_gate_b")?,
            ple_conv_b: ctx.pipeline(&blib, "fn_ple_conv_b")?,
            mtp_fold: ctx.pipeline(&blib, "fn_mtp_fold")?,
            gate_norm_sigmoid_b: ctx.pipeline(&blib, "fn_delta_gate_norm_sigmoid_b")?,
            index_append: ctx.pipeline(&blib, "fn_index_append")?,
            index_blocks: ctx.pipeline(&blib, "fn_index_blocks")?,
            index_q: ctx.pipeline(&blib, "fn_index_q")?,
            index_score: ctx.pipeline(&blib, "fn_index_score")?,
            index_select: ctx.pipeline(&blib, "fn_index_select")?,
            attn_sel: ctx.pipeline(&blib, "fn_attn_part2_q8_sel")?,
            qmm_n8: ctx.pipeline(&lib, "qmm_af4_n8")?,
            qmm_n16: ctx.pipeline(&lib, "qmm_af4_n16")?,
            qmm_w: ctx.pipeline(&lib, "qmm_af4_w")?,
            silu_mul: ctx.pipeline(&lib, "silu_mul")?,
            attn_q_stage: ctx.pipeline(&lib, "attn_q_stage")?,
            attn_kv_stage: ctx.pipeline(&lib, "attn_kv_stage")?,
            gemm_hh: ctx.pipeline(&lib, "gemm_hh")?,
            attn_o_scatter: ctx.pipeline(&lib, "attn_o_scatter")?,
            softmax_sel: ctx.pipeline(&blib, "fn_attn_softmax_sel")?,
            silu_rows: ctx.pipeline(&blib, "fn_silu_rows")?,
            gather_rows: ctx.pipeline(&blib, "fn_gather_rows")?,
            scatter_add_rows: ctx.pipeline(&blib, "fn_scatter_add_rows")?,
            shared_add_rows: ctx.pipeline(&blib, "fn_shared_add_rows")?,
            qmv_small_b: ctx.pipeline(&blib, "fn_qmv_small_b")?,
        };
        // Gpu borrows Packed for its entire lifetime, keeping this
        // page-aligned mapping alive while dense kernels can read it.
        let dense = unsafe { ctx.wrap_mmap(&p.dense)? };

        let q = |prefix: &str| -> Result<Q> {
            let w = p.manifest.dense(&format!("{prefix}.weight"))?;
            let s = p.manifest.dense(&format!("{prefix}.scales"))?;
            let b = p.manifest.dense(&format!("{prefix}.biases"))?;

            anyhow::ensure!(w.dtype == "U32", "{prefix}: not a Q4 weight");
            anyhow::ensure!(w.offset % 16 == 0, "{prefix}: weight offset not 16-aligned");

            Ok(Q {
                w: w.offset as usize,
                s: s.offset as usize,
                b: b.offset as usize,
                out: w.shape[0] as u32,
                inp: (w.shape[1] * 8) as u32,
            })
        };
        let t = |name: &str| -> Result<T> {
            let e = p.manifest.dense(name)?;

            anyhow::ensure!(e.dtype == "BF16", "{name}: expected BF16");

            Ok(T(e.offset as usize))
        };
        let hc = |prefix: &str, inject: bool| -> Result<Hc> {
            Ok(Hc {
                norm: t(&format!("{prefix}.hc_norm.weight"))?,
                down: q(&format!("{prefix}.input_mix_weight_down"))?,
                up: q(&format!("{prefix}.input_mix_weight_up"))?,
                inject: if inject {
                    Some(q(&format!("{prefix}.block_inject_weight"))?)
                } else {
                    None
                },
            })
        };

        let h = c.hidden_size;
        let hh = c.hc_hidden();
        let kv_row = c.num_key_value_heads * c.head_dim;
        let kv_side = kv_q8_side(max_t, kv_row).1;
        let conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim
            + c.linear_num_value_heads * c.linear_value_head_dim;
        let v_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        let state_len = c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim;
        let hist_len = conv_dim * (c.linear_conv_kernel_dim - 1);

        // The shared trunk scratch (and the per-layer DeltaNet snapshot
        // planes and PLE n-gram row) only need to span the rows a decode step
        // ever runs: one committed token plus the drafts, which is also the
        // cap the prefill row path is clamped to. Without MTP the scratch
        // holds a single row.
        let trunk_rows = (1 + options.effective_drafts()).min(MAX_NB);
        let snap_rows = options.effective_drafts().min(MAX_SNAP);

        let load_layer =
            |lp: &str, linear: bool, record_layer: usize, with_ple: bool| -> Result<GLayer> {
                let mix = if linear {
                    let d = format!("{lp}.linear_attn");

                    Mix::Delta(Delta {
                        qkv: q(&format!("{d}.in_proj_qkv"))?,
                        z: q(&format!("{d}.in_proj_z"))?,
                        a: q(&format!("{d}.in_proj_a"))?,
                        b: q(&format!("{d}.in_proj_b"))?,
                        conv: t(&format!("{d}.conv1d.weight"))?,
                        a_log: t(&format!("{d}.A_log"))?,
                        dt_bias: t(&format!("{d}.dt_bias"))?,
                        norm: t(&format!("{d}.norm.weight"))?,
                        o: q(&format!("{d}.out_proj"))?,
                        state: ctx.new_buffer(state_len * 4)?,
                        hist: ctx.new_buffer(hist_len * 4)?,
                        // Zero-length MTL buffers are invalid; the planes are
                        // never used when there are no drafts, so floor to 1.
                        mid: ctx.new_buffer((snap_rows * state_len * 4).max(1))?,
                        mid_hist: ctx.new_buffer((snap_rows * hist_len * 4).max(1))?,
                    })
                } else {
                    let a = format!("{lp}.self_attn");

                    Mix::Attn(Attn {
                        q: q(&format!("{a}.q_proj"))?,
                        k: q(&format!("{a}.k_proj"))?,
                        v: q(&format!("{a}.v_proj"))?,
                        o: q(&format!("{a}.o_proj"))?,
                        qn: t(&format!("{a}.q_norm.weight"))?,
                        kn: t(&format!("{a}.k_norm.weight"))?,
                        kc: ctx.new_buffer(kv_side)?,
                        vc: ctx.new_buffer(kv_side)?,
                        iqk: q(&format!("{a}.indexer.index_qk_proj"))?,
                        iqn: t(&format!("{a}.indexer.q_layernorm.weight"))?,
                        ikn: t(&format!("{a}.indexer.k_layernorm.weight"))?,
                        // The QSA index caches are half precision (see the index
                        // kernels), so each element is 2 bytes.
                        ikc: ctx.new_buffer(max_t * c.indexer_head_dim * 2)?,
                        blk: ctx.new_buffer(
                            (max_t / c.indexer_compress_ratio + 1) * c.indexer_head_dim * 2,
                        )?,
                    })
                };
                let moe = Moe {
                    router: t(&format!("{lp}.mlp.gate.weight"))?,
                    shared_gate: t(&format!("{lp}.mlp.shared_expert_gate.weight"))?,
                    sg: q(&format!("{lp}.mlp.shared_expert.gate_proj"))?,
                    su: q(&format!("{lp}.mlp.shared_expert.up_proj"))?,
                    sd: q(&format!("{lp}.mlp.shared_expert.down_proj"))?,
                    record_layer,
                };
                let ple = if with_ple {
                    let pp = format!("{lp}.ple");
                    let ngram = p.ngram_metadata(&format!("{pp}.ple_embedding"))?;
                    let kernel = p.shape(&format!("{pp}.conv1d.weight"))?[2] as u32;
                    let span = (kernel - 1) * c.ngram_size as u32;

                    Some(Ple {
                        key: q(&format!("{pp}.key_proj"))?,
                        value: q(&format!("{pp}.value_proj"))?,
                        norm_key: t(&format!("{pp}.norm_key.weight"))?,
                        norm_query: t(&format!("{pp}.norm_query.weight"))?,
                        norm_conv: t(&format!("{pp}.norm_conv.weight"))?,
                        conv: t(&format!("{pp}.conv1d.weight"))?,
                        kernel,
                        dilation: c.ngram_size as u32,
                        span,
                        hist: ctx.new_buffer(span as usize * hh * 4)?,
                        e: ctx.new_buffer(trunk_rows * c.ple_embed_dim * 4)?,
                        multipliers: ngram.multipliers,
                        head_offsets: ngram.head_offsets,
                        head_sizes: ngram.head_sizes,
                    })
                } else {
                    None
                };

                Ok(GLayer {
                    attn_hc: hc(&format!("{lp}.attn_hyper_connection"), true)?,
                    mlp_hc: hc(&format!("{lp}.mlp_hyper_connection"), true)?,
                    mix,
                    moe,
                    ple,
                })
            };

        let m = "language_model.model";
        let mut layers = Vec::with_capacity(c.num_hidden_layers);

        for l in 0..c.num_hidden_layers {
            layers.push(load_layer(
                &format!("{m}.layers.{l}"),
                c.is_linear(l),
                p.expert_layer(l),
                c.ple_layer() == Some(l),
            )?);
        }

        let mtp = if c.mtp_num_hidden_layers > 0 && options.drafts > 0 {
            let layer = load_layer("mtp.layers.0", false, p.mtp_expert_layer(0)?, false)?;

            anyhow::ensure!(
                matches!(layer.mix, Mix::Attn(_)),
                "MTP layer must be full attention"
            );

            Some(Mtp {
                enorm: t("mtp.pre_fc_norm_embedding.weight")?,
                hnorm: t("mtp.pre_fc_norm_hidden.weight")?,
                fc_e: q("mtp.fc_embedding")?,
                fc_h: q("mtp.fc_hidden")?,
                layer,
                mixer: hc("mtp.hyper_connection_mixer", false)?,
            })
        } else {
            None
        };

        let max_blk = max_t.div_ceil(ATTN_TB).max(ATTN_MAX_WG);
        let vocab = c.vocab_size;
        let inter = c.moe_intermediate_size;

        anyhow::ensure!(
            c.shared_expert_intermediate_size == inter,
            "shared expert width differs from routed experts"
        );

        // Reduced checkpoints can retain wide attention/DeltaNet heads even
        // when their residual stream is small. Size scratch for every projection.
        let max_in = [
            hh,
            c.ple_embed_dim,
            2 * h,
            c.hc_lowrank,
            v_dim,
            c.num_attention_heads * c.head_dim,
            inter,
        ]
        .into_iter()
        .max()
        .unwrap();
        let half_set = |rows: usize, in_dim: usize| -> Result<HalfSet> {
            Ok(HalfSet {
                xe: ctx.new_buffer(rows * in_dim / 2 * 2)?,
                xo: ctx.new_buffer(rows * in_dim / 2 * 2)?,
                xsum: ctx.new_buffer(rows * in_dim / 32 * 4)?,
            })
        };
        let hc_bufs = || -> Result<HcBufs> {
            Ok(HcBufs {
                normed: ctx.new_buffer(trunk_rows * hh * 4)?,
                d: ctx.new_buffer(trunk_rows * c.hc_lowrank * 4)?,
                u: ctx.new_buffer(trunk_rows * hh * 4)?,
                mixed: ctx.new_buffer(trunk_rows * h * 4)?,
                inj: ctx.new_buffer(trunk_rows * c.hc_count * 4)?,
                h1: half_set(trunk_rows, max_in)?,
                h2: half_set(trunk_rows, max_in)?,
            })
        };
        // The routed-expert buffers are laid out as [expert][row]; with at
        // most `trunk_rows` rows per step the largest union of routed
        // experts is `trunk_rows * k + 1` (last = shared) and its rows are
        // `trunk_rows`.
        let n_u_max = c.num_experts_per_tok * trunk_rows + 1;
        // The MTP scratch is only ever read when the draft head is present, so
        // it sizes to zero otherwise (multiply by 0).
        let use_mtp = usize::from(mtp.is_some() && options.drafts > 0);
        // Zero-length MTL buffers are invalid; they are never read when the
        // head is absent, so floor the gated sizes to a byte.
        let mtp_logits = (trunk_rows * vocab * 4 * use_mtp).max(1);
        let mtp_logits2 = (vocab * 4 * use_mtp).max(1);
        let mtp_hyper = (trunk_rows * hh * 4 * use_mtp).max(1);
        let scratch = Scratch {
            ids: ctx.new_buffer(64 * 4)?,
            e: ctx.new_buffer(trunk_rows * h * 4)?,
            hyper: ctx.new_buffer(trunk_rows * hh * 4)?,
            hc: hc_bufs()?,
            la: hc_bufs()?,
            mix_out: ctx.new_buffer(trunk_rows * h * 4)?,
            qg: ctx.new_buffer(trunk_rows * c.num_attention_heads * c.head_dim * 2 * 4)?,
            k: ctx.new_buffer(trunk_rows * kv_row * 4)?,
            v: ctx.new_buffer(trunk_rows * kv_row * 4)?,
            attn_out: ctx.new_buffer(trunk_rows * c.num_attention_heads * c.head_dim * 4)?,
            attn_parts: ctx.new_buffer(c.num_attention_heads * max_blk * (2 + c.head_dim) * 4)?,
            qkv: ctx.new_buffer(trunk_rows * conv_dim * 4)?,
            z: ctx.new_buffer(trunk_rows * v_dim * 4)?,
            a: ctx.new_buffer(trunk_rows * c.linear_num_value_heads * 4)?,
            b: ctx.new_buffer(trunk_rows * c.linear_num_value_heads * 4)?,
            kqn: ctx
                .new_buffer(trunk_rows * 2 * c.linear_num_key_heads * c.linear_key_head_dim * 4)?,
            gbuf: ctx.new_buffer(trunk_rows * c.linear_num_value_heads * 2 * 4)?,
            delta_y: ctx.new_buffer(trunk_rows * v_dim * 4)?,
            router: ctx.new_buffer(trunk_rows * c.num_experts * 4)?,
            topk_idx: ctx.new_buffer(trunk_rows * c.num_experts_per_tok * 4)?,
            topk_w: ctx.new_buffer(trunk_rows * c.num_experts_per_tok * 4)?,
            la_router: ctx.new_buffer(trunk_rows * c.num_experts * 4)?,
            la_idx: ctx.new_buffer(trunk_rows * 32 * 4)?,
            la_w: ctx.new_buffer(trunk_rows * 32 * 4)?,
            gate_e: ctx.new_buffer(n_u_max * trunk_rows * 2 * inter * 4)?,
            hx: half_set(n_u_max * trunk_rows, inter)?,
            y_e: ctx.new_buffer(n_u_max * trunk_rows * h * 4)?,
            moe_out: ctx.new_buffer(trunk_rows * h * 4)?,
            ple_key: ctx.new_buffer(trunk_rows * hh * 4)?,
            ple_keyn: ctx.new_buffer(trunk_rows * hh * 4)?,
            ple_value: ctx.new_buffer(trunk_rows * h * 4)?,
            ple_query: ctx.new_buffer(trunk_rows * hh * 4)?,
            ple_gated: ctx.new_buffer(trunk_rows * hh * 4)?,
            ple_gvn: ctx.new_buffer(trunk_rows * hh * 4)?,
            ple_out: ctx.new_buffer(trunk_rows * hh * 4)?,
            logits: ctx.new_buffer(trunk_rows * vocab * 4)?,
            mtp_logits: ctx.new_buffer(mtp_logits)?,
            mtp_logits2: ctx.new_buffer(mtp_logits2)?,
            partials: ctx.new_buffer(ARGMAX_TGS * 8)?,
            mtp_hyper: ctx.new_buffer(mtp_hyper)?,
            fe: ctx.new_buffer(trunk_rows * h * 4)?,
            fh: ctx.new_buffer(trunk_rows * hh * 4)?,
            iqk: ctx.new_buffer(trunk_rows * (c.indexer_n_heads + 1) * c.indexer_head_dim * 4)?,
            iq: ctx.new_buffer(trunk_rows * c.indexer_n_heads * c.indexer_head_dim * 4)?,
            bscore: ctx.new_buffer(trunk_rows * (max_t / c.indexer_compress_ratio + 1) * 4)?,
            vis: ctx.new_buffer(trunk_rows * (c.indexer_budget + c.indexer_compress_ratio) * 4)?,
            nvis: ctx.new_buffer(trunk_rows * 4)?,
            vmask: ctx
                .new_buffer(trunk_rows * (max_t / c.indexer_compress_ratio + 1).div_ceil(32) * 4)?,
        };
        let n_records = p.manifest.experts.layers * p.manifest.experts.experts;
        let stride = p.manifest.experts.record_stride as usize;
        // `wmap` is `[layer][MAX_NB][SLOT_STRIDE]`: the combine kernel strides
        // it by the compile-time MAX_NB, so it stays at the full width.
        let n_rows = c.num_hidden_layers + 1;
        let slot_tab = ctx.new_buffer(n_rows * SLOT_STRIDE * 8)?;
        let wmap = ctx.new_buffer(n_rows * MAX_NB * SLOT_STRIDE * 4)?;
        let ring = ctx.new_buffer(prefill::RING * stride)?;

        // A capped diagnostic trunk does not retain the normal adjacent-pass layout.
        let (phase_timer, gpu_timing) = super::phases::PhaseTimer::initialize(
            &ctx,
            p.manifest.experts.layers,
            layer_cap() >= layers.len(),
        );
        let mut activity = ExpertActivity::new(&p.manifest.experts);
        activity.gpu_timestamps_available = phase_timer.is_some();
        activity.gpu_timing = gpu_timing;

        let prefill_bytes =
            prefill::allocation::scratch_bytes(c, max_t, prefill_rows.min(max_t).max(1), false)?;
        let memory = budget::PoolMemory {
            device: ctx.device.recommendedMaxWorkingSetSize() as usize,
            fixed: ctx.device.currentAllocatedSize(),
            host: reserve,
            allocation_limit,
            prefill: prefill_bytes,
        };
        let mut pool_bytes = memory.bytes(options.pool_gb)?;
        let shrink = matches!(options.pool_gb, PoolBudget::Max);

        // CHERENKOV_POOL=set puts the pool in a residency set over the
        // file mapping (page cache as a second tier); the default copies
        // records into a wired buffer.
        let copy = std::env::var("CHERENKOV_POOL").as_deref() != Ok("set");
        // Only one low-bit layout is attached; --miss-experts can differ
        // from the resident precision only when the latter is four bits.
        let all_bits = options.experts;
        let miss_bits = options.miss_bits();

        anyhow::ensure!(
            copy || (all_bits == 4 && miss_bits == 4),
            "the residency-set pool supports only 4-bit experts"
        );

        let low_bits = if all_bits == 3 || all_bits == 2 {
            all_bits
        } else {
            miss_bits
        };
        let all_low_bits = all_bits == 3 || all_bits == 2;
        let mut low_bit_store = None;

        if (low_bits == 3 || low_bits == 2) && copy {
            // Build the store from experts.bin if it is missing, the wrong
            // size, or in an older layout. This pass over the 4-bit source
            // builds only the selected precision; other cached stores stay.
            let l = crate::qwen4_exp::lowbit::ensure_with_policy(
                &p.dir,
                &p.manifest.experts,
                low_bits,
                options.repack,
                options.build_missing_store,
            )?;
            let low_path = p.dir.join(format!("experts{low_bits}.bin"));
            let low_file = std::fs::File::open(&low_path)
                .with_context(|| format!("opening {}", low_path.display()))?;
            let st = l;

            anyhow::ensure!(
                st.stride > 0 && st.stride <= stride,
                "{low_bits}-bit record stride {} does not fit a 4-bit slot",
                st.stride
            );

            {
                use std::os::unix::io::AsRawFd as _;

                unsafe { libc::fcntl(low_file.as_raw_fd(), libc::F_NOCACHE, 1) };
            }

            low_bit_store = Some((st, low_file, low_path));
        }

        // Every record low-bit means a smaller slot and so more of them.
        let slot_stride = match (&low_bit_store, all_low_bits) {
            (Some((st, _, _)), true) => st.stride,
            _ => stride,
        };
        let mut res = loop {
            let minimum = 64.min(n_records);
            let slots = (pool_bytes / slot_stride).min(n_records);

            ensure!(
                slots >= minimum,
                "expert pool budget cannot hold {minimum} records"
            );

            match residency::Pool::new(
                &ctx,
                p.experts.as_ptr(),
                stride,
                slot_stride,
                n_records,
                slots,
                copy,
            ) {
                Ok(res) => break res,
                Err(_) if shrink && pool_bytes > BYTES_PER_GB / 2 => {
                    pool_bytes -= BYTES_PER_GB / 2;
                }
                Err(e) => {
                    return Err(e.context(format!("allocating a {pool_bytes}-byte expert pool")));
                }
            }
        };
        // Misses read through the page cache (their pages are what the
        // residency set pins; what it drops stays cached while memory
        // allows); the prefill ring streams past the cache.
        let pool_file = std::fs::File::open(p.dir.join("experts.bin"))?;
        let pool_file_nocache = std::fs::File::open(p.dir.join("experts.bin"))?;
        let low_bit_store = low_bit_store.map(|(st, low_file, low_path)| {
            res.set_low_bit_store(
                low_file,
                st.stride,
                if all_low_bits {
                    if st.bits == 2 { 2 } else { 1 }
                } else {
                    0
                },
            );
            eprintln!(
                "{}-bit store ({}): {} bytes per record, {:.0}% of 4-bit, for {}",
                st.bits,
                low_path.display(),
                st.stride,
                100.0 * st.stride as f64 / stride as f64,
                if all_low_bits {
                    "every record"
                } else {
                    "synchronous misses"
                }
            );

            st
        });

        {
            use std::os::unix::io::AsRawFd as _;

            unsafe { libc::fcntl(pool_file_nocache.as_raw_fd(), libc::F_NOCACHE, 1) };
        }

        let event_res = {
            use objc2_metal::MTLDevice as _;

            ctx.device.newSharedEvent().context("shared event")?
        };
        let (event, event_cpu) = {
            use objc2_metal::MTLDevice as _;

            (
                ctx.device.newSharedEvent().context("shared event")?,
                ctx.device.newSharedEvent().context("shared event")?,
            )
        };

        Ok(Gpu {
            embed: q(&format!("{m}.embed_tokens"))?,
            final_mixer: hc(&format!("{m}.hyper_connection_mixer"), false)?,
            lm_head: q("language_model.lm_head")?,
            p,
            ctx,
            pipes,
            dense,
            res,
            activity,
            read_tracker: super::activity::reads::ReadTracker::new(p.manifest.experts.layers),
            activity_started: std::time::Instant::now(),
            phase_timer,
            pool_file,
            pool_file_nocache,
            low_bit_store,
            all_low_bits,
            ngram_file: std::fs::File::open(p.dir.join("ngram.bin"))?,
            ngram_prefetch: std::cell::RefCell::new(None),
            step_no: 0,
            pending: None,
            event,
            event_base: 0,
            event_cpu,
            event_res,
            event_cpu_base: 0,
            slot_tab,
            wmap,
            layers,
            mtp,
            scratch,
            max_t,
            trunk_rows,
            pos: 0,
            tokens: Vec::new(),
            batch_pos: 0,
            batch_nb: 0,
            batch_snap: false,
            mtp_len: 0,
            last_experts: Vec::new(),
            expert_history: Vec::new(),
            route_history: Vec::new(),
            dump_states: std::env::var_os("CHERENKOV_DUMP_STATES").is_some(),
            state_history: Vec::new(),
            last_states: Vec::new(),
            last_routes: Vec::new(),
            last_route_w: Vec::new(),
            last_miss: Vec::new(),
            ngram_gather_s: std::cell::Cell::new(0.0),
            ngram_ms: Vec::new(),
            step_ms: Vec::new(),
            gpu_ms: Vec::new(),
            io_ms: Vec::new(),
            step_set_s: 0.0,
            step_read_s: 0.0,
            step_warm: 0,
            warm: Vec::new(),
            folded_mtp: None,
            set_ms: Vec::new(),
            read_ms: Vec::new(),
            misses: Vec::new(),
            miss_bytes: Vec::new(),
            step_misses: 0,
            step_miss_bytes: 0,
            lookahead_hit: Vec::new(),
            lookahead_issued: Vec::new(),
            la_log: Vec::new(),
            la_pending: Vec::new(),
            log_la: std::env::var_os("CHERENKOV_DUMP_LA").is_some(),
            cut_w: options.cut_weak,
            step_cut: 0,
            cut: Vec::new(),
            inflight: Vec::new(),
            lookahead: std::env::var("CHERENKOV_LOOKAHEAD").as_deref() != Ok("0"),
            // Blocking on the event measured ~30 ms/step slower (48 wake-ups)
            // with no thermal benefit; spinning is the default.
            spin_wait: std::env::var("CHERENKOV_SPIN").as_deref() != Ok("0"),
            fake_experts: std::env::var("CHERENKOV_FAKE").as_deref() == Ok("experts"),
            skip: std::env::var("CHERENKOV_SKIP")
                .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default(),
            dispatches: Vec::new(),
            gpu_idle_ms: Vec::new(),
            rows: Vec::new(),
            mtp_ms: Vec::new(),
            dispatch_count: std::cell::Cell::new(0),
            pf: None,
            ring,
            prefill_reserved_bytes: prefill_bytes,
            prefill_stats: Vec::new(),
        })
    }
}
