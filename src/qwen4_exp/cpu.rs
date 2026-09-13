//! CPU f32 reference forward for qwen4-exp (decode form, one token at a
//! time). Slow by design: it is the oracle for the Metal path and mirrors
//! the transformers implementation (modeling_qwen4_exp.py) step for step.
//!
//! Conventions verified on the checkpoint: RMSNorm weights are MLX-sanitized
//! (plain `w * norm(x)`, no `1 + w`); the DeltaNet output gate is a sigmoid
//! (`output_gate_type`); attention uses a per-head sigmoid output gate.

use super::packed::Packed;
use crate::nn;
use crate::quant::QLinear;
use anyhow::{Context, Result};
use half::bf16;
use rayon::prelude::*;

pub struct HcWeights<'a> {
    pub norm: &'a [bf16],
    pub down: QLinear<'a>,
    pub up: QLinear<'a>,
    pub inject: Option<QLinear<'a>>,
}

pub struct AttnWeights<'a> {
    pub q_proj: QLinear<'a>,
    pub k_proj: QLinear<'a>,
    pub v_proj: QLinear<'a>,
    pub o_proj: QLinear<'a>,
    pub q_norm: &'a [bf16],
    pub k_norm: &'a [bf16],
    pub index_qk: QLinear<'a>,
    pub index_q_norm: &'a [bf16],
    pub index_k_norm: &'a [bf16],
}

pub struct DeltaWeights<'a> {
    pub in_proj_qkv: QLinear<'a>,
    pub in_proj_z: QLinear<'a>,
    pub in_proj_a: QLinear<'a>,
    pub in_proj_b: QLinear<'a>,
    pub conv1d: &'a [bf16],
    pub a_log: &'a [bf16],
    pub dt_bias: &'a [bf16],
    pub norm: &'a [bf16],
    pub out_proj: QLinear<'a>,
}

pub enum Mixer<'a> {
    Attn(AttnWeights<'a>),
    Delta(DeltaWeights<'a>),
}

pub struct MoeWeights<'a> {
    /// Router, bf16 `[experts][hidden]`.
    pub router: &'a [bf16],
    pub shared_gate: &'a [bf16],
    pub shared_up: QLinear<'a>,
    pub shared_gate_proj: QLinear<'a>,
    pub shared_down: QLinear<'a>,
    pub record_layer: usize,
}

pub struct PleWeights<'a> {
    pub key_proj: QLinear<'a>,
    pub value_proj: QLinear<'a>,
    pub norm_key: &'a [bf16],
    pub norm_query: &'a [bf16],
    pub norm_conv: &'a [bf16],
    /// Depthwise dilated conv, `[channels][kernel]`.
    pub conv1d: &'a [bf16],
    pub kernel: usize,
    pub dilation: usize,
    pub multipliers: Vec<i64>,
    pub head_offsets: Vec<u64>,
    pub head_sizes: Vec<u64>,
}

pub struct Layer<'a> {
    pub attn_hc: HcWeights<'a>,
    pub mlp_hc: HcWeights<'a>,
    pub mixer: Mixer<'a>,
    pub moe: MoeWeights<'a>,
    pub ple: Option<PleWeights<'a>>,
}

/// The one-layer MTP draft head: folds the next token's embedding into the
/// trunk's wide residual, runs one attention + MoE block, collapses with
/// its own mixer and reuses the trunk's LM head.
pub struct MtpWeights<'a> {
    /// RMSNorm weights in the raw (1 + w) convention: the MLX converter
    /// left these two unshifted (checked against the bf16 checkpoint).
    pub enorm: &'a [bf16],
    pub hnorm: &'a [bf16],
    pub fc_e: QLinear<'a>,
    pub fc_h: QLinear<'a>,
    pub layer: Layer<'a>,
    pub mixer: HcWeights<'a>,
}

pub struct CpuModel<'a> {
    pub p: &'a Packed,
    pub embed: QLinear<'a>,
    pub layers: Vec<Layer<'a>>,
    pub final_mixer: HcWeights<'a>,
    pub lm_head: QLinear<'a>,
    pub mtp: Option<MtpWeights<'a>>,
}

#[derive(Default)]
pub struct KvCache {
    /// [t][kv_heads * head_dim]
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Indexer keys before norm and rope, `[t][index_head_dim]`.
    pub index_k: Vec<f32>,
    pub len: usize,
}

pub struct DeltaState {
    pub s: Vec<f32>,
    pub conv: Vec<f32>,
}

pub struct PleState {
    /// Ring of the last `span` normalized gated values, `[span][hc_hidden]`.
    pub hist: Vec<f32>,
    pub span: usize,
    pub filled: usize,
}

pub struct State {
    pub pos: usize,
    pub tokens: Vec<u32>,
    pub kv: Vec<KvCache>,
    pub delta: Vec<DeltaState>,
    pub ple: Option<PleState>,
    /// Wide residual after the last layer (before the final mixer), the
    /// MTP head's hidden input for the token just processed.
    pub last_hyper: Vec<f32>,
    /// The MTP layer's own KV cache, one entry per trunk position.
    pub mtp_kv: KvCache,
}

fn hc<'a>(p: &'a Packed, prefix: &str, inject: bool) -> Result<HcWeights<'a>> {
    Ok(HcWeights {
        norm: p.bf16(&format!("{prefix}.hc_norm.weight"))?,
        down: p.qlinear(&format!("{prefix}.input_mix_weight_down"))?,
        up: p.qlinear(&format!("{prefix}.input_mix_weight_up"))?,
        inject: if inject {
            Some(p.qlinear(&format!("{prefix}.block_inject_weight"))?)
        } else {
            None
        },
    })
}

impl<'a> CpuModel<'a> {
    pub fn load(p: &'a Packed) -> Result<Self> {
        let c = &p.cfg;
        let m = "language_model.model";
        let mut layers = Vec::with_capacity(c.num_hidden_layers);

        for l in 0..c.num_hidden_layers {
            let lp = format!("{m}.layers.{l}");

            layers.push(load_layer(
                p,
                &lp,
                c.is_linear(l),
                p.expert_layer(l),
                c.ple_layer() == Some(l),
            )?);
        }

        let mtp = if c.mtp_num_hidden_layers > 0 {
            Some(MtpWeights {
                enorm: p.bf16("mtp.pre_fc_norm_embedding.weight")?,
                hnorm: p.bf16("mtp.pre_fc_norm_hidden.weight")?,
                fc_e: p.qlinear("mtp.fc_embedding")?,
                fc_h: p.qlinear("mtp.fc_hidden")?,
                layer: load_layer(p, "mtp.layers.0", false, p.mtp_expert_layer(0)?, false)?,
                mixer: hc(p, "mtp.hyper_connection_mixer", false)?,
            })
        } else {
            None
        };

        Ok(CpuModel {
            p,
            embed: p.qlinear(&format!("{m}.embed_tokens"))?,
            layers,
            final_mixer: hc(p, &format!("{m}.hyper_connection_mixer"), false)?,
            lm_head: p.qlinear("language_model.lm_head")?,
            mtp,
        })
    }
}

fn load_layer<'a>(
    p: &'a Packed,
    lp: &str,
    linear: bool,
    record_layer: usize,
    with_ple: bool,
) -> Result<Layer<'a>> {
    let c = &p.cfg;
    let mixer = if linear {
        let d = format!("{lp}.linear_attn");

        Mixer::Delta(DeltaWeights {
            in_proj_qkv: p.qlinear(&format!("{d}.in_proj_qkv"))?,
            in_proj_z: p.qlinear(&format!("{d}.in_proj_z"))?,
            in_proj_a: p.qlinear(&format!("{d}.in_proj_a"))?,
            in_proj_b: p.qlinear(&format!("{d}.in_proj_b"))?,
            conv1d: p.bf16(&format!("{d}.conv1d.weight"))?,
            a_log: p.bf16(&format!("{d}.A_log"))?,
            dt_bias: p.bf16(&format!("{d}.dt_bias"))?,
            norm: p.bf16(&format!("{d}.norm.weight"))?,
            out_proj: p.qlinear(&format!("{d}.out_proj"))?,
        })
    } else {
        let a = format!("{lp}.self_attn");

        Mixer::Attn(AttnWeights {
            q_proj: p.qlinear(&format!("{a}.q_proj"))?,
            k_proj: p.qlinear(&format!("{a}.k_proj"))?,
            v_proj: p.qlinear(&format!("{a}.v_proj"))?,
            o_proj: p.qlinear(&format!("{a}.o_proj"))?,
            q_norm: p.bf16(&format!("{a}.q_norm.weight"))?,
            k_norm: p.bf16(&format!("{a}.k_norm.weight"))?,
            index_qk: p.qlinear(&format!("{a}.indexer.index_qk_proj"))?,
            index_q_norm: p.bf16(&format!("{a}.indexer.q_layernorm.weight"))?,
            index_k_norm: p.bf16(&format!("{a}.indexer.k_layernorm.weight"))?,
        })
    };
    let moe = MoeWeights {
        router: p.bf16(&format!("{lp}.mlp.gate.weight"))?,
        shared_gate: p.bf16(&format!("{lp}.mlp.shared_expert_gate.weight"))?,
        shared_gate_proj: p.qlinear(&format!("{lp}.mlp.shared_expert.gate_proj"))?,
        shared_up: p.qlinear(&format!("{lp}.mlp.shared_expert.up_proj"))?,
        shared_down: p.qlinear(&format!("{lp}.mlp.shared_expert.down_proj"))?,
        record_layer,
    };
    let ple = if with_ple {
        let pp = format!("{lp}.ple");
        let conv = p.bf16(&format!("{pp}.conv1d.weight"))?;
        let kernel = p.shape(&format!("{pp}.conv1d.weight"))?[2];
        let emb = format!("{pp}.ple_embedding");

        Some(PleWeights {
            key_proj: p.qlinear(&format!("{pp}.key_proj"))?,
            value_proj: p.qlinear(&format!("{pp}.value_proj"))?,
            norm_key: p.bf16(&format!("{pp}.norm_key.weight"))?,
            norm_query: p.bf16(&format!("{pp}.norm_query.weight"))?,
            norm_conv: p.bf16(&format!("{pp}.norm_conv.weight"))?,
            conv1d: conv,
            kernel,
            dilation: c.ngram_size,
            multipliers: p.i64s(&format!("{emb}.layer_multipliers"))?,
            head_offsets: p
                .i64s(&format!("{emb}.ngram_heads_offsets"))?
                .into_iter()
                .map(|v| v as u64)
                .collect(),
            head_sizes: p
                .i64s(&format!("{emb}.ngram_heads_vocab_sizes"))?
                .into_iter()
                .map(|v| v as u64)
                .collect(),
        })
    } else {
        None
    };

    Ok(Layer {
        attn_hc: hc(p, &format!("{lp}.attn_hyper_connection"), true)?,
        mlp_hc: hc(p, &format!("{lp}.mlp_hyper_connection"), true)?,
        mixer,
        moe,
        ple,
    })
}

impl<'a> CpuModel<'a> {
    pub fn new_state(&self) -> State {
        let c = &self.p.cfg;
        let mut kv = Vec::new();
        let mut delta = Vec::new();

        for l in 0..c.num_hidden_layers {
            if c.is_linear(l) {
                delta.push(DeltaState {
                    s: vec![
                        0.0;
                        c.linear_num_value_heads
                            * c.linear_key_head_dim
                            * c.linear_value_head_dim
                    ],
                    conv: vec![
                        0.0;
                        (2 * c.linear_num_key_heads * c.linear_key_head_dim
                            + c.linear_num_value_heads * c.linear_value_head_dim)
                            * (c.linear_conv_kernel_dim - 1)
                    ],
                });
            } else {
                kv.push(KvCache::default());
            }
        }

        let ple = c.ple_layer().map(|_| {
            let span = (c.ple_conv_kernel_size - 1) * c.ngram_size;

            PleState {
                hist: vec![0.0; span * c.hc_hidden()],
                span,
                filled: 0,
            }
        });

        State {
            pos: 0,
            tokens: Vec::new(),
            kv,
            delta,
            ple,
            last_hyper: Vec::new(),
            mtp_kv: KvCache::default(),
        }
    }

    /// MTP draft head at trunk position `pos`: pairs the trunk's wide
    /// residual for that position (`hyper_in`) with the embedding of the
    /// token that follows it, appends to the head's KV cache (which must
    /// hold exactly `pos` entries) and returns (logits for the token after
    /// `token`, the head's wide residual for chained drafting).
    pub fn mtp_forward(
        &self,
        token: u32,
        hyper_in: &[f32],
        pos: usize,
        state: &mut State,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let w = self.mtp.as_ref().context("no MTP head")?;
        let c = &self.p.cfg;
        let h = c.hidden_size;
        let hc = c.hc_count;
        let eps = c.rms_norm_eps as f32;

        anyhow::ensure!(hyper_in.len() == h * hc, "hyper input width");

        let mut e = vec![0.0f32; h];

        self.embed.dequant_row(token as usize, &mut e);

        let en = rms_norm_shift(&e, w.enorm, eps);
        let mut fe = vec![0.0f32; h];

        w.fc_e.matvec(&en, &mut fe);

        let mut hyper = vec![0.0f32; h * hc];
        let mut fh = vec![0.0f32; h];

        for g in 0..hc {
            let hn = rms_norm_shift(
                &hyper_in[g * h..(g + 1) * h],
                &w.hnorm[g * h..(g + 1) * h],
                eps,
            );

            w.fc_h.matvec(&hn, &mut fh);

            for i in 0..h {
                hyper[g * h + i] = fe[i] + fh[i];
            }
        }

        let layer = &w.layer;
        let (mixed, inj) = self.gated_residual(&layer.attn_hc, &hyper, eps);
        let out = match &layer.mixer {
            Mixer::Attn(a) => self.attention(a, &mixed, &mut state.mtp_kv, pos)?,
            Mixer::Delta(_) => anyhow::bail!("MTP layer is expected to be full attention"),
        };

        inject(&mut hyper, &out, &inj);

        let (mixed, inj) = self.gated_residual(&layer.mlp_hc, &hyper, eps);
        let out = self.moe(&layer.moe, &mixed);

        inject(&mut hyper, &out, &inj);

        let (mixed, _) = self.gated_residual(&w.mixer, &hyper, eps);
        let mut logits = vec![0.0f32; self.lm_head.out_dim];

        self.lm_head.matvec(&mixed, &mut logits);

        Ok((logits, hyper))
    }

    /// Forward one token at `state.pos`; returns logits.
    pub fn forward_token(&self, token: u32, state: &mut State) -> Result<Vec<f32>> {
        let c = &self.p.cfg;
        let h = c.hidden_size;
        let hc = c.hc_count;
        let hh = h * hc;
        let eps = c.rms_norm_eps as f32;

        let mut e = vec![0.0f32; h];

        self.embed.dequant_row(token as usize, &mut e);

        let mut hyper = vec![0.0f32; hh];

        for g in 0..hc {
            hyper[g * h..(g + 1) * h].copy_from_slice(&e);
        }

        state.tokens.push(token);

        let mut kv_idx = 0;
        let mut delta_idx = 0;

        for layer in &self.layers {
            if let Some(ple) = &layer.ple {
                let add = self.ple(ple, &hyper, state)?;

                for (x, a) in hyper.iter_mut().zip(&add) {
                    *x += a;
                }
            }

            let (mixed, inj) = self.gated_residual(&layer.attn_hc, &hyper, eps);
            let out = match &layer.mixer {
                Mixer::Attn(a) => {
                    let o = self.attention(a, &mixed, &mut state.kv[kv_idx], state.pos)?;
                    kv_idx += 1;

                    o
                }
                Mixer::Delta(d) => {
                    let o = self.deltanet(d, &mixed, &mut state.delta[delta_idx]);
                    delta_idx += 1;

                    o
                }
            };

            inject(&mut hyper, &out, &inj);

            let (mixed, inj) = self.gated_residual(&layer.mlp_hc, &hyper, eps);
            let out = self.moe(&layer.moe, &mixed);

            inject(&mut hyper, &out, &inj);
        }

        state.pos += 1;

        state.last_hyper.clone_from(&hyper);

        let (mixed, _) = self.gated_residual(&self.final_mixer, &hyper, eps);
        let mut logits = vec![0.0f32; self.lm_head.out_dim];

        self.lm_head.matvec(&mixed, &mut logits);

        Ok(logits)
    }

    /// Gated residual read: returns the mixed hidden input for the block and
    /// the per-stream injection weights (empty for the final mixer).
    fn gated_residual(&self, w: &HcWeights, hyper: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
        let c = &self.p.cfg;
        let h = c.hidden_size;
        let hc = c.hc_count;
        let normed = grouped_rms_norm(hyper, w.norm, h, eps);
        let mut d = vec![0.0f32; w.down.out_dim];

        w.down.matvec(&normed, &mut d);

        for v in d.iter_mut() {
            *v = nn::silu(*v / hc as f32);
        }

        let mut u = vec![0.0f32; w.up.out_dim];

        w.up.matvec(&d, &mut u);

        let mut mixed = vec![0.0f32; h];

        for g in 0..hc {
            for i in 0..h {
                mixed[i] += nn::sigmoid(u[g * h + i]) * normed[g * h + i];
            }
        }

        for v in mixed.iter_mut() {
            *v /= hc as f32;
        }

        let inj = match &w.inject {
            Some(q) => {
                let mut r = vec![0.0f32; q.out_dim];

                q.matvec(&normed, &mut r);

                r.iter().map(|v| 2.0 * nn::sigmoid(v / hc as f32)).collect()
            }
            None => Vec::new(),
        };

        (mixed, inj)
    }

    fn moe(&self, w: &MoeWeights, x: &[f32]) -> Vec<f32> {
        let c = &self.p.cfg;
        let h = c.hidden_size;
        // Router: softmax over all experts in f32, top-k, renormalize.
        let mut logits = bf16_matvec(w.router, c.num_experts, h, x);

        nn::softmax(&mut logits);

        let mut idx: Vec<usize> = (0..c.num_experts).collect();

        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));

        let top: Vec<usize> = idx[..c.num_experts_per_tok].to_vec();
        let sum: f32 = top.iter().map(|&i| logits[i]).sum();
        let mut out = vec![0.0f32; h];

        for &ei in &top {
            let weight = if c.norm_topk_prob {
                logits[ei] / sum
            } else {
                logits[ei]
            };
            let ex = self.p.expert(w.record_layer, ei);
            let y = mlp(&ex.gate, &ex.up, &ex.down, x);

            for (o, v) in out.iter_mut().zip(&y) {
                *o += weight * v;
            }
        }

        let shared = mlp(&w.shared_gate_proj, &w.shared_up, &w.shared_down, x);
        let g = nn::sigmoid(bf16_dot(w.shared_gate, x));

        for (o, v) in out.iter_mut().zip(&shared) {
            *o += g * v;
        }

        out
    }

    fn attention(
        &self,
        w: &AttnWeights,
        x: &[f32],
        cache: &mut KvCache,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.p.cfg;
        let (nh, nkv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let rot = (hd as f64 * c.partial_rotary_factor) as usize;
        let theta = c.rope_parameters.rope_theta as f32;
        let eps = c.rms_norm_eps as f32;
        let kv_row = nkv * hd;

        // Indexer: cache the raw key for this position, then score blocks.
        let ihd = c.indexer_head_dim;
        let inh = c.indexer_n_heads;
        let mut qk = vec![0.0f32; w.index_qk.out_dim];

        w.index_qk.matvec(x, &mut qk);

        let (iq, ik) = qk.split_at(inh * ihd);

        cache.index_k.extend_from_slice(ik);

        let mut qg = vec![0.0f32; w.q_proj.out_dim];
        let mut k = vec![0.0f32; w.k_proj.out_dim];
        let mut v = vec![0.0f32; w.v_proj.out_dim];

        w.q_proj.matvec(x, &mut qg);
        w.k_proj.matvec(x, &mut k);
        w.v_proj.matvec(x, &mut v);

        for hi in 0..nh {
            let q = &mut qg[hi * 2 * hd..hi * 2 * hd + hd];

            nn::rms_norm(q, w.q_norm, eps);
            rope_partial(q, pos, rot, theta);
        }

        for hi in 0..nkv {
            let kh = &mut k[hi * hd..(hi + 1) * hd];

            nn::rms_norm(kh, w.k_norm, eps);
            rope_partial(kh, pos, rot, theta);
        }

        cache.k.extend_from_slice(&k);
        cache.v.extend_from_slice(&v);

        cache.len += 1;
        let t_len = cache.len;

        anyhow::ensure!(t_len == pos + 1, "attention cache out of step");

        // Sparse selection (QSA): all tokens while the context fits the budget.
        let selected = self.select_tokens(w, iq, cache, pos);

        let scale = (hd as f32).powf(-0.5);
        let group = nh / nkv;
        let mut out = vec![0.0f32; nh * hd];
        let mut scores = vec![0.0f32; selected.len()];

        for hi in 0..nh {
            let hk = hi / group;
            let q = &qg[hi * 2 * hd..hi * 2 * hd + hd];

            for (si, &ti) in selected.iter().enumerate() {
                let kt = &cache.k[ti * kv_row + hk * hd..ti * kv_row + (hk + 1) * hd];
                scores[si] = scale * q.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>();
            }

            nn::softmax(&mut scores);

            let oh = &mut out[hi * hd..(hi + 1) * hd];

            for (si, &ti) in selected.iter().enumerate() {
                let vt = &cache.v[ti * kv_row + hk * hd..ti * kv_row + (hk + 1) * hd];
                let p = scores[si];

                for d in 0..hd {
                    oh[d] += p * vt[d];
                }
            }

            let gate = &qg[hi * 2 * hd + hd..(hi + 1) * 2 * hd];

            for d in 0..hd {
                oh[d] *= nn::sigmoid(gate[d]);
            }
        }

        let mut o = vec![0.0f32; w.o_proj.out_dim];

        w.o_proj.matvec(&out, &mut o);

        Ok(o)
    }

    /// QSA indexer for one query at `pos` over the causal prefix: blocks of
    /// `compress_ratio` tokens scored by relu(q . pooled_key) summed over
    /// index heads; the top `budget / ratio` blocks plus the incomplete tail
    /// are visible. Returns visible token indices in ascending order.
    fn select_tokens(
        &self,
        w: &AttnWeights,
        iq: &[f32],
        cache: &KvCache,
        pos: usize,
    ) -> Vec<usize> {
        let c = &self.p.cfg;
        let t_len = pos + 1;
        let ratio = c.indexer_compress_ratio;
        let blocks = t_len / ratio;
        let topk = c.indexer_budget / ratio;

        if blocks <= topk {
            return (0..t_len).collect();
        }

        let ihd = c.indexer_head_dim;
        let inh = c.indexer_n_heads;
        let rot = (c.head_dim as f64 * c.partial_rotary_factor) as usize;
        let theta = c.rope_parameters.rope_theta as f32;
        let eps = c.rms_norm_eps as f32;
        let mut q = iq.to_vec();

        for hi in 0..inh {
            let qh = &mut q[hi * ihd..(hi + 1) * ihd];

            nn::rms_norm(qh, w.index_q_norm, eps);
            rope_partial(qh, pos, rot, theta);
        }

        let mut pooled = vec![0.0f32; ihd];
        let mut block_scores: Vec<(f32, usize)> = Vec::with_capacity(blocks);

        for b in 0..blocks {
            pooled.fill(0.0);

            for t in b * ratio..(b + 1) * ratio {
                for (d, value) in pooled.iter_mut().enumerate() {
                    *value += cache.index_k[t * ihd + d];
                }
            }

            for v in pooled.iter_mut() {
                *v /= ratio as f32;
            }

            nn::rms_norm(&mut pooled, w.index_k_norm, eps);
            rope_partial(&mut pooled, b * ratio, rot, theta);

            let mut s = 0.0f32;

            for hi in 0..inh {
                let qh = &q[hi * ihd..(hi + 1) * ihd];
                let dot: f32 = qh.iter().zip(&pooled).map(|(a, b)| a * b).sum();
                s += dot.max(0.0);
            }

            block_scores.push((s / (ihd as f32).sqrt(), b));
        }

        block_scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        let mut sel: Vec<usize> = block_scores[..topk]
            .iter()
            .flat_map(|&(_, b)| b * ratio..(b + 1) * ratio)
            .collect();

        sel.extend(blocks * ratio..t_len);
        sel.sort_unstable();

        sel
    }

    /// Gated DeltaNet, recurrent form, f32 state; sigmoid output gate.
    fn deltanet(&self, w: &DeltaWeights, x: &[f32], state: &mut DeltaState) -> Vec<f32> {
        let c = &self.p.cfg;
        // n* counts heads; d* counts lanes within a head. Value heads
        // share key/query heads in contiguous groups of nv / nk.
        let (nk, nv) = (c.linear_num_key_heads, c.linear_num_value_heads);
        let (dk, dv) = (c.linear_key_head_dim, c.linear_value_head_dim);
        let qk_dim = nk * dk;
        let v_dim = nv * dv;
        let conv_dim = 2 * qk_dim + v_dim;
        let ck = c.linear_conv_kernel_dim;

        let mut qkv = vec![0.0f32; conv_dim];
        let mut gate_input = vec![0.0f32; v_dim];
        let mut decay_input = vec![0.0f32; nv];
        let mut beta_input = vec![0.0f32; nv];

        w.in_proj_qkv.matvec(x, &mut qkv);
        w.in_proj_z.matvec(x, &mut gate_input);
        w.in_proj_a.matvec(x, &mut decay_input);
        w.in_proj_b.matvec(x, &mut beta_input);

        causal_conv(&mut qkv, w.conv1d, &mut state.conv, ck);

        // Normalize q and k per key head; only q receives 1/sqrt(dk).
        let mut q = qkv[..qk_dim].to_vec();
        let mut keys = qkv[qk_dim..2 * qk_dim].to_vec();
        let v = &qkv[2 * qk_dim..];
        let qscale = (dk as f32).powf(-0.5);

        for hi in 0..nk {
            let qh = &mut q[hi * dk..(hi + 1) * dk];

            nn::l2_norm(qh, 1e-6);

            for qv in qh.iter_mut() {
                *qv *= qscale;
            }

            nn::l2_norm(&mut keys[hi * dk..(hi + 1) * dk], 1e-6);
        }

        let group = nv / nk;
        let mut y = vec![0.0f32; v_dim];
        let mut kv_mem = vec![0.0f32; dv];
        let mut delta = vec![0.0f32; dv];
        let sigmoid_gate = c.output_gate_type == "sigmoid";

        for hv in 0..nv {
            let hk = hv / group;
            let qh = &q[hk * dk..(hk + 1) * dk];
            let kh = &keys[hk * dk..(hk + 1) * dk];
            let beta = nn::sigmoid(beta_input[hv]);
            let g = -(w.a_log[hv].to_f32().exp())
                * nn::softplus(decay_input[hv] + w.dt_bias[hv].to_f32());
            let decay = g.exp();
            // CPU state is [key_lane][value_lane]; delta_scan2 on Metal
            // stores its transpose so each simdgroup can own one value lane.
            let s = &mut state.s[hv * dk * dv..(hv + 1) * dk * dv];

            // S <- decay*S, then read the old prediction S^T k.
            kv_mem.fill(0.0);

            for ik in 0..dk {
                let row = &mut s[ik * dv..(ik + 1) * dv];
                let kw = kh[ik];

                for iv in 0..dv {
                    row[iv] *= decay;
                    kv_mem[iv] += row[iv] * kw;
                }
            }

            // The correction is beta*(v - S^T k).
            for iv in 0..dv {
                delta[iv] = (v[hv * dv + iv] - kv_mem[iv]) * beta;
            }

            // Rank-one update S <- S + k*delta^T, followed by y = S^T q.
            // Preserve these loop orders when comparing with the GPU oracle.
            let yh = &mut y[hv * dv..(hv + 1) * dv];

            for ik in 0..dk {
                let row = &mut s[ik * dv..(ik + 1) * dv];
                let kw = kh[ik];
                let qw = qh[ik];

                for iv in 0..dv {
                    row[iv] += kw * delta[iv];
                    yh[iv] += row[iv] * qw;
                }
            }

            nn::rms_norm(yh, w.norm, 1e-6);

            for iv in 0..dv {
                let zz = gate_input[hv * dv + iv];
                yh[iv] *= if sigmoid_gate {
                    nn::sigmoid(zz)
                } else {
                    nn::silu(zz)
                };
            }
        }

        let mut o = vec![0.0f32; w.out_proj.out_dim];

        w.out_proj.matvec(&y, &mut o);

        o
    }

    fn ngram_ids(&self, w: &PleWeights, tokens: &[u32]) -> Vec<u64> {
        Self::ngram_ids_from(
            &self.p.cfg,
            &w.multipliers,
            &w.head_offsets,
            &w.head_sizes,
            tokens,
        )
    }

    /// Hashed n-gram ids for the current (last) token: bigram heads then
    /// trigram heads. A token before the current segment (past an EOS, or
    /// before the start) reads as EOS. Multiplication wraps like torch int64.
    pub fn ngram_ids_from(
        c: &super::Qwen4ExpConfig,
        multipliers: &[i64],
        head_offsets: &[u64],
        head_sizes: &[u64],
        tokens: &[u32],
    ) -> Vec<u64> {
        let eos = c.eos_token_id;
        let t = tokens.len() - 1;
        let shifted: Vec<i64> = (0..c.ngram_size)
            .map(|k| {
                if k == 0 {
                    return tokens[t] as i64;
                }

                if t < k || tokens[t - k..t].contains(&eos) {
                    eos as i64
                } else {
                    tokens[t - k] as i64
                }
            })
            .collect();
        let mut ids = Vec::with_capacity(head_offsets.len());

        for ngram in 2..=c.ngram_size {
            let mut mixed = shifted[0].wrapping_mul(multipliers[0]);

            for pos in 1..ngram {
                mixed ^= shifted[pos].wrapping_mul(multipliers[pos]);
            }

            let start = (ngram - 2) * c.heads_per_ngram;

            for head in start..start + c.heads_per_ngram {
                let size = head_sizes[head] as i64;

                ids.push(mixed.rem_euclid(size) as u64 + head_offsets[head]);
            }
        }

        ids
    }

    /// PLE block: n-gram embedding, per-stream key/query gating, value,
    /// dilated depthwise conv with SiLU. Returns the hc_hidden-wide addend.
    fn ple(&self, w: &PleWeights, hyper: &[f32], state: &mut State) -> Result<Vec<f32>> {
        let c = &self.p.cfg;
        let h = c.hidden_size;
        let hc = c.hc_count;
        let hh = h * hc;
        let eps = c.rms_norm_eps as f32;
        let dim = self.p.manifest.ngram.dim;
        let ids = self.ngram_ids(w, &state.tokens);

        anyhow::ensure!(
            ids.len() * dim == c.ple_embed_dim,
            "n-gram head layout mismatch"
        );

        let mut e = vec![0.0f32; c.ple_embed_dim];

        for (hi, &id) in ids.iter().enumerate() {
            self.p.ngram_row(id, &mut e[hi * dim..(hi + 1) * dim]);
        }

        let mut key = vec![0.0f32; w.key_proj.out_dim];

        w.key_proj.matvec(&e, &mut key);

        let key = grouped_rms_norm(&key, w.norm_key, h, eps);
        let mut value = vec![0.0f32; w.value_proj.out_dim];

        w.value_proj.matvec(&e, &mut value);

        let query = grouped_rms_norm(hyper, w.norm_query, h, eps);
        let mut gated = vec![0.0f32; hh];

        for g in 0..hc {
            let dot: f32 = key[g * h..(g + 1) * h]
                .iter()
                .zip(&query[g * h..(g + 1) * h])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / (h as f32).sqrt();
            let gate = dot.abs().max(1e-6).sqrt() * dot.signum();
            let s = nn::sigmoid(gate);

            for i in 0..h {
                gated[g * h + i] = s * value[i];
            }
        }

        let gvn = grouped_rms_norm(&gated, w.norm_conv, h, eps);
        // Dilated causal depthwise conv over time: tap k reads the value at
        // t - dilation * (kernel - 1 - k); tap kernel-1 is the current value.
        let ps = state.ple.as_mut().context("PLE state missing")?;
        let span = ps.span;
        let mut out = gated;

        for ch in 0..hh {
            let wrow = &w.conv1d[ch * w.kernel..(ch + 1) * w.kernel];
            let mut acc = wrow[w.kernel - 1].to_f32() * gvn[ch];

            for (k, weight) in wrow.iter().take(w.kernel - 1).enumerate() {
                let back = w.dilation * (w.kernel - 1 - k); // 1..=span

                if back <= ps.filled {
                    let slot = (ps.filled - back) % span;
                    acc += weight.to_f32() * ps.hist[slot * hh + ch];
                }
            }

            out[ch] += nn::silu(acc);
        }

        let slot = ps.filled % span;

        ps.hist[slot * hh..(slot + 1) * hh].copy_from_slice(&gvn);

        ps.filled += 1;

        Ok(out)
    }
}

fn inject(hyper: &mut [f32], out: &[f32], inj: &[f32]) {
    let h = out.len();

    for (g, &wg) in inj.iter().enumerate() {
        for i in 0..h {
            hyper[g * h + i] += out[i] * wg;
        }
    }
}

fn mlp(gate: &QLinear, up: &QLinear, down: &QLinear, x: &[f32]) -> Vec<f32> {
    let mut g = vec![0.0f32; gate.out_dim];
    let mut u = vec![0.0f32; up.out_dim];

    gate.matvec(x, &mut g);
    up.matvec(x, &mut u);

    for (gv, uv) in g.iter_mut().zip(&u) {
        *gv = nn::silu(*gv) * uv;
    }

    let mut out = vec![0.0f32; down.out_dim];

    down.matvec(&g, &mut out);

    out
}

/// RMSNorm with the raw HF weight convention: x * inv * (1 + w).
fn rms_norm_shift(x: &[f32], w: &[bf16], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();

    x.iter()
        .zip(w)
        .map(|(v, w)| v * inv * (1.0 + w.to_f32()))
        .collect()
}

/// RMSNorm applied independently to each `group`-wide slice, with a full
/// width weight vector.
fn grouped_rms_norm(x: &[f32], w: &[bf16], group: usize, eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), w.len());

    let mut out = vec![0.0f32; x.len()];

    for (g, chunk) in x.chunks_exact(group).enumerate() {
        let ms = chunk.iter().map(|v| v * v).sum::<f32>() / group as f32;
        let inv = 1.0 / (ms + eps).sqrt();

        for (i, v) in chunk.iter().enumerate() {
            out[g * group + i] = v * inv * w[g * group + i].to_f32();
        }
    }

    out
}

fn bf16_matvec(w: &[bf16], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);

    (0..rows)
        .into_par_iter()
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a.to_f32() * b)
                .sum()
        })
        .collect()
}

fn bf16_dot(w: &[bf16], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(a, b)| a.to_f32() * b).sum()
}

/// Partial RoPE over the first `rot` dims with half-split pairing
/// (rotate_half): pair (j, j + rot/2) uses inv_freq theta^(-2j/rot).
fn rope_partial(x: &mut [f32], pos: usize, rot: usize, theta: f32) {
    let half = rot / 2;

    for j in 0..half {
        let inv_freq = theta.powf(-(2.0 * j as f32) / rot as f32);
        let angle = pos as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let a = x[j];
        let b = x[j + half];
        x[j] = a * cos - b * sin;
        x[j + half] = b * cos + a * sin;
    }
}

/// Causal depthwise convolution over packed [q | k | v] channels.
/// History is oldest-first; preserve accumulation order for the CPU oracle.
fn causal_conv(qkv: &mut [f32], weights: &[bf16], history: &mut [f32], ck: usize) {
    let km1 = ck - 1;

    for (ch, value) in qkv.iter_mut().enumerate() {
        let wrow = &weights[ch * ck..(ch + 1) * ck];
        let hist = &mut history[ch * km1..(ch + 1) * km1];
        let cur = *value;
        let mut acc = wrow[km1].to_f32() * cur;

        for j in 0..km1 {
            acc += wrow[j].to_f32() * hist[j];
        }

        hist.rotate_left(1);

        hist[km1 - 1] = cur;
        *value = nn::silu(acc);
    }
}
