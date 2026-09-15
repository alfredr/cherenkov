use super::*;

// The QSA indexer kernels against a CPU re-implementation of
// cpu.rs's `select_tokens` on random data, and the top-k selection
// on scores with ties.

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

fn rms_norm(x: &mut [f32], w: &[bf16], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();

    for (v, wi) in x.iter_mut().zip(w) {
        *v = *v * inv * wi.to_f32();
    }
}

struct Geo {
    ihd: usize,
    inh: usize,
    ratio: usize,
    k: usize,
    rot: usize,
    theta: f32,
    eps: f32,
}

/// Visible tokens for one query, as cpu.rs computes them.
#[allow(clippy::too_many_arguments)]
fn select_ref(
    g: &Geo,
    iq_raw: &[f32],
    ikc: &[f32],
    pos: usize,
    wq: &[bf16],
    wk: &[bf16],
) -> Vec<u32> {
    let t_len = pos + 1;
    let blocks = t_len / g.ratio;

    if blocks <= g.k {
        return (0..t_len as u32).collect();
    }

    let mut q = iq_raw.to_vec();

    for h in 0..g.inh {
        let qh = &mut q[h * g.ihd..(h + 1) * g.ihd];

        rms_norm(qh, wq, g.eps);
        rope_partial(qh, pos, g.rot, g.theta);
    }

    let mut scores: Vec<(f32, usize)> = Vec::with_capacity(blocks);
    let mut pooled = vec![0.0f32; g.ihd];

    for b in 0..blocks {
        pooled.fill(0.0);

        for t in b * g.ratio..(b + 1) * g.ratio {
            for d in 0..g.ihd {
                pooled[d] += ikc[t * g.ihd + d];
            }
        }

        for v in pooled.iter_mut() {
            *v /= g.ratio as f32;
        }

        rms_norm(&mut pooled, wk, g.eps);
        rope_partial(&mut pooled, b * g.ratio, g.rot, g.theta);

        let mut s = 0.0f32;

        for h in 0..g.inh {
            let dot: f32 = q[h * g.ihd..(h + 1) * g.ihd]
                .iter()
                .zip(&pooled)
                .map(|(a, b)| a * b)
                .sum();
            s += dot.max(0.0);
        }

        scores.push((s / (g.ihd as f32).sqrt(), b));
    }

    scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    let mut sel: Vec<u32> = scores[..g.k]
        .iter()
        .flat_map(|&(_, b)| (b * g.ratio..(b + 1) * g.ratio).map(|t| t as u32))
        .collect();

    sel.extend((blocks * g.ratio..t_len).map(|t| t as u32));
    sel.sort_unstable();

    sel
}

type DispatchJob<'a> = (&'a Pso, Vec<(usize, &'a Buf)>, usize, usize, usize);

fn run(ctx: &MetalContext, jobs: &[DispatchJob<'_>], params: &IndexParams) {
    let cb = ctx.queue.commandBuffer().unwrap();
    let enc = cb.computeCommandEncoder().unwrap();

    for (pso, binds, pidx, grid, tg) in jobs {
        enc.setComputePipelineState(pso);

        for (i, b) in binds {
            unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, *i) };
        }

        set_bytes(&enc, *pidx, params);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: *grid,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: *tg,
                height: 1,
                depth: 1,
            },
        );
    }

    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

#[test]
fn index_pipeline_matches_cpu() {
    let g = Geo {
        ihd: 128,
        inh: 4,
        ratio: 4,
        k: 512,
        rot: 64,
        theta: 1e7,
        eps: 1e-6,
    };
    let (t_total, base_pos, nb) = (2300usize, 2200usize, 3usize);
    let qk_dim = (g.inh + 1) * g.ihd;
    let mut rng = Lcg(7);
    let mut ikc: Vec<f32> = (0..t_total * g.ihd).map(|_| rng.f()).collect();
    let iqk: Vec<f32> = (0..nb * qk_dim).map(|_| rng.f()).collect();

    for b in 0..nb {
        ikc[(base_pos + b) * g.ihd..(base_pos + b + 1) * g.ihd]
            .copy_from_slice(&iqk[b * qk_dim + g.inh * g.ihd..(b + 1) * qk_dim]);
    }

    let wq: Vec<bf16> = (0..g.ihd)
        .map(|_| bf16::from_f32(1.0 + 0.1 * rng.f()))
        .collect();
    let wk: Vec<bf16> = (0..g.ihd)
        .map(|_| bf16::from_f32(1.0 + 0.1 * rng.f()))
        .collect();

    let ctx = MetalContext::new().unwrap();
    let lib = ctx.compile_library(BATCH_MSL).unwrap();
    let pipes: Vec<Pso> = [
        "fn_index_append",
        "fn_index_blocks",
        "fn_index_q",
        "fn_index_score",
        "fn_index_select",
    ]
    .iter()
    .map(|n| ctx.pipeline(&lib, n).unwrap())
    .collect();
    let max_blocks = t_total / g.ratio + 1;
    let vis_stride = g.k * g.ratio + g.ratio;
    let iqk_b = upload(&ctx, &iqk);
    // The persistent index caches are half precision.
    let ikc_f16: Vec<half::f16> = ikc.iter().map(|&v| half::f16::from_f32(v)).collect();
    let ikc_b = upload::<half::f16>(&ctx, &ikc_f16);
    let blk_b = ctx.new_buffer(max_blocks * g.ihd * 2).unwrap();
    let wq_b = upload(&ctx, &wq);
    let wk_b = upload(&ctx, &wk);
    let iq_b = ctx.new_buffer(nb * g.inh * g.ihd * 4).unwrap();
    let score_b = ctx.new_buffer(nb * max_blocks * 4).unwrap();
    let vis_b = ctx.new_buffer(nb * vis_stride * 4).unwrap();
    let nvis_b = ctx.new_buffer(nb * 4).unwrap();
    let mask_words = max_blocks.div_ceil(32);
    let vmask_b = ctx.new_buffer(nb * mask_words * 4).unwrap();
    let ip = IndexParams {
        ihd: g.ihd as u32,
        inh: g.inh as u32,
        qk_dim: qk_dim as u32,
        ratio: g.ratio as u32,
        rot: g.rot as u32,
        theta: g.theta,
        eps: g.eps,
        base_pos: base_pos as u32,
        nb: nb as u32,
        b0: 0,
        b1: ((base_pos + nb) / g.ratio) as u32,
        k: g.k as u32,
        max_blocks: max_blocks as u32,
        vis_stride: vis_stride as u32,
        mask_words: mask_words as u32,
    };

    run(
        &ctx,
        &[
            (
                &pipes[0],
                vec![(0, &iqk_b), (1, &ikc_b)],
                2,
                (nb * g.ihd).div_ceil(256),
                256,
            ),
            (
                &pipes[1],
                vec![(0, &ikc_b), (1, &blk_b), (2, &wk_b)],
                3,
                ip.b1 as usize,
                g.ihd,
            ),
            (
                &pipes[2],
                vec![(0, &iqk_b), (1, &iq_b), (2, &wq_b)],
                3,
                nb * g.inh,
                g.ihd,
            ),
            (
                &pipes[3],
                vec![(0, &iq_b), (1, &blk_b), (2, &score_b)],
                3,
                nb * max_blocks.div_ceil(256),
                256,
            ),
            (
                &pipes[4],
                vec![(0, &score_b), (1, &vis_b), (2, &nvis_b), (4, &vmask_b)],
                3,
                nb,
                1024,
            ),
        ],
        &ip,
    );

    let vis: Vec<u32> = download(&vis_b, nb * vis_stride);
    let nvis: Vec<u32> = download(&nvis_b, nb);
    let vmask: Vec<u32> = download(&vmask_b, nb * mask_words);

    for b in 0..nb {
        let expect = select_ref(
            &g,
            &iqk[b * qk_dim..b * qk_dim + g.inh * g.ihd],
            &ikc,
            base_pos + b,
            &wq,
            &wk,
        );
        let got = &vis[b * vis_stride..b * vis_stride + nvis[b] as usize];

        assert_eq!(got.len(), expect.len(), "row {b}: visible count");
        assert_eq!(got, &expect[..], "row {b}: visible tokens");

        // The bitmask marks exactly the selected complete blocks.
        let blocks = (base_pos + b + 1) / g.ratio;

        for j in 0..blocks {
            let bit = (vmask[b * mask_words + j / 32] >> (j % 32)) & 1 == 1;
            let selected = expect.binary_search(&((j * g.ratio) as u32)).is_ok();

            assert_eq!(bit, selected, "row {b}: block {j} mask");
        }
    }
}

#[test]
fn index_select_handles_ties() {
    let g = Geo {
        ihd: 128,
        inh: 4,
        ratio: 4,
        k: 512,
        rot: 64,
        theta: 1e7,
        eps: 1e-6,
    };
    let base_pos = 2401usize; // t_len 2402: 600 complete blocks + 2 tail tokens
    let t_len = base_pos + 1;
    let n = t_len / g.ratio;
    let max_blocks = n + 8;
    // Five distinct values, many zeros: the selection must break ties
    // by the lowest block index.
    let scores: Vec<f32> = (0..max_blocks)
        .map(|i| {
            if i % 3 == 0 {
                0.0
            } else {
                ((i * 7) % 5) as f32 * 0.5
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();

    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));

    let mut expect: Vec<u32> = order[..g.k]
        .iter()
        .flat_map(|&b| (b * g.ratio..(b + 1) * g.ratio).map(|t| t as u32))
        .collect();

    expect.extend((n * g.ratio..t_len).map(|t| t as u32));
    expect.sort_unstable();

    let ctx = MetalContext::new().unwrap();
    let lib = ctx.compile_library(BATCH_MSL).unwrap();
    let pso = ctx.pipeline(&lib, "fn_index_select").unwrap();
    let vis_stride = g.k * g.ratio + g.ratio;
    let score_b = upload(&ctx, &scores);
    let vis_b = ctx.new_buffer(vis_stride * 4).unwrap();
    let nvis_b = ctx.new_buffer(4).unwrap();
    let mask_words = max_blocks.div_ceil(32);
    let vmask_b = ctx.new_buffer(mask_words * 4).unwrap();
    let ip = IndexParams {
        ihd: g.ihd as u32,
        inh: g.inh as u32,
        qk_dim: 0,
        ratio: g.ratio as u32,
        rot: g.rot as u32,
        theta: g.theta,
        eps: g.eps,
        base_pos: base_pos as u32,
        nb: 1,
        b0: 0,
        b1: 0,
        k: g.k as u32,
        max_blocks: max_blocks as u32,
        vis_stride: vis_stride as u32,
        mask_words: mask_words as u32,
    };

    run(
        &ctx,
        &[(
            &pso,
            vec![(0, &score_b), (1, &vis_b), (2, &nvis_b), (4, &vmask_b)],
            3,
            1,
            1024,
        )],
        &ip,
    );

    let nvis: Vec<u32> = download(&nvis_b, 1);
    let vis: Vec<u32> = download(&vis_b, nvis[0] as usize);

    assert_eq!(vis, expect);
}
