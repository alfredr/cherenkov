//! CLI generation, prompt processing and shared decode entry points.

use crate::units::BYTES_PER_GB;
use crate::{options::Options, prompt::Prompt, qwen4_exp, tok};
use anyhow::{Context, Result};
use std::path::Path;

mod decode;
mod diagnostics;

pub(crate) use decode::Decode;
use diagnostics::{dump_decode, dump_prefill_chunk, dump_run, qwen4_exp_check_rows};

pub fn run(model_dir: &Path, prompt: &str, options: &Options) -> Result<()> {
    options.validate()?;

    let (max_tokens, max_ctx, raw, check, repeat) = (
        options.max_tokens,
        options.max_ctx,
        options.raw,
        options.check,
        options.repeat,
    );
    let t0 = std::time::Instant::now();
    let mut model_options = options.clone();
    let packed = qwen4_exp::packed::Packed::open(model_dir)?;

    if packed.cfg.mtp_num_hidden_layers == 0 {
        model_options.drafts = 0;
    }

    let options = &model_options;
    let tok = tok::ChatTokenizer::load(model_dir)?;
    let prompt = if raw {
        Prompt::raw(prompt.to_owned())
    } else {
        tok.template
            .as_ref()
            .context("checkpoint has no chat template")?
            .user(prompt)?
    };
    let ids = tok.encode(&prompt.text)?;

    check_budget(ids.len(), max_tokens, options.effective_drafts(), max_ctx)?;

    if options.cut_weak > 0.0 {
        eprintln!(
            "WARNING: --cut-weak skips late weak experts; output depends on disk timing and is not reproducible."
        );
    }

    let mut gpu = qwen4_exp::gpu::Gpu::load(&packed, max_ctx, options)?;
    let cpu = if check {
        Some(qwen4_exp::cpu::CpuModel::load(&packed)?)
    } else {
        None
    };

    eprintln!(
        "cherenkov gpu: max_ctx {max_ctx}, {:.2} GB Metal (expert pool {:.1} GB = {} records, working-set limit {:.2} GB), load {:.2}s, clock probe {:.2} ms",
        gpu.allocated_gb(),
        gpu.pool_bytes() as f64 / BYTES_PER_GB as f64,
        gpu.pool_slots(),
        gpu.working_set_limit_gb(),
        t0.elapsed().as_secs_f64(),
        gpu.throttle_ms()?
    );

    for run in 0..repeat {
        if run > 0 {
            gpu.reset();
            eprintln!("--- repeat {run}: clock probe {:.2} ms", gpu.throttle_ms()?);
        }

        let mut cpu_state = cpu.as_ref().map(|m| m.new_state());

        qwen4_exp_gen_once(
            &mut gpu,
            cpu.as_ref(),
            cpu_state.as_mut(),
            &tok,
            &ids,
            max_tokens,
            options,
            None,
            &mut |token| {
                use std::io::Write as _;

                print!("{}", tok.decode(&[token])?);
                std::io::stdout().flush()?;

                Ok(())
            },
        )?;
        println!();
    }

    eprintln!("clock probe at end {:.2} ms", gpu.throttle_ms()?);

    dump_run(&gpu)
}

/// Stream equal-sized prompt chunks so a short tail does not reread the whole store.
fn prefill_engine(
    gpu: &mut qwen4_exp::gpu::Gpu<'_>,
    cpu: Option<&qwen4_exp::cpu::CpuModel<'_>>,
    mut cpu_state: Option<&mut qwen4_exp::cpu::State>,
    ids: &[u32],
    n_draft: usize,
    pf_chunk: usize,
    mut argmax_lines: Option<&mut Vec<String>>,
) -> Result<PrefillResume> {
    let (mut p, mut cur, mut drafts) = (0, ids[0], Vec::new());
    let check = cpu.is_some();
    // Equal chunks: every chunk streams (nearly) the whole expert
    // store, so a small trailing chunk would cost as much as a full one.
    let n_chunks = ids.len().div_ceil(pf_chunk);
    let chunk_len = ids.len().div_ceil(n_chunks);
    let mut d1 = 0u32;

    while p < ids.len() {
        let n = (ids.len() - p).min(chunk_len);
        let chunk = &ids[p..p + n];
        let next_after = ids.get(p + n).copied();
        let (c, d) = gpu.prefill_chunk(chunk, next_after, check || argmax_lines.is_some())?;
        cur = c;
        d1 = d;

        if let Some(lines) = argmax_lines.as_deref_mut() {
            dump_prefill_chunk(gpu, true, p, n, ids.len(), lines);
        }

        if let (Some(m), Some(st)) = (cpu, cpu_state.as_deref_mut()) {
            let mut next: Vec<u32> = chunk[1..].to_vec();

            next.push(next_after.unwrap_or(cur));
            qwen4_exp_check_rows(
                m,
                st,
                gpu,
                chunk,
                gpu.has_mtp().then_some(&next[..]),
                p,
                "prefill",
                true,
            )?;
        }

        p += n;
    }

    if n_draft > 0 {
        drafts = vec![d1];

        if n_draft >= 2 {
            drafts.push(gpu.mtp_chain(d1)?);
        }
    }

    gpu.prefill_release();

    let st = &gpu.prefill_stats;
    let recs: usize = st.iter().map(|s| s.fetched).sum();
    let sum = |f: fn(&qwen4_exp::gpu::prefill::ChunkStats) -> f64| st.iter().map(f).sum::<f64>();
    let bytes: usize = st.iter().map(|s| s.fetched_bytes).sum();

    eprintln!(
        "prefill engine: {} chunk(s) of up to {pf_chunk}, {recs} expert records streamed ({:.1} GB, {:.0} MB/token), {:.1}s waiting for ring reuse | GPU s: DeltaNet blocks {:.1}, attention blocks {:.1}, expert streams {:.1}, MTP {:.1} | n-gram gather {:.1}s CPU",
        st.len(),
        bytes as f64 / BYTES_PER_GB as f64,
        bytes as f64 / 1e6 / ids.len() as f64,
        sum(|s| s.wait_s),
        sum(|s| s.gpu_delta_s),
        sum(|s| s.gpu_attn_s),
        sum(|s| s.gpu_experts_s),
        sum(|s| s.gpu_mtp_s),
        sum(|s| s.ngram_s),
    );

    Ok(PrefillResume {
        next: cur,
        drafts,
        logits: None,
    })
}

/// Use the decode row kernels for short prompts and explicit row-path checks.
fn prefill_row_batches(
    gpu: &mut qwen4_exp::gpu::Gpu<'_>,
    cpu: Option<&qwen4_exp::cpu::CpuModel<'_>>,
    mut cpu_state: Option<&mut qwen4_exp::cpu::State>,
    ids: &[u32],
    n_draft: usize,
    mut argmax_lines: Option<&mut Vec<String>>,
) -> Result<PrefillResume> {
    use qwen4_exp::gpu::MAX_NB;

    let (mut p, mut cur, mut drafts) = (0, ids[0], Vec::new());
    // Debug: CHERENKOV_ROWS_MAX caps rows per step in this path. The shared
    // trunk scratch is sized for one committed token plus the drafts, so the
    // path is clamped to that cap (it runs `step_rows` like decode).
    let rows_max: usize = std::env::var("CHERENKOV_ROWS_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_NB)
        .clamp(1, MAX_NB)
        .min(gpu.trunk_rows());

    while p < ids.len() {
        let n = (ids.len() - p).min(rows_max);
        let rows = &ids[p..p + n];
        let res = gpu.step_rows(rows, false, false)?;

        if let Some(lines) = argmax_lines.as_deref_mut() {
            dump_prefill_chunk(gpu, false, p, n, ids.len(), lines);
        }

        gpu.commit(n)?;

        let last = p + n == ids.len();
        let mut next: Vec<u32> = ids[p + 1..p + n].to_vec();

        next.push(if last { res[n - 1] } else { ids[p + n] });

        if n_draft > 0 {
            drafts = gpu.mtp_draft(&next, if last { n_draft } else { 1 })?;
        }

        if let (Some(m), Some(st)) = (cpu, cpu_state.as_deref_mut()) {
            qwen4_exp_check_rows(
                m,
                st,
                gpu,
                rows,
                (n_draft > 0).then_some(&next[..]),
                p,
                "prefill",
                false,
            )?;
        }

        p += n;
        cur = res[n - 1];
    }

    Ok(PrefillResume {
        next: cur,
        drafts,
        logits: None,
    })
}

/// Fill the trunk and draft caches, then return the next token and draft chain.
fn prefill_prompt(
    gpu: &mut qwen4_exp::gpu::Gpu<'_>,
    cpu: Option<&qwen4_exp::cpu::CpuModel<'_>>,
    cpu_state: Option<&mut qwen4_exp::cpu::State>,
    ids: &[u32],
    n_draft: usize,
    resume: Option<PrefillResume>,
) -> Result<PrefillResume> {
    use qwen4_exp::gpu::MAX_NB;

    // Prefill in chunks of MAX_NB known tokens; the MTP head follows each
    // chunk to fill its cache and drafts after the last one.
    let t1 = std::time::Instant::now();
    let prepared = resume.is_some();
    let mut seed = resume.unwrap_or_else(|| PrefillResume {
        next: ids[0],
        drafts: Vec::new(),
        logits: None,
    });
    // Prompts from CHERENKOV_PREFILL_MIN tokens (default 64) go through
    // the prefill engine in adaptive chunks, overridden by
    // CHERENKOV_PREFILL_CHUNK (at most 1024 when checking).
    let check = cpu.is_some();
    let pf_min: usize = std::env::var("CHERENKOV_PREFILL_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let engine = !prepared && ids.len() >= pf_min;
    let prefill_fit = if engine {
        gpu.prefill_rows_fit(check || std::env::var_os("CHERENKOV_DUMP_ARGMAX").is_some())?
    } else {
        1
    };
    let mut pf_chunk: usize = std::env::var("CHERENKOV_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(prefill_fit)
        .min(prefill_fit)
        .max(1);

    if check {
        pf_chunk = pf_chunk.min(1024);
    }

    // Debug: CHERENKOV_DUMP_ARGMAX=path writes "pos argmax maxlogit" for
    // every prompt row, to diff the two prefill paths position by position.
    let dump_argmax = std::env::var("CHERENKOV_DUMP_ARGMAX").ok();
    let mut argmax_lines: Vec<String> = Vec::new();
    let lines = dump_argmax.as_ref().map(|_| &mut argmax_lines);

    if engine {
        seed = prefill_engine(gpu, cpu, cpu_state, ids, n_draft, pf_chunk, lines)?;
    } else if !prepared {
        seed = prefill_row_batches(gpu, cpu, cpu_state, ids, n_draft, lines)?;
    }

    if let Some(path) = &dump_argmax {
        std::fs::write(path, argmax_lines.join("\n") + "\n")?;
    }

    // Debug: CHERENKOV_DUMP_LOGITS=path writes the last prompt row's
    // logits (bisecting the prefill engine against the row-batched path).
    // Both paths leave the last row's logits in row 0 of the head buffer
    // only when the last step had one row; the engine always puts them in
    // row 0, the row path in row (rows in last step - 1).
    if let Ok(path) = std::env::var("CHERENKOV_DUMP_LOGITS") {
        let last_rows = if engine {
            1
        } else {
            ((ids.len() - 1) % MAX_NB) + 1
        };
        let l = gpu.logits_row(last_rows - 1);
        let bytes: Vec<u8> = l.iter().flat_map(|v| v.to_le_bytes()).collect();

        std::fs::write(&path, bytes)?;
    }

    let prefill = t1.elapsed().as_secs_f64();

    if !prepared {
        eprintln!(
            "prefill {} tokens in {:.2}s ({:.1} tok/s){}",
            ids.len(),
            prefill,
            ids.len() as f64 / prefill,
            if engine {
                " [engine]"
            } else {
                " [row batches]"
            }
        );
    }

    Ok(seed)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen4_exp_gen_once(
    gpu: &mut qwen4_exp::gpu::Gpu<'_>,
    cpu: Option<&qwen4_exp::cpu::CpuModel<'_>>,
    mut cpu_state: Option<&mut qwen4_exp::cpu::State>,
    tok: &tok::ChatTokenizer,
    ids: &[u32],
    max_tokens: usize,
    options: &Options,
    resume: Option<PrefillResume>,
    emit: &mut dyn FnMut(u32) -> Result<()>,
) -> Result<()> {
    // Up to two drafts per step by default, the second only after a step
    // that accepted its whole batch (adaptive, below): a flat second
    // draft costs more than its extra tokens are worth under the
    // SSD-driven throttle, so the second draft is adaptive.
    let n_draft = options.effective_drafts();

    anyhow::ensure!(
        n_draft == 0 || gpu.has_mtp(),
        "drafting needs the checkpoint's MTP head"
    );
    // Always fold the first draft into the trunk command buffer: neutral
    // in measured speed, but saves one submit/wait. Chain adaptively after
    // fully accepted batches; this measured about 4% faster than always chaining.

    let mut seed = prefill_prompt(gpu, cpu, cpu_state.as_deref_mut(), ids, n_draft, resume)?;

    if !options.sampling.greedy() && seed.logits.is_none() {
        seed.logits = Some(gpu.logits_row(gpu.last_logits_row()).to_vec());
    }

    let prefill_steps = gpu.step_ms.len();
    let mut decoder = Decode::new(seed, ids, tok, options, max_tokens, None)?;
    let t2 = std::time::Instant::now();

    while decoder.finish_reason.is_none() {
        decoder.step(gpu, cpu, cpu_state.as_deref_mut(), emit)?;
    }

    let steps = decoder.steps;
    let accepted = decoder.accepted;
    let out = decoder.tokens;
    let decode = t2.elapsed().as_secs_f64();
    let n = steps.max(1);
    let recent = |v: &[f64]| -> Vec<f64> { v[v.len().saturating_sub(n)..].to_vec() };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let ms = recent(&gpu.step_ms);
    let gpu_recent = recent(&gpu.gpu_ms);
    let io_recent = recent(&gpu.io_ms);
    let set_recent = recent(&gpu.set_ms);
    let read_recent = recent(&gpu.read_ms);
    let idle_recent = recent(&gpu.gpu_idle_ms);
    let mtp_recent = recent(&gpu.mtp_ms);
    let ngram_recent = recent(&gpu.ngram_ms);
    let miss_recent = &gpu.misses[gpu.misses.len().saturating_sub(n)..];
    let la_recent = &gpu.lookahead_issued[gpu.lookahead_issued.len().saturating_sub(n)..];
    let warm_recent = &gpu.warm[gpu.warm.len().saturating_sub(n)..];
    let cut_recent = &gpu.cut[gpu.cut.len().saturating_sub(n)..];

    eprintln!(
        "  dispatches/step {} | cpu turnaround {:.1} ms/step | drafts {n_draft}, accepted {:.2}/step ({:.2} tokens/step) | mtp {:.1} ms/step | n-gram gather {:.1} ms/step",
        gpu.dispatches.last().copied().unwrap_or(0),
        mean(&idle_recent),
        accepted as f64 / n as f64,
        (out.len()) as f64 / n as f64,
        mean(&mtp_recent),
        mean(&ngram_recent),
    );
    eprintln!(
        "decode {} tokens in {:.2}s ({:.2} tok/s) | {} steps, mean step {:.1} ms (min {:.1}, max {:.1}) | gpu-span {:.1} ms | io wait {:.1} ms (set {:.1}, read {:.1}) | sync fetches/step {:.1} + lookahead {:.1} (warm {:.1}, cut {:.1}) | pool {}/{} | {:.2} GB Metal",
        out.len(),
        decode,
        out.len() as f64 / decode.max(1e-9),
        steps,
        mean(&ms),
        ms.iter().cloned().fold(f64::INFINITY, f64::min),
        ms.iter().cloned().fold(0.0, f64::max),
        mean(&gpu_recent),
        mean(&io_recent),
        mean(&set_recent),
        mean(&read_recent),
        miss_recent.iter().sum::<usize>() as f64 / miss_recent.len().max(1) as f64,
        la_recent.iter().sum::<usize>() as f64 / la_recent.len().max(1) as f64,
        warm_recent.iter().sum::<usize>() as f64 / warm_recent.len().max(1) as f64,
        cut_recent.iter().sum::<usize>() as f64 / cut_recent.len().max(1) as f64,
        gpu.pool_resident(),
        gpu.pool_slots(),
        gpu.allocated_gb()
    );
    eprintln!(
        "memory_stats {}",
        serde_json::to_string(&gpu.memory_stats())?
    );
    dump_decode(gpu, ids, &out, prefill_steps)?;

    Ok(())
}

#[derive(Default)]
pub(crate) struct PrefillResume {
    pub next: u32,
    pub drafts: Vec<u32>,
    pub logits: Option<Vec<f32>>,
}

pub(crate) fn check_budget(
    prompt: usize,
    output: usize,
    drafts: usize,
    context: usize,
) -> Result<()> {
    anyhow::ensure!(prompt > 0, "prompt must contain at least one token");

    let need = prompt
        .checked_add(output)
        .and_then(|n| n.checked_add(drafts))
        .ok_or_else(|| anyhow::anyhow!("token budget overflows"))?;

    anyhow::ensure!(
        need <= context,
        "prompt ({prompt}) + output ({output}) + draft lookahead ({drafts}) needs {need} tokens, exceeding --max-ctx {context}"
    );

    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/runner.rs"]
mod tests;
