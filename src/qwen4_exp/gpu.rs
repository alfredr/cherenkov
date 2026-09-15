//! Metal decode path for qwen4-exp: row-batched steps (prefill chunks and
//! speculative verification of MTP drafts) over an explicit wired expert
//! pool.
//!
//! Dense weights are one zero-copy buffer over `dense.bin`. Routed experts
//! live in a fixed pool of page-aligned slots (Metal wires whole buffers per
//! command buffer), LRU managed here
//! and filled by uncached parallel reads. One command buffer per step: after
//! each layer's router the GPU signals a shared event and waits for the CPU
//! to publish that layer's expert slots; a one-layer lookahead router lets
//! most fetches run in the background.
//!
//! A step runs `nb` consecutive positions at once (activations are
//! row-major `[nb][...]`). The gated DeltaNet scan snapshots its state after
//! each row so a partially accepted verify batch can roll back.

use super::cpu::CpuModel;
use super::packed::Packed;
use crate::kernels::{BATCH_MSL, FORWARD_MSL};
use crate::metal::MetalContext;
use crate::options::{Options, PoolBudget};
use crate::units::BYTES_PER_GB;
use anyhow::{Context, Result, ensure};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSharedEvent, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

mod activity;
mod attention;
mod budget;
mod decode;
mod deltanet;
mod dispatch;
mod experts;
mod hyperconnection;
mod load;
mod memory;
mod mtp;
mod params;
mod phases;
mod ple;
mod sampling;
mod state;

pub use activity::layers::{PhaseStats, PredictionStats, QuantStats};
pub use activity::reads::ReadStats;
pub use activity::{ExpertActivity, ExpertCounters, LayerStats};
pub use memory::MemoryStats;
pub use phases::GpuTiming;

use params::*;

pub mod prefill;
mod residency;
mod streaming;

use streaming::PendingRead;

const ARGMAX_TGS: usize = 1024;
const ATTN_TB: usize = 512;
const ATTN_MAX_WG: usize = 256;
/// Rows per step (prefill chunk, or 1 + drafts when verifying).
pub const MAX_NB: usize = 4;
/// Rollback planes: verify batches may have up to this many draft rows.
pub const MAX_SNAP: usize = 3;
/// Slots per layer row of the expert slot table (last entry = union size).
const SLOT_STRIDE: usize = 64;
/// ids buffer layout: trunk inputs, trunk argmax, MTP inputs, MTP argmax.
const IDS_IN: usize = 0;
const IDS_OUT: usize = 8;
const IDS_MTP_IN: usize = 16;
const IDS_MTP_OUT: usize = 24;

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Tokens, expert IDs, routing weights, and misses for one step.
/// Keep the tuple layout for compatibility with existing research dumps.
pub type RouteStep = (Vec<u32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<Vec<u32>>);

/// Affine-Q4 projection: dimensions and byte offsets in its bound weight buffer.
#[derive(Clone, Copy)]
struct Q {
    w: usize,
    s: usize,
    b: usize,
    out: u32,
    inp: u32,
}

impl Q {
    /// Resolve a record-relative projection against a bound buffer.
    fn at_offset(self, bytes: usize) -> Self {
        Self {
            w: self.w + bytes,
            s: self.s + bytes,
            b: self.b + bytes,
            ..self
        }
    }
}

/// Byte offset of a bf16 tensor in dense.bin.
#[derive(Clone, Copy)]
struct T(usize);

struct Hc {
    norm: T,
    down: Q,
    up: Q,
    inject: Option<Q>,
}

struct Attn {
    q: Q,
    k: Q,
    v: Q,
    o: Q,
    qn: T,
    kn: T,
    kc: Buf,
    vc: Buf,
    /// QSA indexer: the index q/k projection, its norms, the raw index
    /// key cache `[max_t][ihd]` and the block keys `[max_t/ratio][ihd]`.
    iqk: Q,
    iqn: T,
    ikn: T,
    ikc: Buf,
    blk: Buf,
}

struct Delta {
    qkv: Q,
    z: Q,
    a: Q,
    b: Q,
    conv: T,
    a_log: T,
    dt_bias: T,
    norm: T,
    o: Q,
    state: Buf,
    hist: Buf,
    /// Snapshot planes of `state` / `hist` after each verify row.
    mid: Buf,
    mid_hist: Buf,
}

enum Mix {
    Attn(Attn),
    Delta(Delta),
}

struct Moe {
    router: T,
    shared_gate: T,
    sg: Q,
    su: Q,
    sd: Q,
    record_layer: usize,
}

struct Ple {
    key: Q,
    value: Q,
    norm_key: T,
    norm_query: T,
    norm_conv: T,
    conv: T,
    kernel: u32,
    dilation: u32,
    span: u32,
    hist: Buf,
    /// `[nb][ple_embed_dim]` n-gram rows gathered on the CPU.
    e: Buf,
    multipliers: Vec<i64>,
    head_offsets: Vec<u64>,
    head_sizes: Vec<u64>,
}

struct GLayer {
    attn_hc: Hc,
    mlp_hc: Hc,
    mix: Mix,
    moe: Moe,
    ple: Option<Ple>,
}

/// The MTP draft head (see cpu.rs `MtpWeights`).
struct Mtp {
    enorm: T,
    hnorm: T,
    fc_e: Q,
    fc_h: Q,
    layer: GLayer,
    mixer: Hc,
}

struct Pipes {
    // Common kernel library (kernels/common/).
    qmv_h: Pso,
    qmv_hn: Pso,
    prep_h: Pso,
    embed_rows: Pso,
    qk_norm_rope_b: Pso,
    conv_b: Pso,
    delta_norms: Pso,
    delta_gates: Pso,
    delta_scan2: Pso,
    kv_append_q8: Pso,
    attn_part2_q8: Pso,
    attn_combine: Pso,
    argmax_partial: Pso,
    argmax_final: Pso,
    add: Pso,
    copy_f32: Pso,
    // qwen4-exp kernel library (kernels/qwen4_exp/).
    zero: Pso,
    replicate_b: Pso,
    group_norm_b: Pso,
    norm_prep_b: Pso,
    qmv_silu_b: [Pso; MAX_NB],
    hc_mix_b: Pso,
    inject_b: Pso,
    bf16_matvec_b: Pso,
    topk_softmax_b: Pso,
    moe_gate_up_b: [Pso; MAX_NB],
    moe_act_b: Pso,
    moe_down_b: [Pso; MAX_NB],
    moe_combine_b: Pso,
    ple_gate_b: Pso,
    ple_conv_b: Pso,
    mtp_fold: Pso,
    gate_norm_sigmoid_b: Pso,
    index_append: Pso,
    index_blocks: Pso,
    index_q: Pso,
    index_score: Pso,
    index_select: Pso,
    attn_sel: Pso,
    // prefill engine
    qmm_n8: Pso,
    qmm_n16: Pso,
    qmm_w: Pso,
    /// [precision 2/3][token tile 8/16/32].
    expert_qmm: [[Pso; 3]; 2],
    silu_mul: Pso,
    attn_q_stage: Pso,
    attn_kv_stage: Pso,
    gemm_hh: Pso,
    attn_o_scatter: Pso,
    softmax_sel: Pso,
    silu_rows: Pso,
    gather_rows: Pso,
    scatter_add_rows: Pso,
    shared_add_rows: Pso,
    qmv_small_b: Pso,
}

/// One lookahead prediction and what became of it (CHERENKOV_DUMP_LA).
#[derive(Clone, Copy, serde::Serialize)]
pub struct LaEntry {
    pub step: u64,
    /// Layer the prediction was for.
    pub layer: usize,
    pub expert: u32,
    /// Largest lookahead router weight over the rows, and best rank.
    pub weight: f32,
    pub rank: u32,
    /// Already in the pool when predicted (no fetch needed).
    pub resident: bool,
    /// Used by that layer in this step.
    pub hit: bool,
}

/// Half even/odd streams plus per-32 group sums of a row batch, the input
/// format of the multi-row Q4 matvecs.
struct HalfSet {
    xe: Buf,
    xo: Buf,
    xsum: Buf,
}

/// Buffers of one gated-residual read (the main set, or the lookahead's).
struct HcBufs {
    normed: Buf,
    d: Buf,
    u: Buf,
    mixed: Buf,
    inj: Buf,
    h1: HalfSet,
    h2: HalfSet,
}

struct Scratch {
    ids: Buf,
    e: Buf,
    hyper: Buf,
    hc: HcBufs,
    la: HcBufs,
    mix_out: Buf,
    qg: Buf,
    k: Buf,
    v: Buf,
    attn_out: Buf,
    attn_parts: Buf,
    qkv: Buf,
    z: Buf,
    a: Buf,
    b: Buf,
    kqn: Buf,
    gbuf: Buf,
    delta_y: Buf,
    router: Buf,
    topk_idx: Buf,
    topk_w: Buf,
    la_router: Buf,
    la_idx: Buf,
    la_w: Buf,
    gate_e: Buf,
    hx: HalfSet,
    y_e: Buf,
    moe_out: Buf,
    ple_key: Buf,
    ple_keyn: Buf,
    ple_value: Buf,
    ple_query: Buf,
    ple_gated: Buf,
    ple_gvn: Buf,
    ple_out: Buf,
    logits: Buf,
    mtp_logits: Buf,
    /// Logits of chained (single-row) MTP passes, kept apart so the first
    /// pass's rows stay readable for checks.
    mtp_logits2: Buf,
    partials: Buf,
    mtp_hyper: Buf,
    fe: Buf,
    fh: Buf,
    /// Indexer: projection `[nb][qk_dim]`, roped queries `[nb][inh*ihd]`,
    /// block scores `[nb][max_blocks]`, visible tokens `[nb][vis_stride]`.
    iqk: Buf,
    iq: Buf,
    bscore: Buf,
    vis: Buf,
    nvis: Buf,
    vmask: Buf,
}

fn set_bytes<T>(enc: &Enc, index: usize, value: &T) {
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::from(value).cast::<c_void>(),
            std::mem::size_of::<T>(),
            index,
        )
    };
}

fn kv_q8_side(max_t: usize, kv_row: usize) -> (usize, usize) {
    let align = |v: usize| v.div_ceil(16384) * 16384;
    let qs = align(max_t * kv_row);
    let sc = align(max_t * kv_row / 32 * 2);

    (qs, qs + sc)
}

/// Debug: CHERENKOV_LAYERS=N runs only the first N decoder blocks (for
/// bisecting one execution path against another).
fn layer_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

    *CAP.get_or_init(|| {
        std::env::var("CHERENKOV_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX)
    })
}

/// Unique entries in first-seen order.
fn union_of(ids: &[u32]) -> Vec<u32> {
    let mut u: Vec<u32> = Vec::with_capacity(ids.len());

    for &e in ids {
        if !u.contains(&e) {
            u.push(e);
        }
    }

    u
}

pub struct Gpu<'a> {
    pub p: &'a Packed,
    ctx: MetalContext,
    pipes: Pipes,
    dense: Buf,
    /// LRU expert pool: wired copy slots by default, or mapped residency
    /// set entries with the developer override. Cached reads serve mapped
    /// entries; uncached reads fill copy slots and the prefill ring.
    res: residency::Pool,
    activity: ExpertActivity,
    read_tracker: activity::reads::ReadTracker,
    activity_started: std::time::Instant,
    phase_timer: Option<phases::PhaseTimer>,
    pool_file: std::fs::File,
    pool_file_nocache: std::fs::File,
    /// The n-gram table, for touching rows' pages ahead of the gather.
    ngram_file: std::fs::File,
    ngram_prefetch: std::cell::RefCell<Option<std::thread::JoinHandle<()>>>,
    step_no: u64,
    /// Background read for the lookahead prediction of the next layer,
    /// with the records to add to the set once it lands.
    pending: Option<PendingRead>,
    /// Decode synchronous misses and prefill ring misses read the low-bit record
    /// (--miss-experts 3|2) when the store exists; lookahead reads
    /// stay 4-bit unless `all_low_bits`.
    low_bit_store: Option<super::lowbit::Layout>,
    /// --experts 3|2: every expert record is low-bit, so the pool
    /// holds more of them. Prefill uses the same records and retains its
    /// most-used experts for decode.
    pub all_low_bits: bool,
    /// GPU <-> CPU handshake: the GPU signals after each layer's router and
    /// waits for the CPU to publish that layer's pool slots in `slot_tab`.
    event: Retained<ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
    event_base: u64,
    /// CPU -> GPU signals of the prefill expert stream (a second event,
    /// so each event has one writer and its values stay monotonic).
    event_cpu: Retained<ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
    /// GPU -> CPU: a block's resident expert part is done (value seq+2).
    event_res: Retained<ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
    event_cpu_base: u64,
    /// [layer][SLOT_STRIDE] GPU addresses (u64) of the union of routed
    /// experts' records (entry 62: resident count, 63: union size) and
    /// [layer][MAX_NB][SLOT_STRIDE] per-row routing weights, written by
    /// the CPU. Row `n_layers` is the MTP's.
    slot_tab: Buf,
    wmap: Buf,
    layers: Vec<GLayer>,
    mtp: Option<Mtp>,
    embed: Q,
    final_mixer: Hc,
    lm_head: Q,
    scratch: Scratch,
    pub max_t: usize,
    /// Trunk rows per step the shared scratch is sized for: `1 + drafts`
    /// (clamped to `MAX_NB`). Decode verify and the prefill row path share
    /// the trunk scratch, so the row cap they may use is exactly this.
    trunk_rows: usize,
    /// Committed positions.
    pub pos: usize,
    /// Committed tokens, plus the rows of the step in flight.
    tokens: Vec<u32>,
    /// The step in flight: first position and row count.
    batch_pos: usize,
    batch_nb: usize,
    batch_snap: bool,
    /// Positions held by the MTP head's KV cache.
    mtp_len: usize,
    /// Router choices of the last step, per layer (union over rows).
    pub last_experts: Vec<Vec<u32>>,
    /// Router choices of every step, `[step][layer][union]` (cache studies).
    pub expert_history: Vec<Vec<Vec<u32>>>,
    /// Per step: the rows' tokens and, per serviced layer, every row's
    /// top-k expert ids (nb * k, row-major), for offline predictor studies.
    pub route_history: Vec<RouteStep>,
    /// CHERENKOV_DUMP_STATES: per step, row 0's router input at every
    /// serviced block (hidden floats) alongside its top-k, for training an
    /// expert predictor offline.
    dump_states: bool,
    pub state_history: Vec<Vec<(Vec<f32>, Vec<u32>)>>,
    last_states: Vec<(Vec<f32>, Vec<u32>)>,
    last_routes: Vec<Vec<u32>>,
    last_route_w: Vec<Vec<f32>>,
    /// Expert ids of each serviced layer's misses (records read in-step).
    last_miss: Vec<Vec<u32>>,
    /// Seconds spent gathering n-gram rows from the mapped table (CPU,
    /// page faults included), cumulative; `ngram_ms` is the per-step view.
    pub ngram_gather_s: std::cell::Cell<f64>,
    pub ngram_ms: Vec<f64>,
    pub step_ms: Vec<f64>,
    /// Command-buffer spans per step, including event waits.
    pub gpu_ms: Vec<f64>,
    /// Time spent waiting for synchronous expert fetches, per step.
    pub io_ms: Vec<f64>,
    /// Records fetched from disk at exact routing time, per step.
    pub misses: Vec<usize>,
    pub miss_bytes: Vec<usize>,
    step_misses: usize,
    step_miss_bytes: usize,
    /// Seconds in residency-set bookkeeping (acquire, add, commit) and
    /// blocked on miss reads, per step.
    step_set_s: f64,
    step_read_s: f64,
    /// Argmax per row of the MTP head folded into the last trunk step
    /// (`step_rows` with fold), consumed by `mtp_draft`.
    folded_mtp: Option<Vec<u32>>,
    /// Records needed this step whose pages were still in the page
    /// cache (no read).
    step_warm: usize,
    pub warm: Vec<usize>,
    pub set_ms: Vec<f64>,
    pub read_ms: Vec<f64>,
    /// One-layer lookahead routing: fraction of a layer's exact experts
    /// predicted by the approximate router run one layer earlier, and the
    /// records the prediction fetched.
    pub lookahead_hit: Vec<f64>,
    pub lookahead_issued: Vec<usize>,
    /// Every lookahead prediction with its outcome (diagnostics), and the
    /// ones awaiting the next layer's exact routing.
    pub la_log: Vec<LaEntry>,
    la_pending: Vec<LaEntry>,
    log_la: bool,
    /// Deadline policy (--cut-weak w, default 0 = off): once the
    /// GPU has finished a block's resident experts, a missing expert whose
    /// weight is below w in every row and whose read has not landed is cut
    /// from the late part instead of waited for; stronger ones are waited
    /// for. Cut reads finish in the background and are joined at step end.
    cut_w: f32,
    step_cut: usize,
    pub cut: Vec<usize>,
    inflight: Vec<(residency::Landed, usize)>,
    lookahead: bool,
    /// CHERENKOV_SPIN=0: block on the shared event instead of spinning.
    spin_wait: bool,
    /// Diagnostics: CHERENKOV_FAKE=experts skips disk fetches (garbage
    /// numerics, true GPU timing); CHERENKOV_SKIP=stage,... skips stages
    /// (mixer, experts, shared, lmhead) to attribute GPU time.
    fake_experts: bool,
    skip: Vec<String>,
    /// Kernel dispatches per step and GPU idle time waiting on the CPU
    /// within a step.
    pub dispatches: Vec<usize>,
    /// Legacy name: CPU service wall time, which overlaps resident GPU work.
    pub gpu_idle_ms: Vec<f64>,
    /// Rows per trunk step and time of MTP draft passes, per step.
    pub rows: Vec<usize>,
    pub mtp_ms: Vec<f64>,
    dispatch_count: std::cell::Cell<usize>,
    /// Prefill engine scratch, allocated for the first long prompt and
    /// released after it, and the ring the expert stream cycles through
    /// (the pool itself keeps each layer's most-used experts).
    pf: Option<prefill::PrefillScratch>,
    ring: Buf,
    /// Per prefill chunk: tokens, seconds, expert records streamed,
    /// seconds waiting on fetches, and GPU seconds in (DeltaNet blocks,
    /// attention blocks, expert streams, MTP).
    prefill_reserved_bytes: usize,
    pub prefill_stats: Vec<prefill::ChunkStats>,
}

/// Compact sequence checkpoint; no weights, expert slots, or device event counters.
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct PrefixState {
    pos: usize,
    mtp_len: usize,
    tokens: Vec<u32>,
    data: Vec<Vec<u8>>,
}

impl<'a> Gpu<'a> {
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Rows per step the shared trunk scratch is sized for (the row cap the
    /// decode verify path and the short-prompt prefill row path may use).
    pub(crate) fn trunk_rows(&self) -> usize {
        self.trunk_rows
    }

    fn skips(&self, stage: &str) -> bool {
        self.skip.iter().any(|s| s == stage)
    }

    fn record_id(&self, record_layer: usize, expert: u32) -> usize {
        record_layer * self.p.manifest.experts.experts + expert as usize
    }

    /// Records resident in the expert pool (the residency set).
    pub fn pool_resident(&self) -> usize {
        self.res.resident()
    }

    pub fn pool_slots(&self) -> usize {
        self.res.budget()
    }

    /// Dependent-FMA chain time in ms (higher = deeper clock throttle).
    pub fn throttle_ms(&self) -> Result<f64> {
        self.ctx.throttle_probe()
    }

    /// Trunk logits of row `r` of the last step.
    pub fn logits_row(&self, r: usize) -> &[f32] {
        let v = self.p.cfg.vocab_size;
        let ptr = self.scratch.logits.contents().cast::<f32>();

        unsafe { std::slice::from_raw_parts(ptr.as_ptr().add(r * v), v) }
    }

    pub fn logits(&self) -> &[f32] {
        self.logits_row(0)
    }

    /// Batched prefill writes one final row; row stepping retains every row.
    pub(crate) fn last_logits_row(&self) -> usize {
        self.batch_nb.saturating_sub(1)
    }

    /// MTP logits of row `r` of the last draft pass.
    pub fn mtp_logits_row(&self, r: usize) -> &[f32] {
        let v = self.p.cfg.vocab_size;
        let ptr = self.scratch.mtp_logits.contents().cast::<f32>();

        unsafe { std::slice::from_raw_parts(ptr.as_ptr().add(r * v), v) }
    }

    pub fn allocated_gb(&self) -> f64 {
        self.allocated_bytes() as f64 / BYTES_PER_GB as f64
    }

    /// What Metal will keep resident at once on this machine.
    pub fn pool_bytes(&self) -> usize {
        self.res.bytes()
    }

    /// Positional KV-cache bytes for one attention layer at `n` positions:
    /// the q8 key/values (kc, vc) with their q4 scales, the QSA index key
    /// cache and the compressed block keys. Mirrors `prefix_regions`.
    fn kv_layer_bytes(&self, n: usize) -> usize {
        let c = &self.p.cfg;
        let kv_row = c.num_key_value_heads * c.head_dim;
        let ihd = c.indexer_head_dim;
        let ratio = c.indexer_compress_ratio.max(1);

        // The QSA index key cache and the block keys are half precision, so
        // each index element is 2 bytes (the raw ikc and compressed blk).
        2 * (n * kv_row + n * kv_row / 32 * 2) + n * ihd * 2 + n.div_ceil(ratio) * ihd * 2
    }

    /// Trunk attention (KV) layers; DeltaNet layers hold no positional KV.
    fn trunk_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.mix, Mix::Attn(_)))
            .count()
    }

    /// Whether the MTP head contributes an attention KV cache (0 or 1).
    fn mtp_attn_layers(&self) -> usize {
        usize::from(
            self.mtp
                .as_ref()
                .is_some_and(|m| matches!(m.layer.mix, Mix::Attn(_))),
        )
    }

    /// Bytes of the positional KV caches now in use by the sequence on the
    /// GPU (trunk layers at `pos`, the MTP head at `mtp_len`).
    pub fn kv_cache_bytes(&self) -> usize {
        self.trunk_attn_layers() * self.kv_layer_bytes(self.pos)
            + self.mtp_attn_layers() * self.kv_layer_bytes(self.mtp_len)
    }

    /// Total capacity of the positional KV caches at `max_t` positions.
    pub fn kv_cache_capacity(&self) -> usize {
        (self.trunk_attn_layers() + self.mtp_attn_layers()) * self.kv_layer_bytes(self.max_t)
    }

    /// Fraction (0..=1) of the positional KV caches in use by the sequence.
    pub fn kv_cache_fullness(&self) -> f64 {
        let cap = self.kv_cache_capacity();

        self.kv_cache_bytes() as f64 / cap.max(1) as f64
    }

    /// Fraction (0..=1) of the context window in use by the sequence.
    pub fn context_fullness(&self) -> f64 {
        self.pos as f64 / self.max_t.max(1) as f64
    }

    pub fn working_set_limit_gb(&self) -> f64 {
        use objc2_metal::MTLDevice as _;

        self.ctx.device.recommendedMaxWorkingSetSize() as f64 / BYTES_PER_GB as f64
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4_exp/gpu/mod.rs"]
mod tests;
