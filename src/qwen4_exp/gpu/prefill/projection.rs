//! Dense prefill projections with narrow-output fallback.

use super::*;

impl Gpu<'_> {
    /// `y[b] = W x[b]` for nb rows with the simdgroup-matrix GEMM: the
    /// x buffer must hold rows padded to the token tile (32).
    pub(super) fn qmm_from(&self, enc: &Enc, wb: &Buf, q: &Q, x: &Buf, y: &Buf, nb: usize) {
        let p = QmvParams {
            out_dim: q.out,
            in_dim: q.inp,
        };

        if q.out < 128 {
            // The GEMM tiles clamp-load and store 8-row fragments; narrow
            // outputs go through the plain per-output kernel.
            let nbu = nb as u32;

            self.dispatch(
                enc,
                &self.pipes.qmv_small_b,
                |e| {
                    self.bind(e, 0, wb, q.w);
                    self.bind(e, 1, wb, q.s);
                    self.bind(e, 2, wb, q.b);
                    self.bind(e, 3, x, 0);
                    self.bind(e, 4, y, 0);
                    set_bytes(e, 5, &p);
                    set_bytes(e, 6, &nbu);
                },
                (nb * q.out as usize).div_ceil(4),
                128,
                true,
            );

            return;
        }

        self.qmm_tiled(
            enc,
            wb,
            q,
            x,
            y,
            nb,
            [&self.pipes.qmm_n8, &self.pipes.qmm_n16, &self.pipes.qmm_w],
        );
    }

    /// Low-bit expert records share the tiled dispatch and output layout,
    /// but use their own unpacking kernels. Dense projections stay Q4.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn expert_qmm_from(
        &self,
        enc: &Enc,
        wb: &Buf,
        q: &Q,
        x: &Buf,
        y: &Buf,
        nb: usize,
        bits: u32,
    ) {
        if bits == 4 {
            self.qmm_from(enc, wb, q, x, y, nb);
        } else {
            let pipes = &self.pipes.expert_qmm[(bits - 2) as usize];

            self.qmm_tiled(enc, wb, q, x, y, nb, [&pipes[0], &pipes[1], &pipes[2]]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qmm_tiled(&self, enc: &Enc, wb: &Buf, q: &Q, x: &Buf, y: &Buf, nb: usize, pipes: [&Pso; 3]) {
        let p = QmvParams {
            out_dim: q.out,
            in_dim: q.inp,
        };
        let (pipe, tile) = if nb <= 8 {
            (pipes[0], 8)
        } else if nb <= 16 {
            (pipes[1], 16)
        } else {
            (pipes[2], 32)
        };
        let ntt = nb.div_ceil(tile).max(1);
        let nttu = ntt as u32;

        self.dispatch(
            enc,
            pipe,
            |e| {
                self.bind(e, 0, wb, q.w);
                self.bind(e, 1, wb, q.s);
                self.bind(e, 2, wb, q.b);
                self.bind(e, 3, x, 0);
                self.bind(e, 4, y, 0);
                set_bytes(e, 5, &p);
                set_bytes(e, 6, &nttu);
            },
            (q.out as usize).div_ceil(128) * ntt,
            128,
            true,
        );
    }

    pub(super) fn qmm(&self, enc: &Enc, q: &Q, x: &Buf, y: &Buf, nb: usize) {
        self.qmm_from(enc, &self.dense, q, x, y, nb);
    }
}
