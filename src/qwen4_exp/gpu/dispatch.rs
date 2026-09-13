//! Buffer binding, dispatch geometry, dense projections, and transfers.

use super::*;

impl Gpu<'_> {
    // ---- small helpers ----

    pub(super) fn bind(&self, enc: &Enc, index: usize, buf: &Buf, offset: usize) {
        unsafe { enc.setBuffer_offset_atIndex(Some(buf), offset, index) };
    }

    pub(super) fn dispatch(
        &self,
        enc: &Enc,
        pso: &Pso,
        setup: impl FnOnce(&Enc),
        grid: usize,
        tg: usize,
        threadgroups: bool,
    ) {
        enc.setComputePipelineState(pso);
        setup(enc);
        self.dispatch_count.set(self.dispatch_count.get() + 1);

        let g = MTLSize {
            width: grid,
            height: 1,
            depth: 1,
        };
        let t = MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        };

        if threadgroups {
            enc.dispatchThreadgroups_threadsPerThreadgroup(g, t);
        } else {
            enc.dispatchThreads_threadsPerThreadgroup(g, t);
        }
    }

    /// Half even/odd streams + group sums of `nb` rows of `x` (f32,
    /// `in_dim` wide, starting at byte offset `x_off`) into `set`.
    pub(super) fn prep_h(
        &self,
        enc: &Enc,
        x: &Buf,
        x_off: usize,
        in_dim: u32,
        nb: usize,
        set: &HalfSet,
    ) {
        let n2 = in_dim / 2;
        let nbu = nb as u32;

        self.dispatch(
            enc,
            &self.pipes.prep_h,
            |e| {
                self.bind(e, 0, x, x_off);
                self.bind(e, 1, &set.xe, 0);
                self.bind(e, 2, &set.xo, 0);
                set_bytes(e, 3, &n2);
                set_bytes(e, 4, &nbu);
                self.bind(e, 5, &set.xsum, 0);
            },
            nb * n2 as usize,
            256,
            false,
        );
    }

    /// `y[b] = W x[b]` over `nb` prepped rows (chunks of up to 8 rows).
    pub(super) fn qmv_h(&self, enc: &Enc, q: &Q, y: &Buf, nb: usize, set: &HalfSet) {
        let p = QmvParams {
            out_dim: q.out,
            in_dim: q.inp,
        };
        let mut r0 = 0;

        while r0 < nb {
            let n = (nb - r0).min(8);
            let pipe = if n <= 3 {
                &self.pipes.qmv_h
            } else {
                &self.pipes.qmv_hn
            };
            let nu = n as u32;
            let x_off = r0 * (q.inp as usize / 2) * 2;
            let s_off = r0 * (q.inp as usize / 32) * 4;
            let y_off = r0 * q.out as usize * 4;

            self.dispatch(
                enc,
                pipe,
                |e| {
                    self.bind(e, 0, &self.dense, q.w);
                    self.bind(e, 1, &self.dense, q.s);
                    self.bind(e, 2, &self.dense, q.b);
                    self.bind(e, 3, &set.xe, x_off);
                    self.bind(e, 4, &set.xo, x_off);
                    self.bind(e, 5, y, y_off);
                    set_bytes(e, 6, &p);
                    set_bytes(e, 7, &nu);
                    self.bind(e, 8, &set.xsum, s_off);
                },
                (q.out as usize).div_ceil(4),
                128,
                true,
            );

            r0 += n;
        }
    }

    pub(super) fn zero(&self, enc: &Enc, x: &Buf, n: u32) {
        self.dispatch(
            enc,
            &self.pipes.zero,
            |e| {
                self.bind(e, 0, x, 0);
                set_bytes(e, 1, &n);
            },
            n as usize,
            256,
            false,
        );
    }

    pub(super) fn add(&self, enc: &Enc, x: &Buf, r: &Buf, n: usize) {
        self.dispatch(
            enc,
            &self.pipes.add,
            |e| {
                self.bind(e, 0, x, 0);
                self.bind(e, 1, r, 0);
            },
            n,
            256,
            false,
        );
    }

    pub(super) fn read_u32(&self, buf: &Buf, n: usize) -> Vec<u32> {
        let ptr = buf.contents().cast::<u32>();

        unsafe { std::slice::from_raw_parts(ptr.as_ptr(), n) }.to_vec()
    }

    pub(super) fn read_f32(&self, buf: &Buf, n: usize) -> Vec<f32> {
        let ptr = buf.contents().cast::<f32>();

        unsafe { std::slice::from_raw_parts(ptr.as_ptr(), n) }.to_vec()
    }
}
