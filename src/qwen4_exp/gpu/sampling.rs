//! Final hyper-connection read, logits projection, and greedy argmax.

use super::*;

impl Gpu<'_> {
    /// Final mixer read of `nb` rows of `hyper` (injecting `pending`
    /// first), LM head into `logits` `[nb][vocab]`, greedy argmax of each
    /// row into ids[ids_out + b].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn head_b(
        &self,
        enc: &Enc,
        mixer: &Hc,
        nb: usize,
        hyper: &Buf,
        hyper_off: usize,
        pending: Option<&Buf>,
        logits: &Buf,
        ids_out: usize,
    ) {
        let s = &self.scratch;
        let vocab = self.p.cfg.vocab_size;

        self.hc_read_b(enc, mixer, nb, hyper, hyper_off, pending, &s.hc, false);

        if !self.skips("lmhead") {
            self.qmv_h(enc, &self.lm_head, logits, nb, &s.hc.h1);
        }

        let n = vocab as u32;
        let np = ARGMAX_TGS as u32;

        for b in 0..nb {
            self.dispatch(
                enc,
                &self.pipes.argmax_partial,
                |e| {
                    self.bind(e, 0, logits, b * vocab * 4);
                    self.bind(e, 1, &s.partials, 0);
                    set_bytes(e, 2, &n);
                },
                ARGMAX_TGS,
                256,
                true,
            );

            let step = (ids_out + b - 1) as u32;

            self.dispatch(
                enc,
                &self.pipes.argmax_final,
                |e| {
                    self.bind(e, 0, &s.partials, 0);
                    self.bind(e, 1, &s.ids, 0);
                    set_bytes(e, 2, &np);
                    set_bytes(e, 3, &step);
                },
                1,
                1024,
                true,
            );
        }
    }
}
