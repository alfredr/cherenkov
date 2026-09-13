//! MLX affine quantization, CPU side. Layout (bits=4, group_size=64):
//! - weight: u32 [out, in/8], 8 nibbles per word, LSB-first along the input dim
//! - scales, biases: bf16 [out, in/64], one (scale, bias) per 64-element group
//! - dequant: x = scale * q + bias with q in 0..=15

use half::bf16;
use rayon::prelude::*;

pub const GROUP_SIZE: usize = 64;
const NIBBLES_PER_WORD: usize = 8;

/// Zero-copy view of one quantized linear layer's tensors.
pub struct QLinear<'a> {
    pub out_dim: usize,
    pub in_dim: usize,
    pub weight: &'a [u32],
    pub scales: &'a [bf16],
    pub biases: &'a [bf16],
}

impl<'a> QLinear<'a> {
    /// Dequantize one output row into `dst` (len in_dim). Used for embedding
    /// lookup and for testing.
    pub fn dequant_row(&self, row: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.in_dim);

        let words_per_row = self.in_dim / NIBBLES_PER_WORD;
        let groups_per_row = self.in_dim / GROUP_SIZE;
        let words = &self.weight[row * words_per_row..(row + 1) * words_per_row];
        let scales = &self.scales[row * groups_per_row..(row + 1) * groups_per_row];
        let biases = &self.biases[row * groups_per_row..(row + 1) * groups_per_row];

        for (wi, &word) in words.iter().enumerate() {
            let base = wi * NIBBLES_PER_WORD;
            let g = base / GROUP_SIZE;
            let scale = scales[g].to_f32();
            let bias = biases[g].to_f32();

            for j in 0..NIBBLES_PER_WORD {
                let q = (word >> (4 * j)) & 0xF;
                dst[base + j] = scale * q as f32 + bias;
            }
        }
    }

    /// y = W x (f32 accumulate), parallel over output rows.
    /// Grouped form: `y[r] = sum_g scale[r,g] * dot(q[r,g], x_g) + bias[r,g] * sum(x_g)`.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.in_dim);
        assert_eq!(y.len(), self.out_dim);

        let groups_per_row = self.in_dim / GROUP_SIZE;
        // Per-group plain sums of x, shared across all rows.
        let xsums: Vec<f32> = x
            .as_chunks::<GROUP_SIZE>()
            .0
            .iter()
            .map(|g| g.iter().sum())
            .collect();
        let words_per_row = self.in_dim / NIBBLES_PER_WORD;

        y.par_iter_mut().enumerate().for_each(|(r, out)| {
            let words = &self.weight[r * words_per_row..(r + 1) * words_per_row];
            let scales = &self.scales[r * groups_per_row..(r + 1) * groups_per_row];
            let biases = &self.biases[r * groups_per_row..(r + 1) * groups_per_row];
            let mut acc = 0.0f32;

            const WORDS_PER_GROUP: usize = GROUP_SIZE / NIBBLES_PER_WORD;

            for g in 0..groups_per_row {
                let mut qdot = 0.0f32;
                let xg = &x[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];

                for wi in 0..WORDS_PER_GROUP {
                    let word = words[g * WORDS_PER_GROUP + wi];
                    let xw = &xg[wi * NIBBLES_PER_WORD..(wi + 1) * NIBBLES_PER_WORD];

                    for (j, &value) in xw.iter().enumerate() {
                        let q = (word >> (4 * j)) & 0xF;
                        qdot += q as f32 * value;
                    }
                }

                acc += scales[g].to_f32() * qdot + biases[g].to_f32() * xsums[g];
            }

            *out = acc;
        });
    }
}

#[cfg(test)]
#[path = "../tests/unit/quant.rs"]
mod tests;
