//! Affine Q4 groups: eight low-first nibbles per word, BF16 scale and bias.

use anyhow::{Result, ensure};
use half::bf16;

pub(super) struct Group {
    pub weight: [u8; 32],
    pub scale: [u8; 2],
    pub bias: [u8; 2],
}

/// Use MLX's affine endpoint selection and ties-to-even rounding. Quantize
/// with float parameters, then store them as BF16, as MLX does for BF16 input.
/// Reference: mlx/backend/cpu/quantized.cpp, quantize<T, U>.
pub(super) fn quantize(bytes: &[u8]) -> Result<Group> {
    let mut values = [0.0_f32; 64];
    let len = bytes.len() / 2;

    ensure!(
        matches!(len, 8 | 16 | 32 | 64),
        "unsupported affine group size {len}"
    );

    for (value, pair) in values.iter_mut().zip(bytes.as_chunks::<2>().0) {
        *value = bf16::from_bits(u16::from_le_bytes(*pair)).to_f32();

        ensure!(
            value.is_finite(),
            "cannot quantize a non-finite BF16 weight"
        );
    }

    let values = &values[..len];
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let magnitude = ((max - min) / 15.0).max(1e-7);
    let (edge, mut scale) = if min.abs() > max.abs() {
        (min, magnitude)
    } else {
        (max, -magnitude)
    };
    let zero = (edge / scale).round_ties_even();
    let mut bias = 0.0;

    if zero != 0.0 {
        scale = edge / zero;
        bias = edge;
    }

    let stored_scale = bf16::from_f32(scale);

    ensure!(
        scale.is_finite() && scale != 0.0 && stored_scale.is_finite(),
        "affine scale overflow"
    );

    let mut group = Group {
        weight: [0; 32],
        scale: stored_scale.to_le_bytes(),
        bias: bf16::from_f32(bias).to_le_bytes(),
    };

    for (i, &value) in values.iter().enumerate() {
        let nibble = ((value - bias) / scale).round_ties_even().clamp(0.0, 15.0) as u8;
        group.weight[i / 2] |= nibble << (4 * (i % 2));
    }

    Ok(group)
}

#[cfg(test)]
#[path = "../../../tests/unit/qwen4_exp/pack/affine.rs"]
mod tests;
