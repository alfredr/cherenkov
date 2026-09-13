use super::*;

fn group(values: &[f32]) -> Group {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
        .collect();

    quantize(&bytes).unwrap()
}

#[test]
fn affine_codes_use_low_first_nibbles_and_signed_scales() {
    let values: Vec<_> = (0..64).map(|i| (i % 16) as f32).collect();
    let packed = group(&values);

    assert_eq!(bf16::from_le_bytes(packed.scale).to_f32(), -1.0);
    assert_eq!(bf16::from_le_bytes(packed.bias).to_f32(), 15.0);
    assert_eq!(
        &packed.weight[..8],
        &[0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]
    );

    let negative: Vec<_> = values.iter().map(|v| -v).collect();
    let packed = group(&negative);

    assert_eq!(bf16::from_le_bytes(packed.scale).to_f32(), 1.0);
    assert_eq!(bf16::from_le_bytes(packed.bias).to_f32(), -15.0);
    assert_eq!(
        &packed.weight[..8],
        &[0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]
    );
}

#[test]
fn constant_groups_are_finite_and_reconstruct_exactly() {
    for size in [8, 16, 32, 64] {
        for value in [-2.0, 0.0, 3.0] {
            let packed = group(&vec![value; size]);
            let scale = bf16::from_le_bytes(packed.scale).to_f32();
            let bias = bf16::from_le_bytes(packed.bias).to_f32();

            assert!(scale.is_finite());

            for index in 0..size {
                let q = (packed.weight[index / 2] >> (4 * (index % 2))) & 15;

                assert_eq!(scale * f32::from(q) + bias, value);
            }
        }
    }
}

#[test]
fn non_finite_weights_fail_instead_of_becoming_zero_codes() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bytes: Vec<_> = [value; 16]
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect();

        assert!(quantize(&bytes).is_err());
    }
}
