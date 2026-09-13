pub(crate) mod mlx_checkpoint;

/// Write fixture bytes without validating them, so tests can supply malformed headers.
pub(crate) fn write_safetensors(
    path: &std::path::Path,
    header: &impl serde::Serialize,
    data: &[u8],
) {
    use std::io::Write;

    let header = serde_json::to_vec(header).unwrap();
    let mut file = std::fs::File::create(path).unwrap();

    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(data).unwrap();
}
