//! Checkpoint names are interpreted only at the architecture boundary.

use super::TensorRole;

pub(crate) fn tensor_role(name: &str, rank: usize) -> TensorRole {
    if name.contains(".visual.") || name.starts_with("vision_tower.") {
        return TensorRole::Opaque;
    }

    if name.contains(".ngram_embedding.") {
        return TensorRole::NgramEmbedding;
    }

    if name.contains(".mlp.switch_mlp.") || name.contains(".mlp.experts.") {
        return TensorRole::Expert;
    }

    let Some(prefix) = name.strip_suffix(".weight") else {
        return TensorRole::Buffer;
    };
    let leaf = prefix.rsplit('.').next().unwrap_or(prefix);

    match leaf {
        "gate" | "shared_expert_gate" if prefix.contains(".mlp.") => TensorRole::Router,
        "conv1d" => TensorRole::Convolution,
        "embed_tokens" => TensorRole::Embedding,
        "lm_head"
        | "q_proj"
        | "k_proj"
        | "v_proj"
        | "o_proj"
        | "out_proj"
        | "in_proj_a"
        | "in_proj_b"
        | "in_proj_qkv"
        | "in_proj_z"
        | "gate_proj"
        | "up_proj"
        | "down_proj"
        | "key_proj"
        | "value_proj"
        | "index_qk_proj"
        | "input_mix_weight_down"
        | "input_mix_weight_up"
        | "block_inject_weight"
        | "fc_embedding"
        | "fc_hidden"
            if rank == 2 =>
        {
            TensorRole::Projection
        }
        _ if rank == 1 && (leaf.contains("norm") || leaf == "hc_scale") => TensorRole::Norm,
        _ => TensorRole::Opaque,
    }
}
