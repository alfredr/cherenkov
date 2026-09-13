use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct ChatTokenizer {
    pub inner: Tokenizer,
    pub template: Option<crate::prompt::ChatTemplate>,
    pub im_end: u32,
    pub endoftext: u32,
}

impl ChatTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let mut inner = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))?;

        // Checkpoints may save training-time padding and truncation. Inference
        // uses actual prompt lengths and checks its own context budget.
        inner.with_padding(None);
        inner
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;

        let tok = |s: &str| -> Result<u32> {
            inner
                .token_to_id(s)
                .with_context(|| format!("special token {s:?} missing"))
        };
        let im_end = tok("<|im_end|>")?;
        let endoftext = tok("<|endoftext|>")?;

        Ok(ChatTokenizer {
            inner,
            template: crate::prompt::ChatTemplate::load(model_dir)?,
            im_end,
            endoftext,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // ChatML scaffolding is already present; do not add template tokens.
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;

        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}
