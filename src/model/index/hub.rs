//! Metadata-only HF discovery. Payload downloads use the existing HF/Xet client.
use super::Source;
use crate::model::{ByteSource, DataSpan, ModelDescription, ObjectId, ObjectInfo};
use anyhow::{Context, Result, ensure};
use cherenkov_model_data::{ContainerFormat, Inventory, read_safetensors};
use reqwest::{
    StatusCode,
    blocking::Client,
    header::{CONTENT_RANGE, RANGE},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    io::{Read, Write},
    time::Duration,
};

const METADATA_LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Deserialize)]
struct Repo {
    sha: String,
    siblings: Vec<Sibling>,
}
#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

pub(super) fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".into())
        .trim_end_matches('/')
        .into()
}

pub(super) fn inspect(
    repo: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<(Source, ModelDescription)> {
    let token = token
        .map(|value| Ok(Some(value.to_owned())))
        .unwrap_or_else(ambient_token)?;
    let hub = Hub {
        client: Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?,
        endpoint: endpoint(),
        token,
    };

    hub.inspect(repo, revision)
}

struct Hub {
    client: Client,
    endpoint: String,
    token: Option<String>,
}

impl Hub {
    fn get(&self, url: reqwest::Url) -> reqwest::blocking::RequestBuilder {
        let request = self.client.get(url);

        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    fn url(&self, components: &[&str]) -> Result<reqwest::Url> {
        let mut url = reqwest::Url::parse(&self.endpoint)?;

        ensure!(
            matches!(url.scheme(), "http" | "https")
                && url.username().is_empty()
                && url.password().is_none(),
            "HF endpoint must be an HTTP(S) URL without credentials"
        );
        url.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid HF endpoint"))?
            .pop_if_empty()
            .extend(components);

        Ok(url)
    }

    fn json(&self, components: &[&str]) -> Result<Value> {
        let response = self.get(self.url(components)?).send()?.error_for_status()?;
        let mut bytes = Vec::new();

        response.take(METADATA_LIMIT + 1).read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= METADATA_LIMIT,
            "HF metadata exceeds 64 MiB"
        );

        Ok(serde_json::from_slice(&bytes)?)
    }

    fn inspect(&self, repo: &str, revision: &str) -> Result<(Source, ModelDescription)> {
        let (owner, name) = repo
            .split_once('/')
            .context("HF source must be owner/repo")?;
        let info: Repo = serde_json::from_value(
            self.json(&["api", "models", owner, name, "revision", revision])?,
        )?;

        ensure!(
            info.sha.len() == 40 && info.sha.bytes().all(|b| b.is_ascii_hexdigit()),
            "HF did not return a full commit"
        );

        let config = self.json(&[owner, name, "resolve", &info.sha, "config.json"])?;
        let indexed = info
            .siblings
            .iter()
            .any(|s| s.rfilename == "model.safetensors.index.json");
        let index = if indexed {
            self.json(&[
                owner,
                name,
                "resolve",
                &info.sha,
                "model.safetensors.index.json",
            ])?
        } else {
            Value::Null
        };
        let files: BTreeSet<String> = if indexed {
            index["weight_map"]
                .as_object()
                .context("missing safetensors weight_map")?
                .values()
                .map(|v| {
                    v.as_str()
                        .context("invalid shard filename")
                        .map(str::to_owned)
                })
                .collect::<Result<_>>()?
        } else {
            info.siblings
                .into_iter()
                .filter_map(|s| s.rfilename.ends_with(".safetensors").then_some(s.rfilename))
                .collect()
        };

        ensure!(
            !files.is_empty(),
            "HF registration requires a safetensors checkpoint"
        );

        let files: Vec<_> = files.into_iter().collect();
        let mut objects = Vec::new();

        for (id, file) in files.iter().enumerate() {
            ensure!(
                crate::storage::safe_component(file),
                "unsupported shard path {file:?}"
            );

            let url = self.url(&[owner, name, "resolve", &info.sha, file])?;
            let (prefix, size) = self.range(url.clone(), 0, 8)?;

            objects.push(RemoteObject {
                id: ObjectId(id),
                url,
                prefix,
                size,
            });
        }

        let reader = RemoteReader { hub: self, objects };
        let mut tensors = Vec::new();
        let mut names = BTreeSet::new();
        let mut metadata = Vec::new();

        for (id, file) in files.iter().enumerate() {
            eprintln!("inspect {repo}: shard {}/{}", id + 1, files.len());

            let shard = read_safetensors(&reader, ObjectId(id))?;

            for tensor in &shard.tensors {
                ensure!(
                    names.insert(tensor.name.clone()),
                    "duplicate tensor {}",
                    tensor.name
                );

                if indexed {
                    ensure!(
                        index["weight_map"][&tensor.name].as_str() == Some(file),
                        "shard index mismatch"
                    );
                }
            }

            metadata.push(shard.metadata);
            tensors.extend(shard.tensors);
        }

        if indexed {
            ensure!(
                names.len() == index["weight_map"].as_object().unwrap().len(),
                "incomplete shard index"
            );
        }

        let inventory = Inventory {
            format: ContainerFormat::Safetensors,
            metadata: json!({"config": config, "index": index, "shards": metadata}),
            tensors,
        };
        let description = crate::model::inspect::describe_inventory(&inventory, &reader, &config)?;

        Ok((
            Source::HuggingFace {
                repo: repo.into(),
                revision: info.sha,
                endpoint: self.endpoint.clone(),
            },
            description,
        ))
    }

    fn range(&self, mut url: reqwest::Url, offset: u64, length: u64) -> Result<(Vec<u8>, u64)> {
        ensure!(
            length > 0 && length <= METADATA_LIMIT,
            "invalid metadata range length"
        );

        let end = offset.checked_add(length - 1).context("range overflow")?;

        url.query_pairs_mut()
            .append_pair("header_range", &format!("{offset}-{end}"));

        let response = self
            .get(url)
            .header(RANGE, format!("bytes={offset}-{end}"))
            .send()?
            .error_for_status()?;

        ensure!(
            response.status() == StatusCode::PARTIAL_CONTENT,
            "HF ignored the byte range; refusing a full shard read"
        );

        let range = response
            .headers()
            .get(CONTENT_RANGE)
            .context("missing content range")?
            .to_str()?;
        let (bounds, total) = range.split_once('/').context("invalid content range")?;

        ensure!(
            bounds == format!("bytes {offset}-{end}"),
            "incorrect content range"
        );

        let total = total.parse()?;
        let mut bytes = Vec::new();

        response.take(length + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() as u64 == length, "incomplete metadata range");

        Ok((bytes, total))
    }
}

struct RemoteObject {
    id: ObjectId,
    url: reqwest::Url,
    prefix: Vec<u8>,
    size: u64,
}
struct RemoteReader<'a> {
    hub: &'a Hub,
    objects: Vec<RemoteObject>,
}

impl ByteSource for RemoteReader<'_> {
    fn objects(&self) -> Vec<ObjectInfo> {
        self.objects
            .iter()
            .map(|o| ObjectInfo {
                id: o.id,
                bytes: o.size,
            })
            .collect()
    }
    fn read(&self, span: &DataSpan, out: &mut dyn Write) -> Result<()> {
        let object = self
            .objects
            .get(span.object.0)
            .context("unknown HF shard")?;

        ensure!(
            span.offset
                .checked_add(span.length)
                .is_some_and(|end| end <= object.size),
            "range exceeds HF shard"
        );

        if span.offset == 0 && span.length == 8 {
            out.write_all(&object.prefix)?;

            return Ok(());
        }

        let (bytes, size) = self
            .hub
            .range(object.url.clone(), span.offset, span.length)?;

        ensure!(size == object.size, "HF shard changed during inspection");
        out.write_all(&bytes)?;

        Ok(())
    }
}

fn ambient_token() -> Result<Option<String>> {
    if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        .is_ok_and(|v| matches!(v.to_uppercase().as_str(), "1" | "ON" | "YES" | "TRUE"))
    {
        return Ok(None);
    }

    if let Ok(token) = std::env::var("HF_TOKEN") {
        return Ok(Some(token));
    }

    let home = std::env::var_os("HF_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache/huggingface")));
    let path = std::env::var_os("HF_TOKEN_PATH")
        .map(std::path::PathBuf::from)
        .or_else(|| home.map(|h| h.join("token")));
    let Some(path) = path else {
        return Ok(None);
    };

    match std::fs::read_to_string(path) {
        Ok(token) => Ok(Some(token.trim().to_owned())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/model/index/hub.rs"]
mod tests;
