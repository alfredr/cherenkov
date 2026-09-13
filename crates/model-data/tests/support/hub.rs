use anyhow::{Context, Result, ensure};
use cherenkov_model_data::{ByteSource, DataSpan, ObjectId, ObjectInfo};
use reqwest::{
    StatusCode,
    blocking::Client,
    header::{CONTENT_RANGE, RANGE},
};
use serde::Deserialize;
use serde_json::Value;
use std::{
    io::{Read, Write},
    time::Duration,
};

pub struct Hub {
    client: Client,
    base: String,
    pub config: Value,
    pub index: Value,
    pub files: Vec<String>,
}

#[derive(Deserialize)]
struct Repo {
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

impl Hub {
    pub fn open(repo: &str, revision: &str) -> Result<Self> {
        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;
        let metadata: Repo = client
            .get(format!(
                "https://huggingface.co/api/models/{repo}/revision/{revision}"
            ))
            .send()?
            .error_for_status()?
            .json()?;
        let base = format!("https://huggingface.co/{repo}/resolve/{revision}");
        let config = client
            .get(format!("{base}/config.json"))
            .send()?
            .error_for_status()?
            .json()?;
        let indexed = metadata
            .siblings
            .iter()
            .any(|s| s.rfilename == "model.safetensors.index.json");
        let index: Value = if indexed {
            client
                .get(format!("{base}/model.safetensors.index.json"))
                .send()?
                .error_for_status()?
                .json()?
        } else {
            Value::Null
        };
        let mut files: Vec<_> = if let Some(map) = index["weight_map"].as_object() {
            map.values()
                .map(|v| v.as_str().context("invalid shard name").map(str::to_owned))
                .collect::<Result<_>>()?
        } else {
            metadata
                .siblings
                .into_iter()
                .filter_map(|s| s.rfilename.ends_with(".safetensors").then_some(s.rfilename))
                .collect()
        };

        files.sort();
        files.dedup();
        ensure!(!files.is_empty(), "no safetensors shards");

        Ok(Self {
            client,
            base,
            config,
            index,
            files,
        })
    }

    pub fn shard(&self, id: usize) -> Result<Shard> {
        let url = format!("{}/{}", self.base, self.files[id]);
        let (prefix, size) = range(&self.client, &url, 0, 8)?;

        Ok(Shard {
            client: self.client.clone(),
            url,
            id: ObjectId(id),
            size,
            prefix,
        })
    }
}

pub struct Shard {
    client: Client,
    url: String,
    id: ObjectId,
    size: u64,
    prefix: Vec<u8>,
}

impl ByteSource for Shard {
    fn objects(&self) -> Vec<ObjectInfo> {
        vec![ObjectInfo {
            id: self.id,
            bytes: self.size,
        }]
    }

    fn read(&self, span: &DataSpan, output: &mut dyn Write) -> Result<()> {
        ensure!(span.object == self.id, "unknown remote object");
        ensure!(
            span.offset
                .checked_add(span.length)
                .is_some_and(|end| end <= self.size),
            "remote span exceeds shard"
        );

        if span.offset == 0 && span.length == 8 {
            output.write_all(&self.prefix)?;

            return Ok(());
        }

        let (bytes, size) = range(&self.client, &self.url, span.offset, span.length)?;

        ensure!(size == self.size, "remote object size changed");
        output.write_all(&bytes)?;

        Ok(())
    }
}

fn range(client: &Client, url: &str, offset: u64, length: u64) -> Result<(Vec<u8>, u64)> {
    ensure!(
        length > 0 && length <= 64 * 1024 * 1024,
        "invalid header read size"
    );

    let end = offset.checked_add(length - 1).context("range overflow")?;
    // Distinct URLs prevent intermediaries from reusing a cached 0-7 response.
    let response = client
        .get(format!("{url}?header_range={offset}-{end}"))
        .header(RANGE, format!("bytes={offset}-{end}"))
        .send()?
        .error_for_status()?;

    ensure!(
        response.status() == StatusCode::PARTIAL_CONTENT,
        "server ignored byte range; refusing full weight download"
    );

    let range = response
        .headers()
        .get(CONTENT_RANGE)
        .context("missing content range")?
        .to_str()?;
    let (bounds, total) = range.split_once('/').context("invalid content range")?;

    ensure!(
        bounds == format!("bytes {offset}-{end}"),
        "wrong content range: {range}"
    );

    let size = total.parse()?;
    let mut bytes = Vec::new();

    response.take(length + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 == length, "incomplete range response");

    Ok((bytes, size))
}
