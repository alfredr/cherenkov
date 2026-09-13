//! Read each shard's prefix and header together, with bounded parallel requests.

use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{SyncSender, sync_channel},
};

const CONCURRENCY: usize = 8;
type Shard = (RemoteObject, Inventory);

struct Request<'a> {
    id: ObjectId,
    file: &'a str,
    url: reqwest::Url,
}

impl Hub {
    /// Workers only fetch data. The caller emits events and restores file order.
    pub(super) fn headers(
        &self,
        root: &[&str],
        files: &[String],
        events: &mut dyn FnMut(ModelEvent),
    ) -> Result<Vec<Shard>> {
        let requests = files
            .iter()
            .enumerate()
            .map(|(id, file)| {
                ensure!(
                    crate::storage::safe_component(file),
                    "unsupported shard path {file:?}"
                );

                let mut components = root.to_vec();

                components.push(file);

                Ok(Request {
                    id: ObjectId(id),
                    file,
                    url: self.url(&components)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let workers = CONCURRENCY.min(requests.len());

        events(ModelEvent::Headers {
            completed: 0,
            total: files.len(),
            file: None,
        });

        std::thread::scope(|scope| {
            let (sender, receiver) = sync_channel(workers);

            for _ in 0..workers {
                let sender = sender.clone();
                let requests = &requests;
                let next = &next;
                let failed = &failed;

                std::thread::Builder::new()
                    .name("hf-header".into())
                    .spawn_scoped(scope, move || {
                        self.fetch_headers(requests, next, failed, sender)
                    })
                    .context("starting HF header worker")?;
            }

            drop(sender);

            let mut shards = Vec::with_capacity(files.len());
            let mut error = None;

            for (id, result) in receiver {
                match result {
                    Ok(shard) => {
                        shards.push((id, shard));
                        events(ModelEvent::Headers {
                            completed: shards.len(),
                            total: files.len(),
                            file: Some(files[id].clone()),
                        });
                    }
                    Err(failure) => {
                        error.get_or_insert(failure);
                    }
                }
            }

            if let Some(error) = error {
                return Err(error);
            }

            shards.sort_unstable_by_key(|(id, _)| *id);

            Ok(shards.into_iter().map(|(_, shard)| shard).collect())
        })
    }

    /// Stop claiming shards after an error; finish and drain shards already claimed.
    fn fetch_headers(
        &self,
        requests: &[Request<'_>],
        next: &AtomicUsize,
        failed: &AtomicBool,
        sender: SyncSender<(usize, Result<Shard>)>,
    ) {
        while !failed.load(Ordering::Relaxed) {
            let id = next.fetch_add(1, Ordering::Relaxed);
            let Some(request) = requests.get(id) else {
                break;
            };
            let result = self
                .header(request)
                .with_context(|| format!("reading shard {}", request.file));

            if result.is_err() {
                failed.store(true, Ordering::Relaxed);
            }

            if sender.send((id, result)).is_err() {
                break;
            }
        }
    }

    fn header(&self, request: &Request<'_>) -> Result<Shard> {
        let (prefix, size) = self.range(request.url.clone(), 0, 8)?;
        let object = RemoteObject {
            id: request.id,
            url: request.url.clone(),
            prefix,
            size,
        };
        let reader = RemoteReader {
            hub: self,
            objects: std::slice::from_ref(&object),
        };
        let shard = read_safetensors(&reader, object.id)?;

        Ok((object, shard))
    }
}
