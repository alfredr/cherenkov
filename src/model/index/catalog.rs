use super::{Catalog, ModelIndex};
use anyhow::{Context, Result, ensure};
use std::{fs::File, io::Write, path::Component};

pub(super) struct LockedCatalog {
    pub value: Catalog,
    index: ModelIndex,
    _lock: File,
}

impl ModelIndex {
    pub(super) fn lock(&self) -> Result<LockedCatalog> {
        crate::storage::create_private_dir(&self.paths.data)?;

        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.paths.data.join("index.lock"))?;

        lock.lock()?;

        let path = self.paths.data.join("index.json");
        let value: Catalog = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("reading model index")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Catalog::default(),
            Err(e) => return Err(e.into()),
        };

        validate(&value)?;

        Ok(LockedCatalog {
            value,
            index: self.clone(),
            _lock: lock,
        })
    }
}

impl LockedCatalog {
    pub fn save(&self) -> Result<()> {
        let temporary = self.index.paths.data.join("index.json.tmp");
        let mut file = File::create(&temporary)?;

        serde_json::to_writer(&mut file, &self.value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        std::fs::rename(temporary, self.index.paths.data.join("index.json"))?;
        File::open(&self.index.paths.data)?.sync_all()?;

        Ok(())
    }
}

fn validate(catalog: &Catalog) -> Result<()> {
    ensure!(
        catalog.version == 1,
        "unsupported model index version {}",
        catalog.version
    );

    for (id, entry) in &catalog.models {
        ensure!(valid_id(id) && id == &entry.id, "invalid model ID");

        for artifact in entry.prepared.iter().chain(&entry.retained_source) {
            ensure!(
                catalog.artifacts.contains_key(artifact),
                "missing indexed artifact {artifact}"
            );
        }
    }

    for (name, store) in &catalog.stores {
        ensure!(
            name == &store.name && crate::storage::safe_component(name),
            "invalid store name"
        );
        ensure!(
            valid_id(&store.id) && store.path.is_absolute(),
            "invalid store registration"
        );
    }

    for (id, artifact) in &catalog.artifacts {
        ensure!(valid_id(id) && id == &artifact.id, "invalid artifact ID");
        ensure!(
            artifact
                .entry
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
            "artifact entry must stay within its store"
        );

        if let Some(path) = &artifact.external {
            ensure!(
                path.is_absolute(),
                "external artifact path must be absolute"
            );
        }
    }

    Ok(())
}

fn valid_id(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())
}
