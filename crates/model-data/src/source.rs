use crate::{DataSpan, ObjectId};
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use std::{fs::File, io::Write, path::Path, sync::Arc};

/// An object's identity and total size within a byte source.
#[derive(Debug, Clone)]
pub struct ObjectInfo {
    /// Identity assigned by the source.
    pub id: ObjectId,
    /// Object length in bytes.
    pub bytes: u64,
}

/// Object IDs belong to one source; they are neither paths nor content hashes.
/// A copy/import operation must rebind them to its destination's identities.
pub trait ByteSource {
    /// List the objects addressable through this source.
    fn objects(&self) -> Vec<ObjectInfo>;
    /// Write the requested byte range to `output`, or return an error.
    fn read(&self, span: &DataSpan, output: &mut dyn Write) -> Result<()>;

    /// Return a retained view, or `None` if this source cannot map bytes.
    /// Invalid ranges on a mappable source return an error.
    fn map(&self, _span: &DataSpan) -> Result<Option<MappedBytes>> {
        Ok(None)
    }
}

/// Read-only file mappings, numbered in the order passed to [`Self::open`].
#[derive(Clone, Default)]
pub struct MappedObjects {
    mappings: Vec<Arc<Mmap>>,
}

/// A byte range that keeps its backing mapping alive after the source is dropped.
/// The backing file must remain unchanged for the lifetime of every clone.
#[derive(Clone)]
pub struct MappedBytes {
    backing: Arc<Mmap>,
    range: std::ops::Range<usize>,
}

impl AsRef<[u8]> for MappedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.backing[self.range.clone()]
    }
}

impl MappedObjects {
    /// Map files read-only, assigning object IDs from zero in iteration order.
    /// Files must remain unchanged while any returned mapping is alive.
    pub fn open<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Result<Self> {
        let mappings = paths
            .into_iter()
            .map(|path| {
                let file =
                    File::open(path).with_context(|| format!("opening {}", path.display()))?;
                // The caller owns the immutable-file contract, including views
                // retained after this source is dropped.
                let map = unsafe { Mmap::map(&file) }
                    .with_context(|| format!("mapping {}", path.display()))?;

                Ok(Arc::new(map))
            })
            .collect::<Result<_>>()?;

        Ok(Self { mappings })
    }

    /// Borrow an entire object mapping; fail if the object index is out of range.
    /// The caller must use IDs from this source.
    pub fn mapping(&self, id: ObjectId) -> Result<&Mmap> {
        self.mappings
            .get(id.0)
            .map(AsRef::as_ref)
            .context("unknown checkpoint object")
    }

    /// Borrow a byte range after checking its object ID and bounds.
    pub fn bytes(&self, span: &DataSpan) -> Result<&[u8]> {
        let map = self.mapping(span.object)?;
        let range = checked_range(span, map.len())?;

        Ok(&map[range])
    }

    /// Retain a byte range without copying its contents, checking ID and bounds.
    pub fn view(&self, span: &DataSpan) -> Result<MappedBytes> {
        let backing = self
            .mappings
            .get(span.object.0)
            .context("unknown checkpoint object")?
            .clone();
        let range = checked_range(span, backing.len())?;

        Ok(MappedBytes { backing, range })
    }
}

impl ByteSource for MappedObjects {
    fn objects(&self) -> Vec<ObjectInfo> {
        self.mappings
            .iter()
            .enumerate()
            .map(|(id, map)| ObjectInfo {
                id: ObjectId(id),
                bytes: map.len() as u64,
            })
            .collect()
    }

    fn read(&self, span: &DataSpan, output: &mut dyn Write) -> Result<()> {
        output.write_all(self.bytes(span)?)?;

        Ok(())
    }

    fn map(&self, span: &DataSpan) -> Result<Option<MappedBytes>> {
        Ok(Some(self.view(span)?))
    }
}

fn checked_range(span: &DataSpan, size: usize) -> Result<std::ops::Range<usize>> {
    let end = span
        .offset
        .checked_add(span.length)
        .context("data span overflow")?;

    ensure!(end <= size as u64, "data span exceeds checkpoint object");

    Ok(span.offset as usize..end as usize)
}
