//! Machine description and a disk read sample for benchmark provenance.
//!
//! Portable facts come from libc: kernel, memory, CPU count, and the volume
//! holding the model. GPU and drive identity come from `system_profiler` on
//! macOS and are absent elsewhere. Only hardware-class fields are recorded;
//! serial numbers, host names and user paths stay out so a report can be
//! shared.

use crate::util;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    ffi::CStr,
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

/// Bytes sampled from the expert store to estimate uncached read bandwidth.
pub const READ_SAMPLE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Parallel readers for the sample, matching the engine's per-record fan-out.
pub const READ_SAMPLE_THREADS: usize = 8;
const READ_BLOCK: usize = 8 * 1024 * 1024;

/// Describe this machine and the volume holding `model`. A probe that fails
/// records its error in place instead of aborting the benchmark.
pub fn describe(model: &Path) -> Value {
    let store = model.join("packed/experts.bin");
    let read = if store.is_file() {
        read_rate(&store, READ_SAMPLE_BYTES, READ_SAMPLE_THREADS)
    } else {
        Err(anyhow::anyhow!(
            "no packed expert store at <model>/packed/experts.bin"
        ))
    };

    json!({
        "kernel": kernel(),
        "os_version": os_version(),
        "memory_bytes": physical_memory(),
        "cpu_threads": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "profile": or_error(profile()),
        "model_volume": or_error(volume(model)),
        "store_read": or_error(read),
    })
}

fn or_error(result: Result<Value>) -> Value {
    result.unwrap_or_else(|e| json!({"error": e.to_string()}))
}

/// `sysname release machine`, the fields of `uname -srm`.
pub fn kernel() -> Option<String> {
    let mut name: libc::utsname = unsafe { std::mem::zeroed() };

    if unsafe { libc::uname(&mut name) } != 0 {
        return None;
    }

    let field = |bytes: &[libc::c_char]| {
        unsafe { CStr::from_ptr(bytes.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };

    Some(format!(
        "{} {} {}",
        field(&name.sysname),
        field(&name.release),
        field(&name.machine)
    ))
}

/// Installed physical memory in bytes.
pub fn physical_memory() -> Option<u64> {
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };

    (pages > 0 && page > 0).then(|| pages as u64 * page as u64)
}

/// Product version of the operating system, where the platform reports one.
fn os_version() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        util::output(&["sw_vers", "-productVersion"]).ok()
    }

    #[cfg(not(target_os = "macos"))]
    {
        std::fs::read_to_string("/etc/os-release")
            .ok()?
            .lines()
            .find_map(|line| line.strip_prefix("PRETTY_NAME="))
            .map(|name| name.trim_matches('"').to_owned())
    }
}

/// Chip, GPU and drive identity. macOS exposes these through
/// `system_profiler`; other platforms record nothing here.
fn profile() -> Result<Value> {
    #[cfg(target_os = "macos")]
    {
        let text = util::output(&[
            "system_profiler",
            "SPHardwareDataType",
            "SPDisplaysDataType",
            "SPNVMeDataType",
            "-json",
        ])?;

        Ok(profile_fields(
            &serde_json::from_str(&text).context("system_profiler JSON")?,
        ))
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(Value::Null)
    }
}

/// Pick the hardware-class fields from `system_profiler -json` output.
pub fn profile_fields(value: &Value) -> Value {
    let hardware = &value["SPHardwareDataType"][0];
    let gpu = &value["SPDisplaysDataType"][0];
    let drives: Vec<Value> = value["SPNVMeDataType"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|controller| controller["_items"].as_array())
        .flatten()
        .map(|drive| {
            json!({
                "model": drive["device_model"],
                "size": drive["size"],
                "size_bytes": drive["size_in_bytes"],
            })
        })
        .collect();

    json!({
        "machine": hardware["machine_name"],
        "model_identifier": hardware["machine_model"],
        "chip": hardware["chip_type"],
        "cpu_cores": hardware["number_processors"],
        "memory": hardware["physical_memory"],
        "gpu": gpu["sppci_model"],
        "gpu_cores": gpu["sppci_cores"],
        "metal_family": gpu["spdisplays_mtlgpufamilysupport"],
        "nvme": drives,
    })
}

/// Capacity and free space of the file system holding `path`.
pub fn volume(path: &Path) -> Result<Value> {
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };

    ensure!(
        unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } == 0,
        "statvfs failed: {}",
        std::io::Error::last_os_error()
    );

    let fragment = stat.f_frsize as u64;

    Ok(json!({
        "total_bytes": stat.f_blocks as u64 * fragment,
        "free_bytes": stat.f_bavail as u64 * fragment,
    }))
}

/// Read `bytes` from `path` past the page cache with `threads` parallel
/// readers at spread offsets, and report the achieved rate in decimal GB/s.
pub fn read_rate(path: &Path, bytes: u64, threads: usize) -> Result<Value> {
    ensure!(threads > 0, "read sample needs a reader");

    let len = path.metadata()?.len();

    ensure!(len > 0, "empty file");

    let total = bytes.min(len);
    let per_thread = (total / threads as u64).max(1);
    let stride = len / threads as u64;
    let started = Instant::now();
    let done: u64 = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|i| {
                let path: PathBuf = path.to_owned();
                let offset = (stride * i as u64).min(len.saturating_sub(per_thread));

                scope.spawn(move || read_uncached(&path, offset, per_thread))
            })
            .collect();

        workers
            .into_iter()
            .map(|worker| worker.join().unwrap_or(Ok(0)))
            .sum::<Result<u64>>()
    })?;
    let seconds = started.elapsed().as_secs_f64();

    ensure!(done > 0 && seconds > 0.0, "read sample produced no data");

    Ok(json!({
        "file": path.file_name().map(|f| f.to_string_lossy().into_owned()),
        "bytes": done,
        "seconds": seconds,
        "gbps": done as f64 / seconds / 1e9,
        "threads": threads,
        "method": "uncached 8 MiB reads at spread offsets; a warm-machine sample, not a cold-disk benchmark",
    }))
}

fn read_uncached(path: &Path, offset: u64, bytes: u64) -> Result<u64> {
    use std::os::unix::{fs::FileExt, io::AsRawFd};

    let file = File::open(path)?;

    bypass_cache(file.as_raw_fd(), offset, bytes);

    let mut buffer = vec![0u8; READ_BLOCK];
    let mut done = 0u64;

    while done < bytes {
        let want = buffer.len().min((bytes - done) as usize);
        let read = file.read_at(&mut buffer[..want], offset + done)?;

        if read == 0 {
            break;
        }

        done += read as u64;
    }

    Ok(done)
}

/// Advisory: keep the sample off the page cache so it measures the device.
#[cfg(target_os = "macos")]
fn bypass_cache(fd: libc::c_int, _offset: u64, _bytes: u64) {
    unsafe {
        libc::fcntl(fd, libc::F_NOCACHE, 1);
    }
}

#[cfg(target_os = "linux")]
fn bypass_cache(fd: libc::c_int, offset: u64, bytes: u64) {
    unsafe {
        libc::posix_fadvise(
            fd,
            offset as libc::off_t,
            bytes as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn bypass_cache(_fd: libc::c_int, _offset: u64, _bytes: u64) {}

/// Zip a results directory beside itself for sharing; returns the archive
/// path. Entries sit under the directory's name so the archive unpacks to a
/// folder wherever it is sent.
pub fn archive(out: &Path) -> Result<PathBuf> {
    use std::io::Write;

    let parent = out.parent().context("results directory parent")?;
    let name = out
        .file_name()
        .context("results directory name")?
        .to_string_lossy()
        .into_owned();
    let zip = parent.join(format!("{name}.zip"));
    let mut writer = zip::ZipWriter::new(File::create(&zip)?);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .large_file(true);

    for path in util::files(out)? {
        let relative = path.strip_prefix(out)?;

        writer.start_file(format!("{name}/{}", relative.to_string_lossy()), options)?;
        writer.write_all(&std::fs::read(&path)?)?;
    }

    writer.finish()?;

    Ok(zip)
}
