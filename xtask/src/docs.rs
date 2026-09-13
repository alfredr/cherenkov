//! Build the mdBook guide and workspace rustdoc as one static site.

use crate::util;
use anyhow::{Context, Result, ensure};
use std::{fs, path::Path, process::Command};

pub fn build() -> Result<()> {
    let root = util::root();
    let staging = tempfile::tempdir()?;
    let target = root.join("target/site-rustdoc");

    // Keep the nested Cargo invocation separate from the running xtask's cache.
    // Rustdoc output is cumulative, so remove old docs while retaining builds.
    let status = util::command(&["cargo", "clean", "--doc", "--target-dir"])
        .arg(&target)
        .status()
        .context("cleaning previous rustdoc output")?;

    ensure!(status.success(), "rustdoc cleanup failed");

    let status = util::command(&[
        "cargo",
        "doc",
        "--locked",
        "--workspace",
        "--no-deps",
        "--document-private-items",
        "--target-dir",
    ])
    .arg(&target)
    .env("CARGO_ENCODED_RUSTDOCFLAGS", "-Dwarnings")
    .status()
    .context("building workspace rustdoc")?;

    ensure!(status.success(), "rustdoc build failed");

    stage(&root, staging.path())?;

    let status = util::command(&["mdbook", "build"])
        .arg(staging.path())
        .arg("--dest-dir")
        .arg(root.join("_site"))
        .status()
        .context("running mdbook; install it with mise install github:rust-lang/mdBook")?;

    ensure!(status.success(), "mdBook build failed");
    publish_api(&target.join("doc"), &root.join("_site"))?;

    Ok(())
}

/// Publish the whole rustdoc tree, including shared search and source assets.
pub fn publish_api(source: &Path, site: &Path) -> Result<()> {
    let destination = site.join("api");

    // Fail before replacing the previous API if generation produced no docs.
    ensure!(
        source.join("crates.js").is_file(),
        "missing rustdoc crate index"
    );

    if destination.exists() {
        fs::remove_dir_all(&destination)?;
    }

    for path in util::files(source)? {
        let target = destination.join(path.strip_prefix(source)?);

        fs::create_dir_all(target.parent().context("rustdoc file parent")?)?;
        fs::copy(path, target)?;
    }

    fs::write(
        destination.join("index.html"),
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">\
         <title>Rust API</title>\
         <meta http-equiv=\"refresh\" content=\"0; url=../docs/rust-api.html\">\
         <a href=\"../docs/rust-api.html\">Rust API reference</a></html>\n",
    )?;

    Ok(())
}

pub fn stage(root: &Path, destination: &Path) -> Result<()> {
    let tracked = Command::new("git")
        .args(["ls-files", "-z", "--", ":(attr:site)"])
        .current_dir(root)
        .output()?;
    let paths = util::checked(tracked)?;

    fs::create_dir_all(destination)?;
    fs::copy(root.join("book.toml"), destination.join("book.toml"))?;

    // Use working-tree contents, but only tracked files: local benchmark runs
    // and model downloads must never become website assets.
    for name in paths.split('\0').filter(|name| !name.is_empty()) {
        let path = Path::new(name);
        let target = destination.join(path);

        fs::create_dir_all(target.parent().context("site file parent")?)?;
        fs::copy(root.join(path), &target)?;
    }

    Ok(())
}
