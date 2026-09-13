use anyhow::{Result, ensure};
use std::{fs, path::Path, process::Command};
use xtask::docs;

const BOOK: &str = include_str!("../../book.toml");
const ATTRIBUTES: &str = include_str!("../../.gitattributes");

fn git(root: &Path, args: &[&str]) -> Result<()> {
    let output = Command::new("git").args(args).current_dir(root).output()?;

    ensure!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    Ok(())
}

/// Track the initial files before a test mutates the working tree.
fn track_files(root: &Path) -> Result<()> {
    git(root, &["init", "--quiet"])?;

    git(root, &["add", "."])
}

fn write_files(root: &Path, files: &[(&str, &str)]) -> Result<()> {
    for (name, text) in files {
        let path = root.join(name);

        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, text)?;
    }

    Ok(())
}

#[test]
fn stages_tracked_docs_and_assets_from_the_working_tree() -> Result<()> {
    let repo = tempfile::tempdir()?;
    let site = tempfile::tempdir()?;
    let root = repo.path();
    let gallery = "<a href=\"summary.md\">Report</a>";
    let files = [
        ("book.toml", BOOK),
        ("SUMMARY.md", "# Summary\n\n[Home](README.md)\n"),
        ("README.md", "# Original\n"),
        ("LICENSE", "license"),
        ("cherenkov.example.toml", "[server]"),
        ("benchmarks/suite.json", "{}"),
        ("docs/third-party-notices.txt", "notice"),
        ("results/saved/gallery.html", gallery),
        ("results/saved/pelicans/exact.svg", "<svg/>"),
        ("results/saved/outputs/code.txt", "answer"),
        ("src/main.rs", "fn main() {}"),
        ("results/weights.bin", "private"),
    ];

    write_files(root, &files)?;
    fs::write(root.join(".gitattributes"), ATTRIBUTES)?;
    track_files(root)?;
    fs::write(root.join("README.md"), "# Edited\n")?;
    fs::write(root.join("results/local.html"), "unreviewed run")?;
    docs::stage(root, site.path())?;

    let content = site.path();

    assert_eq!(fs::read_to_string(content.join("README.md"))?, "# Edited\n");
    assert_eq!(
        fs::read_to_string(root.join("results/saved/gallery.html"))?,
        gallery
    );

    for name in [
        "src/main.rs",
        "results/weights.bin",
        "results/local.html",
        "results/saved/gallery.html",
    ] {
        assert!(!content.join(name).exists(), "unexpected site file: {name}");
    }

    for (name, text) in files {
        if matches!(
            name,
            "book.toml"
                | "README.md"
                | "results/saved/gallery.html"
                | "src/main.rs"
                | "results/weights.bin"
        ) {
            continue;
        }

        assert_eq!(fs::read_to_string(content.join(name))?, text, "{name}");
    }

    let config = fs::read_to_string(site.path().join("book.toml"))?;

    assert_eq!(config, BOOK);

    Ok(())
}

#[test]
fn missing_tracked_document_fails_the_build() -> Result<()> {
    let repo = tempfile::tempdir()?;
    let site = tempfile::tempdir()?;
    let root = repo.path();

    fs::write(root.join("book.toml"), BOOK)?;
    fs::write(root.join(".gitattributes"), ATTRIBUTES)?;
    fs::write(root.join("README.md"), "# Home\n")?;
    track_files(root)?;
    fs::remove_file(root.join("README.md"))?;

    assert!(docs::stage(root, site.path()).is_err());

    Ok(())
}

#[test]
fn attributes_select_files_without_changing_relative_paths() -> Result<()> {
    let repo = tempfile::tempdir()?;
    let site = tempfile::tempdir()?;
    let root = repo.path();

    write_files(
        root,
        &[
            ("book.toml", BOOK),
            (".gitattributes", "/notes/**/*.txt site\n"),
            ("notes/deep/example file.txt", "included"),
            ("notes/README.md", "excluded"),
        ],
    )?;
    track_files(root)?;
    docs::stage(root, site.path())?;

    assert_eq!(
        fs::read_to_string(site.path().join("notes/deep/example file.txt"))?,
        "included"
    );
    assert!(!site.path().join("notes/README.md").exists());
    assert_eq!(fs::read_to_string(site.path().join("book.toml"))?, BOOK);

    Ok(())
}

#[test]
fn publishes_complete_rustdoc_tree_and_removes_stale_api_files() -> Result<()> {
    let rustdoc = tempfile::tempdir()?;
    let site = tempfile::tempdir()?;
    let files = [
        ("crates.js", "crate index"),
        ("cherenkov/index.html", "engine"),
        ("cherenkov_model_data/index.html", "model data"),
        ("xtask/index.html", "automation"),
        ("static.files/main-hash.js", "script"),
        ("search.index/name/chunk.js", "search shard"),
        ("src/cherenkov/metal.rs.html", "source"),
        ("trait.impl/core/clone/trait.Clone.js", "implementations"),
    ];

    write_files(rustdoc.path(), &files)?;
    write_files(
        site.path(),
        &[
            ("index.html", "book"),
            ("docs/rust-api.html", "reference chapter"),
            ("api/removed_crate/index.html", "stale"),
        ],
    )?;
    docs::publish_api(rustdoc.path(), site.path())?;

    for (name, content) in files {
        assert_eq!(
            fs::read_to_string(site.path().join("api").join(name))?,
            content
        );
    }

    assert!(!site.path().join("api/removed_crate").exists());
    assert_eq!(fs::read_to_string(site.path().join("index.html"))?, "book");
    assert_eq!(
        fs::read_to_string(site.path().join("docs/rust-api.html"))?,
        "reference chapter"
    );
    assert!(
        fs::read_to_string(site.path().join("api/index.html"))?
            .contains("href=\"../docs/rust-api.html\"")
    );

    Ok(())
}

#[test]
fn missing_rustdoc_output_preserves_existing_api() -> Result<()> {
    let rustdoc = tempfile::tempdir()?;
    let site = tempfile::tempdir()?;

    write_files(site.path(), &[("api/index.html", "previous build")])?;

    assert!(docs::publish_api(rustdoc.path(), site.path()).is_err());
    assert_eq!(
        fs::read_to_string(site.path().join("api/index.html"))?,
        "previous build"
    );

    Ok(())
}
