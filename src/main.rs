use anyhow::{Context, Result};
use cherenkov::{
    config::{self, Overrides, Source},
    control, download,
    options::Options,
    runner,
    storage::{DEFAULT_REPO, DEFAULT_REVISION, Paths},
};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

mod cli_output;
mod model_cli;

#[derive(Parser)]
#[command(
    name = "cherenkov",
    version,
    about = "qwen4-exp text inference on Apple Silicon",
    next_line_help = false,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// Put model data, scratch and config under one root instead of XDG directories
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[arg(required = true)]
    model_dir: Option<PathBuf>,
    #[arg(required = true)]
    prompt: Option<String>,
    #[command(flatten)]
    options: Options,
}

#[derive(Args)]
#[command(next_line_help = false)]
struct Serve {
    /// Server configuration file (TOML)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Print resolved TOML without loading the model
    #[arg(long)]
    print_config: bool,
    #[command(flatten)]
    overrides: Overrides,
}

#[derive(Subcommand)]
enum Command {
    /// Open a live dashboard for the resident server
    Dash {
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Serve OpenAI-compatible completions and local control commands
    Serve(Box<Serve>),
    /// Query the resident server
    Status {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Query per-layer or per-expert activity from the resident server
    Stats {
        #[command(subcommand)]
        target: StatsTarget,
        #[arg(long, global = true)]
        socket: Option<PathBuf>,
        /// Emit the full JSON response instead of the summary
        #[arg(long, global = true)]
        json: bool,
    },
    /// Inspect or reload the resident server's configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
        #[arg(long, global = true)]
        socket: Option<PathBuf>,
    },
    /// Show resolved data, scratch, config and default model locations
    Paths,
    /// Manage registered models and their prepared stores
    Model {
        #[command(subcommand)]
        action: model_cli::Action,
        #[arg(long, global = true)]
        json: bool,
    },
    /// Manage named filesystem stores
    Store {
        #[command(subcommand)]
        action: model_cli::StoreAction,
        #[arg(long, global = true)]
        json: bool,
    },
    /// Describe a checkpoint's tensors, storage layout and preparation requirements
    Inspect {
        /// Model alias, source URI, checkpoint directory or GGUF file
        path: PathBuf,
        /// Emit the complete tensor and storage description
        #[arg(long)]
        json: bool,
    },
    /// Deprecated: use `prepare SOURCE`
    Download {
        #[arg(default_value = DEFAULT_REPO)]
        repo: String,
        /// Branch, tag or commit; defaults to the measured checkpoint revision
        #[arg(long, default_value = DEFAULT_REVISION)]
        revision: String,
        /// Hugging Face access token (alternatively HF_TOKEN or an existing HF login)
        #[arg(long)]
        hf_token: Option<String>,
        /// Fetch configuration and tokenizer without the weight shards
        #[arg(long)]
        metadata_only: bool,
    },
    /// Register a source and prepare missing inference artifacts
    #[command(name = "prepare", alias = "pack")]
    Prepare {
        /// Model alias, source URI or checkpoint path
        #[arg(value_name = "SOURCE")]
        model_dir: Option<PathBuf>,
        /// Optional index alias for this model
        #[arg(long)]
        name: Option<String>,
        /// HF branch, tag or commit (or append @revision to the source URI)
        #[arg(long)]
        revision: Option<String>,
        /// Export to a new directory instead of the managed artifact store
        #[arg(long)]
        output: Option<PathBuf>,
        /// Expert precisions (4,3,2); comma-separated or repeated; reuse existing stores
        #[arg(long, default_value = "4", value_delimiter = ',', num_args = 1..,
            value_parser = clap::value_parser!(u32).range(2..=4))]
        experts: Vec<u32>,
        /// Retain source weights downloaded for an indexed model
        #[arg(long)]
        keep_source: bool,
        /// Hugging Face token for an indexed import (or HF_TOKEN)
        #[arg(long)]
        hf_token: Option<String>,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    Show,
    Reload,
}

#[derive(Args)]
struct PageArgs {
    /// First entry to return
    #[arg(long, default_value_t = 0)]
    offset: usize,
    /// Maximum entries per response (1-128)
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u16).range(1..=128))]
    limit: u16,
}

#[derive(Subcommand)]
enum StatsTarget {
    /// Show overall read rates and CPU/GPU phase totals
    Summary,
    /// Sum expert counters for each layer
    Layers(PageArgs),
    /// Show individual experts in a packed-table layer
    Experts {
        layer: usize,
        #[command(flatten)]
        page: PageArgs,
    },
}

impl StatsTarget {
    fn command(self) -> control::Command {
        match self {
            Self::Summary => control::Command::StatsSummary,
            Self::Layers(page) => control::Command::StatsLayers {
                offset: page.offset,
                limit: usize::from(page.limit),
            },
            Self::Experts { layer, page } => control::Command::StatsExperts {
                layer,
                offset: page.offset,
                limit: usize::from(page.limit),
            },
        }
    }
}

impl Serve {
    fn source(mut self, root: Option<PathBuf>) -> Result<Source> {
        self.overrides.root = root.as_deref().map(config::absolute).transpose()?;
        let path = match self.config {
            Some(path) => Some(config::absolute(&path)?),
            None => {
                let path = Paths::new(self.overrides.root.as_deref())?.config;

                match std::fs::metadata(&path) {
                    Ok(_) => Some(path),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                    Err(e) => return Err(e.into()),
                }
            }
        };
        self.overrides.model_dir = self
            .overrides
            .model_dir
            .as_deref()
            .map(|p| cherenkov::model::index::anchor_selector(p, &std::env::current_dir()?))
            .transpose()?;
        self.overrides.socket = self
            .overrides
            .socket
            .as_deref()
            .map(config::absolute)
            .transpose()?;

        Ok(Source {
            path,
            overrides: self.overrides,
        })
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Dash { socket }) => {
            cli_output::dash::run(socket.unwrap_or_else(config::default_socket))
        }
        Some(Command::Serve(args)) => {
            let print = args.print_config;
            let source = args.source(cli.root)?;

            if print {
                println!("{}", toml::to_string_pretty(&source.resolve()?)?);

                Ok(())
            } else {
                cherenkov::server::serve(source)
            }
        }
        Some(Command::Status { socket, json }) => {
            let result = control::query(
                &socket.unwrap_or_else(config::default_socket),
                control::Command::Status,
            )?;

            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                let s = &result["stats"];

                println!(
                    "{} | {} active, {} queued | {} completed, {} failed\n{} tokens | cache {} entries, {} bytes\nMetal {} bytes (last observation)",
                    if s["ready"] == true {
                        "ready"
                    } else {
                        "loading"
                    },
                    s["active_requests"],
                    s["queued_requests"],
                    s["completed_requests"],
                    s["failed_requests"],
                    s["generated_tokens"],
                    s["cache"]["entries"],
                    s["cache"]["bytes"],
                    s["memory"]["metal_allocated_bytes_observed"]
                );
            }

            Ok(())
        }
        Some(Command::Stats {
            target,
            socket,
            json,
        }) => {
            let command = target.command();
            let result = control::query(
                &socket.unwrap_or_else(config::default_socket),
                command.clone(),
            )?;

            cli_output::print_stats(&command, &result, json)
        }
        Some(Command::Config { action, socket }) => {
            let command = match action {
                ConfigAction::Show => control::Command::ConfigShow,
                ConfigAction::Reload => control::Command::ConfigReload,
            };
            let result = control::query(&socket.unwrap_or_else(config::default_socket), command)?;

            println!("{}", serde_json::to_string_pretty(&result)?);

            Ok(())
        }
        Some(Command::Model { action, json }) => {
            model_cli::run(Paths::new(cli.root.as_deref())?, action, json)
        }
        Some(Command::Store { action, json }) => {
            model_cli::store(Paths::new(cli.root.as_deref())?, action, json)
        }
        Some(Command::Inspect { path, json }) => {
            model_cli::inspect(Paths::new(cli.root.as_deref())?, &path, json)
        }
        Some(Command::Paths) => {
            let paths = Paths::new(cli.root.as_deref())?;

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "data": paths.data, "scratch": paths.scratch, "config": paths.config,
                    "default_model": cherenkov::storage::default_model_reference(),
                    "downloaded_model": paths.default_model(),
                }))?
            );

            Ok(())
        }
        Some(Command::Download {
            repo,
            revision,
            hf_token,
            metadata_only,
        }) => {
            eprintln!("warning: `download` is deprecated; use `prepare SOURCE`");

            let paths = Paths::new(cli.root.as_deref())?;

            // SAFETY: download is a standalone command. No worker or transfer
            // threads have started; Xet reads this setting when its session starts.
            if std::env::var_os("HF_XET_CACHE").is_none() {
                unsafe {
                    std::env::set_var("HF_XET_CACHE", paths.scratch.join("xet"));
                }
            }

            let model = download::run(
                &paths,
                download::Download {
                    repo: &repo,
                    revision: &revision,
                    token: hf_token.as_deref(),
                    metadata_only,
                },
            )?;

            println!("{}", model.display());

            Ok(())
        }
        Some(Command::Prepare {
            model_dir,
            name,
            revision,
            output,
            experts,
            keep_source,
            hf_token,
        }) => {
            let model_dir = match model_dir {
                Some(dir) => dir,
                None if revision.is_some() => PathBuf::from(format!("hf://{DEFAULT_REPO}")),
                None => PathBuf::from(cherenkov::storage::default_model_reference()),
            };

            model_cli::prepare(
                Paths::new(cli.root.as_deref())?,
                &model_dir,
                cherenkov::model::index::ResolveOptions {
                    name: name.as_deref(),
                    revision: revision.as_deref(),
                    token: hf_token.as_deref(),
                },
                output.as_deref(),
                &experts,
                keep_source,
            )
        }
        None => {
            let mut options = cli.options;

            options.validate()?;

            let model = cherenkov::model::index::resolve_runtime(
                Paths::new(cli.root.as_deref())?,
                &cli.model_dir.context("model directory required")?,
                &mut options,
            )?;

            runner::run(
                &model.path,
                &cli.prompt.context("prompt required")?,
                &options,
            )
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/cli.rs"]
mod tests;
