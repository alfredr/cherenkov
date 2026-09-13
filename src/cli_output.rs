//! Command reports share numeric view data with future interactive clients.

pub(crate) mod dash;
pub(crate) mod models;
mod report;
mod stats;
mod terminal;

use anyhow::Result;
use cherenkov::control::Command;
use report::{Report, Section, TableData};
use serde_json::Value;
use std::io::{self, IsTerminal};

pub fn print_stats(command: &Command, response: &Value, json: bool) -> Result<()> {
    if json || !io::stdout().is_terminal() {
        println!("{}", format_stats(command, response, json)?);

        return Ok(());
    }

    terminal::print(&stats::View::from_response(command, response)?.report())
}

fn format_stats(command: &Command, response: &Value, json: bool) -> Result<String> {
    if json {
        return Ok(serde_json::to_string_pretty(response)?);
    }

    Ok(stats::View::from_response(command, response)?
        .report()
        .markdown())
}

#[cfg(test)]
#[path = "../tests/unit/cli_output.rs"]
mod tests;
