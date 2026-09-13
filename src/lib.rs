//! Standalone qwen4-exp text inference on Apple Silicon.

pub mod config;
pub mod control;
pub mod download;
mod kernels;
mod metal;
pub mod model;
mod nn;
pub mod options;
mod prefix_cache;
mod prompt;
mod quant;
pub mod qwen4_exp;
pub mod runner;
pub mod sampling;
pub mod server;
pub mod storage;
mod tensors;
mod tok;
mod units;

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod test_support;
