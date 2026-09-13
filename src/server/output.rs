//! Slow clients consume bounded writer queues, never the GPU worker thread.

use super::{
    ApiKind, registry::Ticket, request::Request, response::Response, sessions::Commit,
    tool_call::WireToolCall,
};
use crate::control::State;
use anyhow::{Result, bail};
use serde_json::Value;
use std::{
    net::TcpStream,
    sync::{Arc, mpsc},
    time::{SystemTime, UNIX_EPOCH},
};

pub(super) enum Frame {
    Text(String),
    Finish {
        text: String,
        reason: &'static str,
        usage: Value,
        turn: Option<Box<Commit>>,
        tool_calls: Option<Vec<WireToolCall>>,
    },
    Error(String),
}

pub(super) struct Output {
    sender: mpsc::SyncSender<Frame>,
    ticket: Arc<Ticket>,
}

impl Output {
    pub(super) fn new(
        mut stream: TcpStream,
        kind: ApiKind,
        request: &Request,
        ticket: Arc<Ticket>,
        state: Arc<State>,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel(16);
        let writer_ticket = ticket.clone();
        let (streaming, include_usage) = (request.stream, request.include_usage);

        std::thread::spawn(move || {
            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let mut response = Response::for_writer(
                &mut stream,
                kind,
                &writer_ticket.id,
                created,
                streaming,
                include_usage,
            );
            let result = write_frames(&mut response, receiver, &writer_ticket);

            if result.is_err() {
                writer_ticket.cancel();
            }

            state.update(|s| {
                if writer_ticket.cancelled() {
                    s.cancelled_requests += 1;
                } else if matches!(result, Ok(true)) {
                    s.completed_requests += 1;
                } else {
                    s.failed_requests += 1;
                }
            });
        });

        Self { sender, ticket }
    }

    #[cfg(test)]
    pub(super) fn for_test(ticket: Arc<Ticket>) -> (Self, mpsc::Receiver<Frame>) {
        let (sender, receiver) = mpsc::sync_channel(16);

        (Self { sender, ticket }, receiver)
    }

    pub(super) fn send(&self, frame: Frame) -> Result<()> {
        if self.sender.try_send(frame).is_err() {
            self.ticket.cancel();
            bail!("client disconnected or output queue is full");
        }

        Ok(())
    }
}

pub(super) fn write_frames(
    response: &mut Response<'_, impl std::io::Write>,
    receiver: mpsc::Receiver<Frame>,
    ticket: &Ticket,
) -> Result<bool> {
    response.start()?;

    for frame in receiver {
        match frame {
            Frame::Text(text) => response.text(&text)?,
            Frame::Error(message) => {
                response.fail(&message)?;

                return Ok(false);
            }
            Frame::Finish {
                text,
                reason,
                usage,
                turn,
                tool_calls,
            } => {
                // Cancellation must win before publication. Dropping an unpublished
                // turn rolls it back before the final response is written.
                let completed = ticket.complete();

                if let Some(turn) = turn.filter(|_| completed) {
                    turn.publish();
                }

                response.finish(
                    &text,
                    tool_calls.as_deref(),
                    if completed { reason } else { "cancelled" },
                    usage,
                )?;

                return Ok(completed);
            }
        }
    }

    bail!("output ended without a terminal response")
}

#[cfg(test)]
#[path = "../../tests/unit/server/output.rs"]
mod tests;
