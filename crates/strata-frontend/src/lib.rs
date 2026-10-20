//! The text side of a request, as PolyStrata's Python server does it: an OpenAI request normalized into the chat
//! template's messages, tools and kwargs (`request`), literal special-token strings inside messages kept as text
//! (`literals`), and the prompt rendered and tokenized (`prompt`). Checked against golden files recorded from the
//! Python code.

pub mod literals;
pub mod output;
pub mod prompt;
pub mod request;

/// A request the server answers with 400.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct RequestError(pub String);

pub type Result<T> = std::result::Result<T, RequestError>;

pub use prompt::{Prompt, PromptBuilder};
pub use request::{openai_to_messages, Prepared, ToolOrders};
