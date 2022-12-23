//! Search algorithms like Monte Carlo Tree Search, AlphaZero etc.

mod arena;
mod mcts_core;
mod mcts;
mod node;
mod state;

pub use crate::search::arena::{Arena};
pub use crate::search::mcts_core::{MCTSCore};
pub use crate::search::mcts::{MCTS, SearchResult};
pub use crate::search::node::{Node, NodeID};
pub use crate::search::state::{State};