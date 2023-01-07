//! Search algorithms like Monte Carlo Tree Search, AlphaZero etc.

mod arena;
mod mcts_core;
mod mcts;
mod az_core;
mod az;
mod evaluator;
mod node;
mod result;
mod state;

pub use crate::search::arena::{Arena};
pub use crate::search::mcts_core::{MCTSCore};
pub use crate::search::mcts::{MCTS};
pub use crate::search::az_core::{AZCore};
pub use crate::search::az::{AlphaZero};
pub use crate::search::evaluator::{Evaluator};
pub use crate::search::node::{Node, NodeID};
pub use crate::search::result::{SearchResult};
pub use crate::search::state::{State};
