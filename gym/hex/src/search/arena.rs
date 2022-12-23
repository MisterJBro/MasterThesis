use crate::gym::{Action};
use crate::search::{Node, State};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::fmt::{Debug, Formatter};


/// Arena for managing nodes in a tree. Used for MCTS and its variants, so no delete operation. Shareable between threads
pub struct Arena {
    /// The nodes are stored within a Vector. The vector is pre-initialized with empty Nodes. The size of the vector cannot be increased later on.
    nodes: Vec<Node>,

    /// Current idx i.e. how many nodes are currently in the tree
    idx: AtomicUsize,

    /// Maximum number of nodes, the tree can hold. Hard limit
    capacity: usize,
}

impl Arena {
    /// Creates a new empty Arena.
    pub fn new(capacity: usize) -> Arena {
        Arena {
            nodes: (0..capacity).into_iter().map(|_| Node::new()).collect(),
            idx: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Checks if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.idx.load(Ordering::Acquire) == 0
    }

    /// Add a new node to the arena, if capacity is reached, return Err
    #[inline]
    pub fn add_node<'a>(&'a self, state: State, action: Option<Action>, next_legal_act: Vec<Action>, parent: Option<&'a Node>) -> Result<&'a Node, &'static str> {
        let curr_idx = self.idx.fetch_add(1, Ordering::AcqRel);

        if curr_idx < self.capacity {
            let node = &self.nodes[curr_idx];
            let num_acts = state.num_actions();
            node.is_terminal.store(state.done, Ordering::Release);
            node.action.fill(action).unwrap();
            node.state.fill(state).unwrap();
            node.children.fill((0..num_acts).into_iter().map(|_| AtomicUsize::new(0)).collect()).unwrap();
            node.num_acts.store(num_acts, Ordering::Release);
            node.legal_acts.fill(next_legal_act).unwrap();
            node.arena_id.store(curr_idx, Ordering::Release);

            if let Some(parent) = parent {
                node.parent_id.fill(Some(parent.arena_id.load(Ordering::Acquire))).unwrap();
            } else {
                node.parent_id.fill(None).unwrap();
            }

            Ok(node)
        } else {
            Err("Arena is full!")
        }
  }

    /// Add child node to parent, then increment num of children
    #[inline]
    pub fn add_child(&self, parent: &Node, child: &Node, act_idx: usize) {
        let children = parent.children.borrow().unwrap();
        let child_id = child.arena_id.load(Ordering::Acquire);
        children[act_idx].store(child_id, Ordering::Release);
        parent.num_children.fetch_add(1, Ordering::AcqRel);
    }

    /// Get node by idx
    #[inline]
    pub fn get_node(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }

}

impl Debug for Arena {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "Arena {{ nodes: {:?} }}", self.nodes.len())
    }
}