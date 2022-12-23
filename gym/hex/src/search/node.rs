use crate::gym::{Action};
use crate::search::{Arena, State,};
use std::sync::atomic::{AtomicUsize, Ordering, AtomicBool};
use rand::Rng;
use lazycell::AtomicLazyCell;
use fixed::{types::extra::U13, FixedI64};
use rand::seq::SliceRandom;


/// Node of Tree.
pub struct Node {
    /// Stores both, sum_returns (upper 40 bits as fixed point float) and num_visits (lower 24 bits as u32). So both are updated atomically
    pub stats: AtomicUsize,
    pub num_acts: AtomicUsize,
    pub action: AtomicLazyCell<Option<Action>>,
    pub state: AtomicLazyCell<State>,
    pub legal_acts: AtomicLazyCell<Vec<Action>>,
    pub act_idx: AtomicUsize,
    pub children: AtomicLazyCell<Vec<AtomicUsize>>,
    pub num_children: AtomicUsize,
    pub is_terminal: AtomicBool,
    pub parent_id: AtomicLazyCell<Option<usize>>,
    pub arena_id: AtomicUsize,
    pub virtual_loss: AtomicUsize,
}
impl Node {
    pub fn new() -> Node {
        Node {
            stats: AtomicUsize::new(0),
            num_acts: AtomicUsize::new(0),
            action: AtomicLazyCell::new(),
            state: AtomicLazyCell::new(),
            legal_acts: AtomicLazyCell::new(),
            act_idx: AtomicUsize::new(0),
            children: AtomicLazyCell::new(),
            num_children: AtomicUsize::new(0),
            is_terminal: AtomicBool::new(false),
            parent_id: AtomicLazyCell::new(),
            arena_id: AtomicUsize::new(0),
            virtual_loss: AtomicUsize::new(0),
        }
    }

    /// Checks if node is fully expanded, and not a leaf
    #[inline]
    pub fn is_fully_expanded(&self) -> bool {
        self.num_children.load(Ordering::Acquire) == self.num_acts.load(Ordering::Acquire)
    }

    /// Get sum_returns as fixed point float and num_visits from stats atomic
    #[inline]
    pub fn from_stats(&self) -> (f32, u32) {
        let stats = self.stats.load(Ordering::Acquire);
        // Get sum returns from fixed point float to ieee float
        let sum_returns_bits = (stats as i64) >> 24;
        let sum_returns_fixed = FixedI64::<U13>::from_bits(sum_returns_bits);
        let sum_returns = sum_returns_fixed.saturating_to_num::<f32>();

        let num_visits = ((stats << 40) >> 40) as u32;
        (sum_returns, num_visits)
    }

    /// Combine sum_returns and num_visits to stats atomic
    #[inline]
    pub fn to_stats(sum_returns: f32, num_visits: u32) -> usize {
        // Get sum returns from ieee float to fixed point float
        let sum_returns_fixed = FixedI64::<U13>::saturating_from_num(sum_returns);
        let sum_returns_bits = sum_returns_fixed.to_bits() as usize;
        let sum_returns_bits_trim = sum_returns_bits << 24;

        let num_visits = ((num_visits as usize) << 40) >> 40;

        let stats = sum_returns_bits_trim | num_visits;
        stats
    }

    /// Updates stats atomically
    pub fn add_stats(&self, sum_returns: f32, num_visits: u32) {
        let to_add = Node::to_stats(sum_returns, num_visits);
        self.stats.fetch_add(to_add, Ordering::AcqRel);
    }

    /// Select one of the child by using uct
    #[inline]
    pub fn select_child<'a>(&self, c: f32, arena: &'a Arena, use_virtual_loss: bool) -> &'a Node {
        let (_, num_visits) = self.from_stats();
        let virtual_loss = if use_virtual_loss { self.virtual_loss.load(Ordering::Acquire) as f32 } else { 0f32 };

        // Get child ids
        let children = self.children.borrow().unwrap();
        let child_ids = children
            .iter()
            .map(|child| child.load(Ordering::Acquire))
            .collect::<Vec<_>>();

        // Get uct values
        let ucts = child_ids.iter().map(|child_id| {
            let child = arena.get_node(*child_id);
            child.uct(c, num_visits as f32, virtual_loss)
        }).collect::<Vec<_>>();

        // Get max
        let max = ucts.iter()
            .max_by(|x, y| x.total_cmp(y))
            .unwrap();

        // Get indices of max
        let max_indices = ucts.iter()
            .enumerate()
            .filter(|(_, v)| **v == *max)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        // Get index randomly
        let idx = if max_indices.len() == 1 {
            max_indices[0]
        } else {
            let mut rng = rand::thread_rng();
            let rand_idx = rng.gen_range(0..max_indices.len());
            max_indices[rand_idx]
        };
        let idx = child_ids[idx];

        // Get node
        let next_node = arena.get_node(idx);
        next_node.virtual_loss.fetch_add(1, Ordering::AcqRel);
        next_node
    }

    /// Upper Confidence Bound for Trees
    #[inline]
    pub fn uct(&self, c: f32, parent_visits: f32, virtual_loss: f32) -> f32 {
        let (sum_returns, num_visits) = self.from_stats();
        if num_visits == 0 {
            return f32::INFINITY;
        }

        (sum_returns - virtual_loss) / (num_visits as f32 + virtual_loss) + c * (parent_visits.ln() / (num_visits as f32 + virtual_loss)).sqrt()
    }

    /// Value function
    #[inline]
    pub fn get_v(&self) -> f32 {
        let (sum_returns, num_visits) = self.from_stats();
        if num_visits == 0 {
            return 0.0;
        }
        sum_returns / num_visits as f32
    }

    /// Is terminal node
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.is_terminal.load(Ordering::Acquire)
    }

    /// Get action index
    #[inline]
    pub fn get_action_index(&self) -> usize {
        self.act_idx.fetch_add(1, Ordering::AcqRel)
    }

    /// Get action which is still unexpanded
    #[inline]
    pub fn get_unexpanded_action(&self, index: usize) -> Action {
        let legal_acts = self.legal_acts.borrow().unwrap();
        legal_acts[index]
    }

    /// Do a random rollout play
    #[inline]
    pub fn rollout(&self, discount_factor: f32, num_players: u32) -> f32 {
        let state = self.state.borrow().unwrap();
        let mut env = state.env.clone();
        let mut _obs;
        let (mut rew, mut done, mut info) = (state.rew, state.done, state.info.clone());
        let mut rng = rand::thread_rng();
        let mut ret = 0f32;
        let mut player = 0u32;

        // Simulate
        while !done {
            player = (player + 1) % num_players;

            // Action
            let acts = info.legal_act.iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| i as Action).collect::<Vec<Action>>();
            let act = *acts.choose(&mut rng).unwrap();

            // Step
            (_obs, rew, done, info) = env.step(act);

            // Return
            if num_players == 2 {
                if player == 0 {
                    ret = rew + discount_factor * ret;
                } else {
                    ret = -rew + discount_factor * ret;
                }
            } else {
                ret = rew + discount_factor * ret;
            }
        }

        ret
    }

}

/// Reference for nodes within arena, just an index
pub type NodeID = usize;