use crate::gym::{Action};
use crate::search::{Arena, State, SearchResult, Node};
use std::sync::{Mutex, Condvar};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::fmt::{Debug};
use numpy::ndarray::{Array};
use rand::seq::SliceRandom;


/// Core algorithm of MCTS, with all its functionality. Shareable between threads.
#[derive(Debug)]
pub struct MCTSCore {
    /// Exploration coefficient c, for trade-off between exploration and exploitation used in UCT
    expl_coeff: f32,

    /// Discount Factor Lambda
    discount_factor: f32,

    /// Whether to use a virtual loss to increase more exploration between threads
    use_virtual_loss: bool,

    /// Number of players in the environment
    num_players: u32,

    /// Arena, to store all nodes
    arena: Arena,

    /// How many threads have started their iterations
    iters_start: AtomicUsize,

    /// How many threads have finished their iterations
    iters_end: AtomicUsize,

    /// Mutex blocks until search finished
    result: Mutex<Option<SearchResult>>,

    /// Conditional variable that sends a signal, once the search is finished
    result_ready: Condvar,
}

impl MCTSCore {
    pub fn new(state: State, capacity: usize) -> MCTSCore {
        let arena = Arena::new(capacity);

        // Add root node
        let legal_acts = MCTSCore::get_legal_actions(&state);
        arena.add_node(state, None, legal_acts, None).expect("Could not add root node!");

        MCTSCore {
            expl_coeff: 2.0,
            discount_factor: 1.0,
            use_virtual_loss: true,
            num_players: 2,
            arena,
            iters_start: AtomicUsize::new(0),
            iters_end: AtomicUsize::new(0),
            result: Mutex::new(None),
            result_ready: Condvar::new(),
        }
    }

    /// Search for the given number of iterations
    pub fn search(&self, num_iters: usize) {
        let mut iter = self.iters_start.fetch_add(1, Ordering::AcqRel);
        let mut iters_end = 0;

        while iter < num_iters {
            let leaf = self.select();
            let new_leaf = self.expand(leaf);

            if new_leaf.is_err() {
                self.remove_virtual_loss(leaf);
                continue;
            }
            let new_leaf = new_leaf.unwrap();

            let ret = self.simulate(&new_leaf);
            self.backpropagate(&new_leaf, ret);

            iter = self.iters_start.fetch_add(1, Ordering::AcqRel);
            iters_end = self.iters_end.fetch_add(1, Ordering::AcqRel);
        }

        self.write_result((iters_end+1) == num_iters);
    }

    /// Get root node
    #[inline]
    pub fn get_root(&self) -> &Node {
        let root = self.arena.get_node(0);
        root.virtual_loss.fetch_add(1, Ordering::AcqRel);
        root
    }

    #[inline]
    pub fn select(&self) -> &Node {
        let mut node = self.get_root();

        while node.is_fully_expanded() {
            node = node.select_child(self.expl_coeff, &self.arena, self.use_virtual_loss);
        }
        node
    }

    #[inline]
    pub fn expand<'a>(&'a self, node: &'a Node) -> Result<&'a Node, &'static str> {
        if node.is_terminal() {
            return Ok(node);
        }

        // Get unique index for child expansion
        let act_idx = node.get_action_index();
        if act_idx >= node.num_acts.load(Ordering::Acquire) {
            return Err("No more actions to expand!");
        }

        // Create new child node
        let action = node.get_unexpanded_action(act_idx);
        let state = node.state.borrow().unwrap();
        let next_state = state.transition(action);
        let next_legal_acts = MCTSCore::get_legal_actions(&next_state);
        let new_node = self.arena.add_node(next_state, Some(action), next_legal_acts, Some(node)).expect("Arena is full!");

        // Finalize by adding child to parent
        self.arena.add_child(node, new_node, act_idx);

        Ok(new_node)
    }

    #[inline]
    pub fn simulate(&self, node: &Node) -> f32 {
        if node.is_terminal() {
            return 0.0;
        }

        node.rollout(self.discount_factor, self.num_players)
    }

    #[inline]
    pub fn backpropagate(&self, node: &Node, ret: f32) {
        let mut node = node;
        let mut curr_ret = ret;

        loop {
            // Remove virtual loss
            node.virtual_loss.fetch_sub(1, Ordering::AcqRel);

            // Add current return
            let rew = node.state.borrow().unwrap().rew;
            curr_ret = rew + self.discount_factor * curr_ret;
            node.add_stats(curr_ret, 1);

            // Flip for next player
            let flip = if self.num_players == 2 { -1.0f32 } else { 1.0f32 };
            curr_ret *= flip;

            // Switch to parent
            let parent_id = node.parent_id.borrow().unwrap();
            if let Some(id) = parent_id {
                node = self.arena.get_node(*id);
            } else {
                break;
            }
        }
    }

    /// Remove the virtual loss from the node and its parent
    #[inline]
    pub fn remove_virtual_loss(&self, node: &Node) {
        let mut node = node;

        loop {
            // Remove virtual loss
            node.virtual_loss.fetch_sub(1, Ordering::AcqRel);

            // Switch to parent
            let parent_id = node.parent_id.borrow().unwrap();
            if let Some(id) = parent_id {
                node = self.arena.get_node(*id);
            } else {
                break;
            }
        }
    }

    /// Write search result and notify
    pub fn write_result(&self, should_write: bool) {
        if should_write {
            if let Ok(mut result) = self.result.lock() {
                *result = Some(self.get_search_result());
                self.result_ready.notify_one();
            }
        }
    }

    /// Get search result from root node. Value, Q function as well as policy.
    pub fn get_search_result(&self) -> SearchResult {
        let root = self.get_root();
        let num_acts = root.num_acts.load(Ordering::Acquire);

        let size = root.state.borrow().unwrap().env.size as usize;
        let v = root.get_v();
        let mut q = Array::from_elem(size*size, -1_000_000f32);
        let mut pi = Array::from_elem(size*size, -1_000_000f32);
        for child_id in root.children.borrow().unwrap().iter() {
            let child = self.arena.get_node(child_id.load(Ordering::Acquire));
            let act = child.action.borrow().unwrap().unwrap();

            q[act as usize] = child.get_v();
            pi[act as usize] = child.get_v();
        }

        // Softmax for pi
        pi = pi.mapv(|x| x.exp());
        let pi_sum = pi.sum();
        pi = pi / pi_sum;


        SearchResult { pi, q, v }
    }

    /// Get legal actions of state, the order is important! Children are added based on order. Random shuffle for MCTS
    #[inline]
    pub fn get_legal_actions(state: &State) -> Vec<Action> {
        let legal_act = &state.info.legal_act;
        let mut acts = legal_act.iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| i as Action).collect::<Vec<Action>>();
        let mut rng = rand::thread_rng();
        acts.shuffle(&mut rng);

        acts
    }

    /// Block until the result is written and we are notified
    #[inline]
    pub fn wait_for_result(&self) -> SearchResult {
        let mut result = self.result.lock().unwrap();
        while result.is_none() {
            result = self.result_ready.wait(result).unwrap();
        }

        result.take().unwrap()
    }
}