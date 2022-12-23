use std::thread;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering, AtomicBool};
use std::fmt::{Debug, Formatter};
use numpy::ndarray::{Array, Ix1};
use hexgame::gym::{Env, Envs, Action, Obs, Info, Infos, Episode};
use crossbeam::channel::{unbounded, bounded, Sender, Receiver};
use rand::Rng;
use rand::seq::SliceRandom;
use lazycell::AtomicLazyCell;
use std::time::{Duration, Instant};
use fixed::{types::extra::U13, FixedI64};


// Worker Messages
#[derive(Debug)]
pub enum MCTSWorkerMessageIn {
    Search {mcts: Arc<MCTSCore>, iters: usize},
}

#[derive(Clone, Debug)]
pub struct MCTSWorkerMessageOut {
}

/// MCTS
pub struct MCTS {
    workers_ins: Vec<Sender<MCTSWorkerMessageIn>>,
    workers_out: Receiver<MCTSWorkerMessageOut>,
}

impl MCTS {
    pub fn new(num_threads: usize) -> MCTS {
        // Create Worker threads
        let mut workers = Vec::with_capacity(num_threads);
        let mut workers_ins = Vec::with_capacity(num_threads);
        let (master_sender, workers_out) = bounded(num_threads);
        for id in 0..num_threads {
            let (s, r) = bounded(1);
            workers_ins.push(s);
            workers.push(MCTSWorker::new(id, r, master_sender.clone()));
        }

        MCTS {
            workers_ins,
            workers_out,
        }
    }

    pub fn search(&self, state: State, iters: usize) {
        let mcts = Arc::new(MCTSCore::new(state, 10_000));

        for worker in &self.workers_ins {
            worker.send(MCTSWorkerMessageIn::Search{mcts: Arc::clone(&mcts), iters}).unwrap();
        }
    }
}

/// Worker thread for Envs
pub struct MCTSWorker {
    thread: Option<thread::JoinHandle<()>>,
}

impl MCTSWorker {
    pub fn new(id: usize, in_channel: Receiver<MCTSWorkerMessageIn>, out_channel: Sender<MCTSWorkerMessageOut>) -> MCTSWorker {

        let thread = thread::spawn(move || {
            loop {
                if let Ok(message) = in_channel.recv() {
                    match message {
                        MCTSWorkerMessageIn::Search{mcts, iters} => {
                            mcts.search(iters);
                        }
                    }
                }
            }
        });

        MCTSWorker {
            thread: Some(thread),
        }
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.thread.take() {
            handle.join().expect("Could not join mcts worker thread!");
        }
    }
}

pub struct SearchResult {
    Q: Array<f32, Ix1>,
    V: f32,
    pi: Array<f32, Ix1>,
}

#[derive(Debug)]
pub struct MCTSCore {
    expl_coeff: f32,
    discount_factor: f32,
    num_players: u32,
    arena: Arena,
    iters: AtomicUsize,
}

impl MCTSCore {
    pub fn new(state: State, capacity: usize) -> MCTSCore {
        let arena = Arena::new(capacity);
        let legal_acts = MCTSCore::get_legal_actions(&state);
        arena.add_node(state, None, legal_acts, None).expect("Could not add root node!");

        MCTSCore {
            expl_coeff: 2.0,
            discount_factor: 1.0,
            num_players: 2,
            arena,
            iters: AtomicUsize::new(0),
        }
    }

    pub fn search(&self, iters: usize) {
        loop {
            if self.iters.fetch_add(1, Ordering::AcqRel) >= iters {
                break;
            }

            let leaf = self.select();
            println!("\tSelected leaf: {:?}", leaf.arena_id);
            let new_leaf = self.expand(leaf);

            if new_leaf.is_err() {
                println!("\tLeaf Error: {:?}", new_leaf.err());
                continue;
            }
            let new_leaf = new_leaf.unwrap();
            println!("\tExpanded leaf: {:?}, act: {:?}", new_leaf.arena_id, new_leaf.action.borrow().unwrap());

            let ret = self.simulate(&new_leaf);
            self.backpropagate(&new_leaf, ret);
        }

        //self.get_search_result()
    }

    pub fn get_root(&self) -> &Node {
        &(self.arena.nodes[0])
    }

    pub fn select(&self) -> &Node {
        let mut node = self.get_root();

        while node.is_fully_expanded() {
            node = node.select_child(self.expl_coeff, &self.arena);
        }
        node
    }

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

    pub fn simulate(&self, node: &Node) -> f32 {
        if node.is_terminal() {
            return 0.0;
        }

        node.rollout(self.discount_factor, self.num_players)
    }

    pub fn backpropagate(&self, node: &Node, ret: f32) {
        let mut node = node;
        let mut curr_ret = ret;

        loop {
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
                node = &(self.arena.nodes[*id]);
            } else {
                break;
            }
        }
    }

    pub fn get_search_result(&self) -> SearchResult {
        SearchResult {
            Q: Array::zeros(1),
            V: 0.0,
            pi: Array::zeros(1),
        }
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
}

/// Node of Tree.
#[derive()]
pub struct Node {
    /// Stores both, sum_returns (lower 40 bits as fixed point float) and num_visits (upper 24 bits as u32). So both are updated atomically
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

    pub fn add_stats(&self, sum_returns: f32, num_visits: u32) {
        let to_add = Node::to_stats(sum_returns, num_visits);
        self.stats.fetch_add(to_add, Ordering::AcqRel);
    }

    /// Select one of the child by using uct
    #[inline]
    pub fn select_child<'a>(&self, c: f32, arena: &'a Arena) -> &'a Node {
        let (_, num_visits) = self.from_stats();

        // Get child ids
        let children = self.children.borrow().unwrap();
        let child_ids = children
            .iter()
            .map(|child| child.load(Ordering::Acquire))
            .collect::<Vec<_>>();

        // Get uct values
        let ucts = child_ids.iter().map(|child_id| {
            let child = &arena.nodes[*child_id];
            child.uct(c, num_visits as f32)
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

        &arena.nodes[idx]
    }

    /// Upper Confidence Bound for Trees
    #[inline]
    pub fn uct(&self, c: f32, parent_visits: f32) -> f32 {
        let (sum_returns, num_visits) = self.from_stats();
        if num_visits == 0 {
            return f32::INFINITY;
        }

        sum_returns / num_visits as f32 + c * (parent_visits.ln() / num_visits as f32).sqrt()
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

/// State description
#[derive(Clone, Debug)]
pub struct State {
    pub obs: Obs,
    pub rew: f32,
    pub done: bool,
    pub info: Info,
    pub env: Env,
}

impl State {
    pub fn new(obs: Obs, rew: f32, done: bool, info: Info, env: Env) -> State {
        State {
            obs,
            rew,
            done,
            info,
            env,
        }
    }

    /// Get number of legal actions
    #[inline]
    pub fn num_actions(&self) -> usize {
        self.info.legal_act.len()
    }

    /// Transition to the next state with the given action
    #[inline]
    pub fn transition(&self, action: Action) -> State {
        let mut next_env = self.env.clone();
        let (obs, rew, done, info) = next_env.step(action);
        State::new(obs, rew, done, info, next_env)
    }

    /// Transition inplace
    #[inline]
    pub fn transition_inplace(&mut self, action: Action) {
        (self.obs, self.rew, self.done ,self.info) = self.env.step(action);
    }

}

/// Arena for managing nodes in a tree. Used for MCTS and its variants, so no delete operation. Specialized for sharing between threads
pub struct Arena {
    nodes: Vec<Node>,
    idx: AtomicUsize,
    capacity: usize,
}

impl Arena {
    /// Creates a new empty `Arena`.
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
}

impl Debug for Arena {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "Arena {{ nodes: {:?} }}", self.nodes.len())
    }
}


fn main () {
    // Env
    let mut env = Env::new(2);
    let (mut obs, mut info) = env.reset();

    // MCTS
    let mcts = MCTS::new(3);
    let state = State { obs, rew: 0.0, done: false, info, env };
    mcts.search(state, 5);

    thread::sleep(std::time::Duration::from_secs(2));

    // Create a new arena
    //let arena = Arc::new(Arena::new(4));

    // Add some new nodes to the arena
    //let root = arena.add_node(Node1 {num_visits: 12});

    // Share tree between trees
    /*for i in 0..3 {
        let arena = Arc::clone(&arena);
        thread::spawn(move || {
            let new_node = arena.add_node(Node1 {num_visits: i});
            arena.add_child(0, new_node);
        });
    }*/

    // Sleep for a second
    //thread::sleep(std::time::Duration::from_secs(1));
    //println!("{:?}", arena);
}

