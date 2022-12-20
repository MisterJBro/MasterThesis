use std::ops::Deref;
use std::thread;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering, AtomicBool};
use std::fmt::{Debug, Formatter};
use numpy::ndarray::{Array, Ix1};
use hexgame::gym::{Env, Envs, Action, Obs, Info, Infos, Episode};
use crossbeam::channel::{unbounded, bounded, Sender, Receiver};
use atomic_array::{AtomicOptionRefArray, AtomicUsizeArray, AtomicRefArray};
use atomic_float::AtomicF32;
use atomic_ref::AtomicRef;
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
        let mcts = Arc::new(MCTSCore::new(state));

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
    arena: Arena,
}

impl MCTSCore {
    pub fn new(state: State) -> MCTSCore {
        let mut arena = Arena::new(1000);
        arena.add_node(state);

        MCTSCore {
            expl_coeff: 2.0,
            arena,
        }
    }

    pub fn search(&self, iters: usize) {
        for i in 0..iters {
            let leaf = self.select();
            let new_leaf = self.expand(leaf);
            let ret = self.simulate(new_leaf);
            self.backpropagate(new_leaf, ret);
        }

        //self.get_search_result()
    }

    pub fn get_root(&self) -> Node {
        self.arena.nodes[0]
    }

    pub fn select(&self) -> Node {
        let mut node = self.get_root();

        while node.is_fully_expanded() {
            node = node.select_child(self.expl_coeff, &self.arena);
        }
        node
    }

    pub fn expand(&self, node: Node) -> Node {
        if node.is_terminal() {
            return node;
        }

        // Create new child node
        let action = node.get_legal_action();
        let state = node.state.borrow().unwrap();
        let next_state = state.transition(action);
        let legal_acts = self.get_legal_actions(state);
        let new_node = self.arena.add_node(next_state, action, legal_acts).unwrap();


    }

    pub fn simulate(&self, leaf: NodeID) -> f32 {
        0.0
    }

    pub fn backpropagate(&self, leaf: NodeID, ret: f32) {

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
    pub fn get_legal_actions(&self, state: &State) -> Vec<Action> {
        let legal_act = state.info.legal_act;
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
    pub action: AtomicLazyCell<Action>,
    pub state: AtomicLazyCell<State>,
    pub children: AtomicLazyCell<Vec<AtomicUsize>>,
    pub legal_acts: AtomicLazyCell<Vec<Action>>,
    pub child_idx: AtomicUsize,
    pub is_terminal: AtomicBool,
    pub is_fully_expanded: AtomicBool,
}
impl Node {
    pub fn new() -> Node {
        Node {
            stats: AtomicUsize::new(0),
            num_acts: AtomicUsize::new(0),
            action: AtomicLazyCell::new(),
            state: AtomicLazyCell::new(),
            children: AtomicLazyCell::new(),
            legal_acts: AtomicLazyCell::new(),
            child_idx: AtomicUsize::new(0),
            is_terminal: AtomicBool::new(false),
            is_fully_expanded: AtomicBool::new(false),
        }
    }

    /// Checks if node is fully expanded, and not a leaf
    #[inline]
    pub fn is_fully_expanded(&self) -> bool {
        self.is_fully_expanded.load(Ordering::Acquire)
    }

    /// Get sum_returns as fixed point float and num_visits from stats atomic
    #[inline]
    pub fn from_stats(&self) -> (u32, f32) {
        let stats = self.stats.load(Ordering::Acquire);
        let num_visits = (stats >> 40) as u32;

        // Get sum returns from fixed point float to ieee float
        let sum_returns_bits = ((stats as i64) << 24) >> 24;
        let sum_returns_fixed = FixedI64::<U13>::from_bits(sum_returns_bits);
        let sum_returns = sum_returns_fixed.saturating_to_num::<f32>();

        (num_visits, sum_returns)
    }

    /// Combine sum_returns and num_visits to stats atomic
    #[inline]
    pub fn to_stats(num_visits: u32, sum_returns: f32) -> usize {
        let num_visits = (num_visits as usize) << 40;

        // Get sum returns from ieee float to fixed point float
        let sum_returns_fixed = FixedI64::<U13>::saturating_from_num(sum_returns);
        let sum_returns_bits = sum_returns_fixed.to_bits() as usize;
        let sum_returns_bits_trim = (sum_returns_bits << 24) >> 24;

        let stats = sum_returns_bits_trim | num_visits;
        stats
    }

    /// Select one of the child by using uct
    #[inline]
    pub fn select_child(&self, c: f32, arena: &Arena) -> Node {
        let (num_visits, _) = self.from_stats();

        // Get child ids
        let mut children = self.children.borrow().unwrap();
        let child_ids = children
            .iter()
            .map(|child| child.load(Ordering::Acquire))
            .collect::<Vec<_>>();

        // Get uct values
        let mut ucts = child_ids.iter().map(|childID| {
            let child = arena.nodes[*childID];
            child.uct(c, num_visits as f32)
        }).collect::<Vec<_>>();

        // Get max
        let max = ucts.iter()
            .max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
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

        arena.nodes[idx]
    }

    /// Upper Confidence Bound for Trees
    #[inline]
    pub fn uct(&self, c: f32, parent_visits: f32) -> f32 {
        let (num_visits, sum_returns) = self.from_stats();
        if num_visits == 0 {
            return f32::INFINITY;
        }

        sum_returns / num_visits as f32 + c * (parent_visits.ln() / num_visits as f32).sqrt()
    }

    /// Value function
    #[inline]
    pub fn get_V(&self) -> f32 {
        let (num_visits, sum_returns) = self.from_stats();
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

    /// Get action from info legal act at index
    #[inline]
    pub fn get_legal_action(&self, index: usize) -> Action {
        let legal_acts = self.legal_acts.borrow().unwrap();
        legal_acts[index]
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
        let next_env = self.env.clone();
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
    pub fn add_node(&self, state: State, action: Action, legal_act: Vec<Action>) -> Result<NodeID, &'static str> {
        let curr_idx = self.idx.fetch_add(1, Ordering::AcqRel);

        if curr_idx < self.capacity {
            let node = self.nodes[curr_idx];
            let num_acts = state.num_actions();
            node.action.fill(action).unwrap();
            node.state.fill(state).unwrap();
            node.children.fill((0..num_acts).into_iter().map(|_| AtomicUsize::new(0)).collect()).unwrap();
            node.num_acts.store(num_acts, Ordering::Release);
            node.is_terminal.store(state.done, Ordering::Release);
            node.legal_acts.fill(legal_act).unwrap();

            Ok(curr_idx)
        } else {
            Err("Arena is full!")
        }
  }

    /// Add child node to parent
    #[inline]
    pub fn add_child(&self, parentID: NodeID, childID: NodeID, action: Action) {
        let parent = self.nodes[parentID];
        let mut children = parent.children.write().unwrap();
        let new_child = NodeChild {
            action,
            childID,
        };
        (*children).push(new_child);
    }

    // Get node
    //#[inline]
    //pub fn get_node(&self, id: NodeID) -> Option<Node<T>> {
    //}

}

impl Debug for Arena {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "Arena {{ nodes: {:?} }}", self.nodes.len())
    }
}


fn main () {
    // Env
    let mut env = Env::new(5);
    let (mut obs, mut info) = env.reset();

    // MCTS
    //let mcts = MCTS::new(1);
    //let state = State { obs, info, env };
    //mcts.search(state, 1);

    // Create a new arena
    let arena = Arc::new(Arena::new(4, 4));

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

