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
        let mut arena = Arena::new(1000, 128);
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
        while !node.is_leaf() {
            node = node.select_child(self.expl_coeff, &self.arena);
        }
        node
    }

    pub fn expand(&self, node: Node) -> Node {
        if node.is_terminal() {
            return node;
        }

        // Create new child node
        let action = node.get_rand_action();


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
}

/// Node of Tree.
#[derive()]
pub struct Node {
    pub num_visits: AtomicUsize,
    pub sum_returns: AtomicF32,
    pub num_acts: AtomicUsize,
    pub state: RwLock<Option<State>>,
    pub children: RwLock<Vec<NodeChild>>,
    pub child_idx: AtomicUsize,
    pub is_terminal: AtomicBool,
}
impl Node {
    pub fn new(state: State) -> Node {
        let num_acts = state.num_actions();
        Node {
            num_visits: AtomicUsize::new(0),
            sum_returns: AtomicF32::new(0.0),
            num_acts: AtomicUsize::new(num_acts),
            state: RwLock::new(Some(state)),
            children: RwLock::new(Vec::with_capacity(num_acts)),
            child_idx: AtomicUsize::new(0),
            is_terminal: AtomicBool::new(state.done),
        }
    }

    pub fn with_capacity(capacity: usize) -> Node {
        Node {
            num_visits: AtomicUsize::new(0),
            sum_returns: AtomicF32::new(0.0),
            num_acts: AtomicUsize::new(0),
            state: RwLock::new(None),
            children: RwLock::new(Vec::with_capacity(capacity)),
            child_idx: AtomicUsize::new(0),
            is_terminal: AtomicBool::new(false),
        }
    }

    /// Checks if node is leaf, leaf = having not all child nodes expanded
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.child_idx.load(Ordering::Acquire) == self.num_acts.load(Ordering::Acquire)
    }

    /// Select one of the child by using uct
    #[inline]
    pub fn select_child(&self, c: f32, arena: &Arena) -> Node {
        let num_vists = self.num_visits.load(Ordering::Acquire);

        // Get all child ids
        let mut children = self.children.read().unwrap();
        let child_ids = children.iter().map(|child| child.childID).collect::<Vec<_>>();
        drop(children);

        // Get uct values
        let mut ucts = child_ids.iter().map(|childID| {
            let child = arena.nodes[*childID];
            child.uct(c, num_vists as f32)
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
        let num_visits = self.num_visits.load(Ordering::Acquire);
        if num_visits == 0 {
            return f32::INFINITY;
        }
        let sum_returns = self.sum_returns.load(Ordering::Acquire);

        sum_returns / num_visits as f32 + c * (parent_visits.ln() / num_visits as f32).sqrt()
    }

    /// Value function
    #[inline]
    pub fn get_V(&self) -> f32 {
        let num_visits = self.num_visits.load(Ordering::Acquire);
        if num_visits == 0 {
            return 0.0;
        }
        let sum_returns = self.sum_returns.load(Ordering::Acquire);
        sum_returns / num_visits as f32
    }

    /// Is terminal node
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.is_terminal.load(Ordering::Acquire)
    }

    /// Get action from info legal act, which was not already used
    #[inline]
    pub fn get_rand_action(&self) -> Action {
        let mut children = self.children.write().unwrap();
        let child_idx = self.child_idx.fetch_add(1, Ordering::AcqRel);
        let action = children[child_idx].action;
        drop(children);
        action
    }
}

/// Reference for nodes within arena, just an index
pub type NodeID = usize;

/// Representation of child of node.
#[derive(Clone, Debug)]
pub struct NodeChild {
    pub action: Action,
    pub childID: NodeID,
}

impl Default for NodeChild {
    fn default() -> NodeChild {
        NodeChild {
            action: 0,
            childID: 0,
        }
    }
}

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
}

/// Arena for managing nodes in a tree. Used for MCTS and its variants, so no delete operation. Specialized for sharing between threads
pub struct Arena {
    nodes: Vec<Node>,
    idx: AtomicUsize,
    capacity: usize,
}

impl Arena {
    /// Creates a new empty `Arena`.
    pub fn new(capacity: usize, action_capacity: usize) -> Arena {
        Arena {
            nodes: (0..capacity).into_iter().map(|_| Node::with_capacity(action_capacity)).collect(),
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
    pub fn add_node(&self, state: State) -> Result<NodeID, &'static str> {
        let curr_idx = self.idx.fetch_add(1, Ordering::AcqRel);

        if curr_idx < self.capacity {
            let new_node = Some(Node::new(state));
            let mut node_state = self.nodes[curr_idx].state.write().unwrap();
            *node_state = Some(state);

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

