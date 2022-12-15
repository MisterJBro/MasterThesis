use std::thread;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::fmt::{Debug, Formatter, Result};
use numpy::ndarray::{Array, Ix1};
use hexgame::gym::{Env, Envs, Action, Obs, Info, Infos, Episode};
use crossbeam::channel::{unbounded, bounded, Sender, Receiver};

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

#[derive(Clone, Debug)]
pub struct State {
    pub obs: Obs,
    pub info: Info,
    pub env: Env,
}

pub struct SearchResult {
    Q: Array<f32, Ix1>,
    V: f32,
    pi: Array<f32, Ix1>,
}

#[derive(Debug)]
pub struct MCTSCore {
    expl_coeff: f32,
    arena: Arena<State>,
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

    pub fn select(&self) -> NodeID {
        let mut node = 0;
        while !node.is_leaf() {
            node = node.select_child(self.expl_coeff);
        }
        node
    }

    pub fn expand(&self, leaf: NodeID) -> NodeID {
        0
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
#[derive(Clone, Debug)]
pub struct Node<T> where T : Clone + Debug {
    pub data: T,
    pub children: Vec<usize>,
}

pub type NodeID = usize;

/// Arena for managing nodes in a tree. Used for MCTS and its variants, so no delete operation. Specialized for sharing between threads
pub struct Arena<T> where T : Clone + Debug {
    nodes: RwLock<Vec<RwLock<Option<Node<T>>>>>,
    idx: AtomicUsize,
    capacity: AtomicUsize,
}

impl<T> Arena<T> where T : Clone + Debug {
    /// Creates a new empty `Arena`.
    pub fn new(capacity: usize) -> Arena<T> {
        Arena {
            nodes: RwLock::new((0..capacity).into_iter().map(|_| RwLock::new(None)).collect()),
            idx: AtomicUsize::new(0),
            capacity : AtomicUsize::new(capacity),
        }
    }

    /// Checks if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.idx.load(Ordering::Acquire) == 0
    }

    /// Add a new node to the arena
    #[inline]
    pub fn add_node(&self, data: T) -> NodeID {
        let curr_idx = self.idx.fetch_add(1, Ordering::AcqRel);

        if curr_idx < self.capacity.load(Ordering::Acquire) {
            let nodes = self.nodes.read().unwrap();

            let mut new_node = nodes[curr_idx].write().unwrap();
            *new_node = Some(Node {
               data,
               children: vec![],
            });
        }
        else {
            // Add more capacity
            let mut nodes = self.nodes.write().unwrap();
            let capacity = self.capacity.load(Ordering::Acquire);
            nodes.append(&mut (0..capacity).into_iter().map(|_| RwLock::new(None)).collect());
            self.capacity.fetch_add(capacity, Ordering::AcqRel);

            let mut new_node = nodes[curr_idx].write().unwrap();
            *new_node = Some(Node {
               data,
               children: vec![],
            });
        }

        return curr_idx;
  }

    /// Add child node to parent
    #[inline]
    pub fn add_child(&self, parent: NodeID, child: NodeID) {
        let nodes = self.nodes.read().unwrap();
        let mut parent_node = nodes[parent].write().unwrap();
        parent_node.as_mut().unwrap().children.push(child);
    }

    /// Get node
    #[inline]
    pub fn get_node(&self, id: NodeID) -> Option<Node<T>> {
        let nodes = self.nodes.read().unwrap();
        let node = nodes[id].read().unwrap();
        return node.clone();
    }

}

impl<T> Debug for Arena<T> where T : Clone + Debug {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Arena {{ nodes: {:?} }}", self.nodes)
    }
}


fn main () {
    // Env
    let mut env = Env::new(5);
    let (mut obs, mut info) = env.reset();

    // MCTS
    let mcts = MCTS::new(1);
    let state = State { obs, info, env };
    mcts.search(state, 1);

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