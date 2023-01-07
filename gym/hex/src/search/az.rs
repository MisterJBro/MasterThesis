use std::thread;
use std::sync::{Arc};
use std::fmt::{Debug};
use crossbeam::channel::{bounded, Sender, Receiver};
use crate::search::{State, AZCore, SearchResult};


// Worker Messages
#[derive(Debug)]
pub enum AZWorkerMessageIn {
    Search {AZ: Arc<AZCore>, iters: usize},
}

#[derive(Clone, Debug)]
pub struct AZWorkerMessageOut {
    result: SearchResult,
}

/// Alpha Zero
pub struct AlphaZero {
    workers_ins: Vec<Sender<AZWorkerMessageIn>>,
    workers_out: Receiver<AZWorkerMessageOut>,
}

impl AlphaZero {
    pub fn new(num_threads: usize) -> AlphaZero {
        // Create Worker threads
        let mut workers = Vec::with_capacity(num_threads);
        let mut workers_ins = Vec::with_capacity(num_threads);
        let (master_sender, workers_out) = bounded(num_threads);
        for id in 0..num_threads {
            let (s, r) = bounded(1);
            workers_ins.push(s);
            workers.push(AZWorker::new(id, r, master_sender.clone()));
        }

        AlphaZero {
            workers_ins,
            workers_out,
        }
    }

    pub fn search(&self, state: State, iters: usize) -> SearchResult {
        let az = Arc::new(AZCore::new(state, iters+1));

        // Search
        for worker in &self.workers_ins {
            worker.send(AZWorkerMessageIn::Search{AZ: Arc::clone(&az), iters}).unwrap();
        }

        az.wait_for_result()
    }
}

/// Worker thread for Envs
pub struct AZWorker {
    thread: Option<thread::JoinHandle<()>>,
}

impl AZWorker {
    pub fn new(id: usize, in_channel: Receiver<AZWorkerMessageIn>, out_channel: Sender<AZWorkerMessageOut>) -> AZWorker {

        let thread = thread::spawn(move || {
            loop {
                if let Ok(message) = in_channel.recv() {
                    match message {
                        AZWorkerMessageIn::Search{AZ, iters} => {
                            AZ.search(iters);
                        }
                    }
                }
            }
        });

        AZWorker {
            thread: Some(thread),
        }
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.thread.take() {
            handle.join().expect("Could not join AZ worker thread!");
        }
    }
}
