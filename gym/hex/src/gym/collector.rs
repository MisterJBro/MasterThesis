
use std::thread;
use crate::gym::{Action, Obs, Info, WorkerMessageOut};
use crossbeam::channel::{Sender, Receiver};
use numpy::ndarray::{Array, Ix1, stack, Axis};

#[derive(Clone)]
pub struct Trajectory {
    pub obs: Vec<Obs>,
    pub act: Vec<Action>,
    pub rew: Vec<f32>,
    pub done: Vec<bool>,
    pub pid: Vec<u8>,
    pub legal_act: Vec<Array<bool, Ix1>>,
}
impl Trajectory {
    pub fn new(max_len: usize) -> Trajectory {
        Trajectory{
            obs: Vec::with_capacity(max_len),
            act: Vec::with_capacity(max_len),
            rew: Vec::with_capacity(max_len),
            done: Vec::with_capacity(max_len),
            pid: Vec::with_capacity(max_len),
            legal_act: Vec::with_capacity(max_len),
        }
    }

    pub fn push(&mut self, obs: Obs, act: Action, rew: f32, done: bool, pid: u8, legal_act: Array<bool, Ix1>) {
        self.obs.push(obs);
        self.act.push(act);
        self.rew.push(rew);
        self.done.push(done);
        self.pid.push(pid);
        self.legal_act.push(legal_act);
    }
}

// Collector Messages
#[derive(Debug)]
pub enum CollectorMessageIn {
    AddAct{act: Vec<(usize, Action)>},
    AddMsg{msg: WorkerMessageOut},
    Clear,
    Get,
}
#[derive(Debug)]
pub struct CollectorMessageOut {
}

/// Sample Collector and processor
pub struct Collector {
    thread: Option<thread::JoinHandle<()>>,
}

impl Collector {
    pub fn new(num_envs: usize, max_len: usize, in_channel: Receiver<CollectorMessageIn>, out_channel: Sender<CollectorMessageOut>) -> Collector {
        let mut trajs = vec![Trajectory::new(max_len); num_envs];


        let thread = thread::spawn(move || {
            loop {
                if let Ok(message) = in_channel.recv() {
                    match message {
                        CollectorMessageIn::AddAct{act} => {
                            for (eid, act) in act {
                                trajs[eid].act.push(act);
                            }
                        },
                        CollectorMessageIn::AddMsg{msg} => {
                            let eid = msg.eid;
                            trajs[eid].obs.push(msg.obs);
                            if let Some(rew) = msg.rew {
                                trajs[eid].rew.push(rew);
                            }
                            if let Some(done) = msg.done {
                                trajs[eid].done.push(done);
                            }
                            trajs[eid].pid.push(msg.info.pid);
                            trajs[eid].legal_act.push(msg.info.legal_act);
                        },
                        CollectorMessageIn::Clear => {
                            trajs = vec![Trajectory::new(max_len); num_envs];
                        },
                        CollectorMessageIn::Get => {
                            // Get samples from buffer
                            let msg = CollectorMessageOut{};
                            let episodes = vec![1];
                            // Send
                            if out_channel.try_send(msg).is_err() {
                                panic!("Error sending message to master");
                            }
                        },
                    }
                }
            }
        });

        Collector {
            thread: Some(thread),
        }
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.thread.take() {
            handle.join().expect("Could not join collector thread!");
        }
    }
}

/// Process single episode of data
fn process_episode() {

}