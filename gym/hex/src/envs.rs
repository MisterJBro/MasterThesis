use crate::{Coords};
use crate::{Env, Game, Status};
use serde::{Deserialize, Serialize};
use crate::board::{Board};
use std::thread;
use std::sync::Arc;
use crossbeam::queue::ArrayQueue;
use std::sync::Mutex;
use numpy::ndarray::{Array, Ix3, Ix4, stack_new_axis, stack, Axis};
use crossbeam::channel::{bounded, TrySendError, Sender, Receiver};

// Basic types
type Action = u16;
type Obs = Array<f32, Ix3>;

/// The Environment
#[derive()]
pub struct Envs {
    num_envs: usize,
    workers: Vec<Worker>,
    workers_channels: Vec<Sender<MasterMessage>>,
    master_channel: Receiver<WorkerMessage>,
}

impl Envs {
    pub fn new(num_workers: usize, num_envs_per_worker: usize, size: u8) -> Envs {
        let num_envs = num_workers * num_envs_per_worker;
        let mut workers = Vec::with_capacity(num_workers);
        let mut workers_channels = Vec::with_capacity(num_workers);
        let (master_sender, master_channel) = bounded(num_envs*2);

        for id in 0..num_workers {
            let (s, r) = bounded(num_envs_per_worker*2);
            workers_channels.push(s);
            workers.push(Worker::new(id, num_envs_per_worker, r, master_sender.clone(), size));
        }

        Envs {
            num_envs,
            workers,
            workers_channels,
            master_channel,
        }
    }

    /// Reset env
    pub fn reset(&self) -> (Array<f32, Ix4>, Vec<Vec<Action>>) {
        // Send
        for c in self.workers_channels.iter() {
            if c.try_send(MasterMessage::Reset).is_err() {
                panic!("Error sending message to worker");
            }
        }

        // Receive
        let mut msgs = Vec::with_capacity(self.num_envs);
        for _ in 0..self.num_envs {
            match self.master_channel.recv() {
                Ok(msg) => {
                    msgs.push(msg);
                },
                _ => panic!("Error receiving message from worker"),
            }
        }
        msgs.sort_by_key(|m| m.eid);

        // Process
        let mut obs = Vec::with_capacity(self.num_envs);
        let mut legal_acts = Vec::with_capacity(self.num_envs);
        for msg in msgs.into_iter() {
            obs.push(msg.obs);
            legal_acts.push(msg.legal_act);
        }
        let obs = stack(Axis(0), &obs.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
        println!("Obs: {:?}", obs.shape());
        (obs, legal_acts)
    }

    /// Execute next action
    pub fn step(&mut self, action: Action) {
    }

    /// Board representation
    pub fn to_string(&self) {
    }

    /// Render
    pub fn render(&self) {
    }

    // Get legal actions
    pub fn legal_actions(&self) {
    }

    // Close environment
    pub fn close(&self) { }
}


#[derive(Debug)]
enum MasterMessage {
    Reset,
    Step{action: Action},
}

#[derive(Debug)]
struct WorkerMessage {
    obs: Obs,
    legal_act: Vec<Action>,
    rew: Option<f32>,
    done: Option<bool>,
    eid: usize,
}

struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
}

impl Worker {
    fn new(id: usize, num_envs_per_worker: usize, in_channel: Receiver<MasterMessage>, out_channel: Sender<WorkerMessage>, size: u8) -> Worker {
        let eid_start = id * num_envs_per_worker;

        let thread = thread::spawn(move || {
            let mut envs = vec![Env::new(size); num_envs_per_worker];

            loop {
                if let Ok(message) = in_channel.recv() {
                    match message {
                        MasterMessage::Reset => {
                            // Reset
                            let mut mgss = Vec::with_capacity(num_envs_per_worker);
                            for (eid, env) in envs.iter_mut().enumerate() {
                                let (obs, legal_act) = env.reset();
                                mgss.push(WorkerMessage{
                                    obs,
                                    legal_act,
                                    rew: None,
                                    done: None,
                                    eid: eid_start+eid,
                                });
                            }

                            // Send
                            for msg in mgss.into_iter() {
                                if out_channel.try_send(msg).is_err() {
                                    panic!("Error sending message to master");
                                }
                            }
                            //let obss = stack(Axis(0), &obss.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
                            //println!("{:?}", obss);
                        },
                        MasterMessage::Step{action} => {
                            println!("Action: {}", action);
                            for env in envs.iter_mut() {
                                env.step(action);
                            }
                        },
                    }
                }
            }
        });

        Worker {
            id,
            thread,
        }
    }
}