use crate::{Env,};
use std::thread;
use numpy::ndarray::{Array, Ix1, Ix3, Ix4, stack, Axis};
use crossbeam::channel::{bounded, Sender, Receiver};

// Basic types
type Action = u16;
type Obs = Array<f32, Ix3>;

/// The Environment
#[derive()]
pub struct Envs {
    num_envs: usize,
    num_envs_per_worker: usize,
    workers: Vec<Worker>,
    workers_channels: Vec<Sender<MasterMessage>>,
    master_channel: Receiver<WorkerMessage>,
}

impl Envs {
    pub fn new(num_workers: usize, num_envs_per_worker: usize, size: u8) -> Envs {
        let num_envs = num_workers * num_envs_per_worker;
        let mut workers = Vec::with_capacity(num_workers);
        let mut workers_channels = Vec::with_capacity(num_workers);
        let (master_sender, master_channel) = bounded(num_envs*4);

        for id in 0..num_workers {
            let (s, r) = bounded(num_envs_per_worker*4);
            workers_channels.push(s);
            workers.push(Worker::new(id, num_envs_per_worker, r, master_sender.clone(), size));
        }

        Envs {
            num_envs,
            num_envs_per_worker,
            workers,
            workers_channels,
            master_channel,
        }
    }

    /// Reset env
    pub fn reset(&self) -> (Array<f32, Ix4>, Vec<Vec<Action>>) {
        // Send
        for (cid, c) in self.workers_channels.iter().enumerate() {
            for local_eid in 0..self.num_envs_per_worker {
                let eid = local_eid + cid * self.num_envs_per_worker;
                if c.try_send(MasterMessage::Reset{eid: eid}).is_err() {
                    panic!("Error sending message RESET to worker");
                }
            }
        }

        // Receive
        let mut msgs = Vec::with_capacity(self.num_envs);
        for _ in 0..self.num_envs {
            match self.master_channel.recv() {
                Ok(msg) => {
                    msgs.push(msg);
                },
                _ => panic!("Error receiving message RESET from worker"),
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
        (obs, legal_acts)
    }

    /// Execute next action
    pub fn step(&self, acts: Vec<Action>) -> (Array<f32, Ix4>, Array<f32, Ix1>, Array<bool, Ix1>, Vec<Vec<Action>>)  {
        // Send
        for (cid, c) in self.workers_channels.iter().enumerate() {
            for local_eid in 0..self.num_envs_per_worker {
                let eid = local_eid + cid * self.num_envs_per_worker;
                if c.try_send(MasterMessage::Step{eid: eid, act: acts[eid]}).is_err() {
                    panic!("Error sending message STEP to worker");
                }
            }
        }

        // Receive
        let mut msgs = Vec::with_capacity(self.num_envs);
        for _ in 0..self.num_envs {
            match self.master_channel.recv() {
                Ok(msg) => {
                    msgs.push(msg);
                },
                _ => panic!("Error receiving message STEP from worker"),
            }
        }
        msgs.sort_by_key(|m| m.eid);

        // Process also rew and done
        let mut obs = Vec::with_capacity(self.num_envs);
        let mut rews = Vec::with_capacity(self.num_envs);
        let mut dones = Vec::with_capacity(self.num_envs);
        let mut legal_acts = Vec::with_capacity(self.num_envs);
        for msg in msgs.into_iter() {
            obs.push(msg.obs);
            rews.push(msg.rew.unwrap());
            dones.push(msg.done.unwrap());
            legal_acts.push(msg.legal_act);
        }
        // rews and dones to ndarrays
        let rews = Array::from_shape_vec((self.num_envs,), rews).unwrap();
        let dones = Array::from_shape_vec((self.num_envs,), dones).unwrap();
        let obs = stack(Axis(0), &obs.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
        (obs, rews, dones, legal_acts)
    }

    /// Render
    pub fn render(&self) {
        for (cid, c) in self.workers_channels.iter().enumerate() {
            for local_eid in 0..self.num_envs_per_worker {
                let eid = local_eid + cid * self.num_envs_per_worker;
                if c.try_send(MasterMessage::Render{eid: eid}).is_err() {
                    panic!("Error sending message RENDER to worker");
                }
            }
        }
    }

    // Get legal actions
    pub fn legal_actions(&self) {
    }

    // Close environment
    pub fn close(&self) {
        for (cid, c) in self.workers_channels.iter().enumerate() {
            for local_eid in 0..self.num_envs_per_worker {
                let eid = local_eid + cid * self.num_envs_per_worker;
                if c.try_send(MasterMessage::Close{eid: eid}).is_err() {
                    panic!("Error sending message CLOSE to worker");
                }
            }
        }
        //for worker in self.workers.into_iter() {
        //    worker.close();
        //}
    }
}


#[derive(Debug)]
enum MasterMessage {
    Reset{eid: usize},
    Step{eid: usize, act: Action},
    Render{eid: usize},
    Close{eid: usize},
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
                        MasterMessage::Reset{eid} => {
                            // Reset
                            let local_eid = eid - eid_start;
                            let (obs, legal_act) = envs[local_eid].reset();
                            let msg = WorkerMessage{
                                obs,
                                legal_act,
                                rew: None,
                                done: None,
                                eid,
                            };

                            // Send
                            if out_channel.try_send(msg).is_err() {
                                panic!("Error sending message to master");
                            }
                        },
                        MasterMessage::Step{eid, act} => {
                            // Step
                            let local_eid = eid - eid_start;
                            let (mut obs, rew, done, mut legal_act) = envs[local_eid].step(act);

                            if done {
                                let result = envs[local_eid].reset();
                                obs = result.0;
                                legal_act = result.1;
                            }
                            let msg = WorkerMessage{
                                obs,
                                legal_act,
                                rew: Some(rew),
                                done: Some(done),
                                eid,
                            };

                            // Send
                            if out_channel.try_send(msg).is_err() {
                                panic!("Error sending message to master");
                            }
                        },
                        MasterMessage::Render{eid} => {
                            // Render
                            let local_eid = eid - eid_start;
                            println!("Render Env ID: {}\n{}", eid, envs[local_eid].to_string());
                        },
                        MasterMessage::Close{eid} => {
                            // Close
                            let local_eid = eid - eid_start;
                            envs[local_eid].close();
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

    pub fn close(self) {
        self.thread.join().unwrap();
    }
}