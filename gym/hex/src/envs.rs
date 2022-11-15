use crate::{Env, Info};
use std::thread;
use core_affinity::CoreId;
use numpy::ndarray::{Array, Ix1, Ix2, Ix3, Ix4, stack, Axis};
use numpy::{IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, IntoPyDict};
use crossbeam::channel::{bounded, Sender, Receiver};

// Basic types
type Action = u16;
type Obs = Array<f32, Ix3>;

#[derive(Debug)]
pub struct Infos {
    pub pid: Array<u8, Ix1>,
    pub eid: Array<usize, Ix1>,
    pub legal_act: Array<bool, Ix2>,
}
impl IntoPy<PyObject> for Infos {
    fn into_py(self, py: Python) -> PyObject {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("pid", self.pid.into_pyarray(py).to_object(py)),
            ("eid", self.eid.into_pyarray(py).to_object(py)),
            ("legal_act", self.legal_act.into_pyarray(py).to_object(py)),
        ];
        let dict = key_vals.into_py_dict(py);
        dict.to_object(py)
    }
}

/// The Environment
#[derive()]
pub struct Envs {
    num_envs: usize,
    num_envs_per_worker: usize,
    num_pending_request: usize,
    workers: Vec<Worker>,
    workers_channels: Vec<Sender<MasterMessage>>,
    master_channel: Receiver<WorkerMessage>,
}

impl Envs {
    pub fn new(num_workers: usize, num_envs_per_worker: usize, mut core_pinning: bool, size: u8) -> Envs {
        let num_envs = num_workers * num_envs_per_worker;
        let mut workers = Vec::with_capacity(num_workers);
        let mut workers_channels = Vec::with_capacity(num_workers);
        let (master_sender, master_channel) = bounded(num_envs*4);
        let core_ids = core_affinity::get_core_ids().unwrap();
        let num_cores = core_ids.len();
        if num_workers > num_cores {
            println!("Warning: More workers: {} than cpu cores: {}. Deactivating thread to core pinning", num_workers, num_cores);
            core_pinning = false;
        }

        for id in 0..num_workers {
            // Channel
            let (s, r) = bounded(num_envs_per_worker*4);
            workers_channels.push(s);

            // Core
            let core_id = if core_pinning {
                Some(core_ids[id])
            } else {
                None
            };

            workers.push(Worker::new(id, num_envs_per_worker, r, master_sender.clone(), core_id, size));
        }

        Envs {
            num_envs,
            num_envs_per_worker,
            num_pending_request: 0,
            workers,
            workers_channels,
            master_channel,
        }
    }

    /// Reset env
    pub fn reset(&mut self) -> (Array<f32, Ix4>, Infos) {
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
        let mut pid = Vec::with_capacity(self.num_envs);
        let mut legal_act = Vec::with_capacity(self.num_envs);
        for msg in msgs.into_iter() {
            obs.push(msg.obs);
            pid.push(msg.info.pid);
            legal_act.push(msg.info.legal_act);
        }
        let obs = stack(Axis(0), &obs.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();

        // Create info
        let pid = Array::from_vec(pid);
        let eid = Array::from_iter(0..self.num_envs);
        let legal_act = stack(Axis(0), &legal_act.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
        let info = Infos{pid, eid, legal_act};

        (obs, info)
    }

    /// Execute next action
    pub fn step(&mut self, acts: Vec<(usize, Action)>, num_wait: usize) -> (Array<f32, Ix4>, Array<f32, Ix1>, Array<bool, Ix1>, Infos)  {
        // Send
        for (eid, act) in acts {
            let cid = eid / self.num_envs_per_worker;
            let c = &self.workers_channels[cid];
            if c.try_send(MasterMessage::Step{eid: eid, act: act}).is_err() {
                panic!("Error sending message STEP to worker");
            } else {
                self.num_pending_request += 1;
            }
        }

        // Receive
        let num_wait = num_wait.min(self.num_pending_request);
        while self.master_channel.len() < num_wait { }
        let mut msgs: Vec<WorkerMessage> = self.master_channel.try_iter().collect();
        let num_msgs = msgs.len();
        self.num_pending_request -= num_msgs;
        msgs.sort_by_key(|m| m.eid);

        // Process also rew and done
        let mut obs = Vec::with_capacity(num_msgs);
        let mut rews = Vec::with_capacity(num_msgs);
        let mut dones = Vec::with_capacity(num_msgs);
        let mut pid = Vec::with_capacity(num_msgs);
        let mut eid = Vec::with_capacity(num_msgs);
        let mut legal_act = Vec::with_capacity(num_msgs);

        for msg in msgs.into_iter() {
            obs.push(msg.obs);
            rews.push(msg.rew.unwrap());
            dones.push(msg.done.unwrap());
            pid.push(msg.info.pid);
            eid.push(msg.eid);
            legal_act.push(msg.info.legal_act);
        }
        // To ndarray
        let obs = stack(Axis(0), &obs.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
        let rews = Array::from_vec(rews);
        let dones = Array::from_vec(dones);

        // Create info
        let pid = Array::from_vec(pid);
        let eid = Array::from_vec(eid);
        let legal_act = stack(Axis(0), &legal_act.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
        let info = Infos{pid, eid, legal_act};
        (obs, rews, dones, info)
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
    pub fn close(&mut self) {
        for (cid, c) in self.workers_channels.iter().enumerate() {
            if c.try_send(MasterMessage::Shutdown).is_err() {
                panic!("Error sending message SHUTDOWN to worker");
            }
        }
        for worker in self.workers.iter_mut() {
            worker.close();
        }
    }
}


#[derive(Debug)]
enum MasterMessage {
    Reset{eid: usize},
    Step{eid: usize, act: Action},
    Render{eid: usize},
    Close{eid: usize},
    Shutdown,
}

#[derive(Debug)]
struct WorkerMessage {
    obs: Obs,
    rew: Option<f32>,
    done: Option<bool>,
    info: Info,
    eid: usize,
}

struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(id: usize, num_envs_per_worker: usize, in_channel: Receiver<MasterMessage>, out_channel: Sender<WorkerMessage>, core_id: Option<CoreId>, size: u8) -> Worker {
        let eid_start = id * num_envs_per_worker;

        let thread = thread::spawn(move || {
            // Set core affinity
            if let Some(core_id) = core_id {
                core_affinity::set_for_current(core_id);
            }

            let mut envs = vec![Env::new(size); num_envs_per_worker];

            loop {
                if let Ok(message) = in_channel.recv() {
                    match message {
                        MasterMessage::Reset{eid} => {
                            // Reset
                            let local_eid = eid - eid_start;
                            let (obs, info) = envs[local_eid].reset();
                            let msg = WorkerMessage{
                                obs,
                                rew: None,
                                done: None,
                                info,
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
                            let (mut obs, rew, done, mut info) = envs[local_eid].step(act);

                            if done {
                                let result = envs[local_eid].reset();
                                obs = result.0;
                                info = result.1;
                            }
                            let msg = WorkerMessage{
                                obs,
                                rew: Some(rew),
                                done: Some(done),
                                info,
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
                        MasterMessage::Shutdown => {
                            for env in envs.iter_mut() {
                                env.close();
                            }
                            break;
                        },
                    }
                }
            }
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.thread.take() {
            handle.join().expect("Could not join env worker thread!");
        }
    }
}