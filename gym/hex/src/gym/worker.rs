
use std::thread;
use core_affinity::CoreId;
use crossbeam::channel::{Sender, Receiver};
use crate::gym::{Obs, Info, Env, EnvsMessage};


/// Message the worker sends to master
#[derive(Debug)]
pub struct WorkerMessage {
    pub obs: Obs,
    pub rew: Option<f32>,
    pub done: Option<bool>,
    pub info: Info,
    pub eid: usize,
}

/// Worker thread for Envs
pub struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    pub fn new(id: usize, num_envs_per_worker: usize, in_channel: Receiver<EnvsMessage>, out_channel: Sender<WorkerMessage>, core_id: Option<CoreId>, size: u8) -> Worker {
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
                        EnvsMessage::Reset{eid} => {
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
                        EnvsMessage::Step{eid, act} => {
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
                        EnvsMessage::Render{eid} => {
                            // Render
                            let local_eid = eid - eid_start;
                            println!("Render Env ID: {}\n{}", eid, envs[local_eid].to_string());
                        },
                        EnvsMessage::Close{eid} => {
                            // Close
                            let local_eid = eid - eid_start;
                            envs[local_eid].close();
                        },
                        EnvsMessage::Shutdown => {
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