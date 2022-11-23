
use std::thread;
use core_affinity::CoreId;
use crossbeam::channel::{Sender, Receiver};
use numpy::ndarray::{Array, Ix1};
use crate::gym::{Action, Obs, Info, Env, Episode};


// Worker Messages
#[derive(Debug)]
pub enum WorkerMessageIn {
    Reset{eid: usize},
    Step{act: Action, eid: usize, dist: Array<f32, Ix1>, pol_id: usize},
    Render{eid: usize},
    Close{eid: usize},
    Shutdown,
}
#[derive(Clone, Debug)]
pub struct WorkerMessageOut {
    pub obs: Obs,
    pub rew: Option<f32>,
    pub done: Option<bool>,
    pub info: Info,
    pub eid: usize,
}

/// Worker thread for Envs
pub struct Worker {
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    pub fn new(id: usize, num_envs_per_worker: usize, in_channel: Receiver<WorkerMessageIn>, out_channel: Sender<WorkerMessageOut>, eps_in: Sender<Episode>, gamma: f32, max_len: usize, core_id: Option<CoreId>, size: u8) -> Worker {
        let eid_start = id * num_envs_per_worker;
        let mut episodes = vec![Some(Episode::new(max_len)); num_envs_per_worker];

        let thread = thread::spawn(move || {
            // Set core affinity
            if let Some(core_id) = core_id {
                core_affinity::set_for_current(core_id);
            }

            let mut envs = vec![Env::new(size); num_envs_per_worker];

            loop {
                if let Ok(message) = in_channel.recv() {
                    match message {
                        WorkerMessageIn::Reset{eid} => {
                            // Reset
                            let local_eid = eid - eid_start;
                            let (obs, info) = envs[local_eid].reset();
                            let msg = WorkerMessageOut{
                                obs: obs.clone(),
                                rew: None,
                                done: None,
                                info: info.clone(),
                                eid,
                            };

                            // Send
                            if out_channel.try_send(msg).is_err() {
                                panic!("Error sending message to master");
                            }

                            // Collect
                            let episode = episodes[local_eid].as_mut().expect("Reset episode is None, but should always be some");
                            episode.clear();
                            episode.obs.push(obs);
                            episode.pid.push(info.pid);
                            episode.legal_act.push(info.legal_act);
                        },
                        WorkerMessageIn::Step{act, eid, dist, pol_id} => {
                            // Step
                            let local_eid = eid - eid_start;
                            let (obs, rew, done, info) = envs[local_eid].step(act);

                            if done {
                                let (next_obs, next_info) = envs[local_eid].reset();

                                // Send
                                if out_channel.try_send(WorkerMessageOut{
                                    obs: next_obs.clone(),
                                    rew: Some(rew.clone()),
                                    done: Some(done.clone()),
                                    info: next_info.clone(),
                                    eid,
                                }).is_err() {
                                    panic!("Error sending message to master");
                                }

                                // Take episode, send to process and create new one
                                let mut episode = episodes[local_eid].take().expect("Step episode is None, but should always be some");
                                episode.act.push(act);
                                episode.dist.push(dist);
                                episode.rew.push(rew);
                                episode.done.push(done);
                                episode.pol_id.push(pol_id);

                                episode.process(gamma);
                                if eps_in.try_send(episode).is_err() {
                                    panic!("Error sending episode to master");
                                }

                                let mut new_episode = Episode::new(max_len);
                                new_episode.obs.push(next_obs);
                                new_episode.pid.push(next_info.pid);
                                new_episode.legal_act.push(next_info.legal_act);
                                episodes[local_eid] = Some(new_episode);
                            } else {
                                // Send
                                if out_channel.try_send(WorkerMessageOut{
                                    obs: obs.clone(),
                                    rew: Some(rew.clone()),
                                    done: Some(done.clone()),
                                    info: info.clone(),
                                    eid,
                                }).is_err() {
                                    panic!("Error sending message to master");
                                }

                                let episode = episodes[local_eid].as_mut().expect("Step episode is None, but should always be some");
                                episode.obs.push(obs);
                                episode.act.push(act);
                                episode.dist.push(dist);
                                episode.rew.push(rew);
                                episode.done.push(done);
                                episode.pol_id.push(pol_id);
                                episode.pid.push(info.pid);
                                episode.legal_act.push(info.legal_act);
                            }
                        },
                        WorkerMessageIn::Render{eid} => {
                            // Render
                            let local_eid = eid - eid_start;
                            println!("Render Env ID: {}\n{}", eid, envs[local_eid].to_string());
                        },
                        WorkerMessageIn::Close{eid} => {
                            // Close
                            let local_eid = eid - eid_start;
                            envs[local_eid].close();
                        },
                        WorkerMessageIn::Shutdown => {
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
            thread: Some(thread),
        }
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.thread.take() {
            handle.join().expect("Could not join env worker thread!");
        }
    }
}