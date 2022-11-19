use crate::gym::{Action, Collector, Episode, Obs, Obss, Infos, Worker, WorkerMessageIn, WorkerMessageOut, CollectorMessageIn, CollectorMessageOut};
use numpy::ndarray::{Array, Ix1, stack, Axis};
use crossbeam::channel::{unbounded, bounded, Sender, Receiver};
use itertools::izip;

/// The Environment
#[derive()]
pub struct Envs {
    num_envs: usize,
    num_envs_per_worker: usize,
    num_pending_request: usize,
    workers: Vec<Worker>,
    workers_ins: Vec<Sender<WorkerMessageIn>>,
    workers_out: Receiver<WorkerMessageOut>,
    collector: Collector,
    collector_in: Sender<CollectorMessageIn>,
    collector_out: Receiver<CollectorMessageOut>,
    eps_out: Receiver<Episode>,
}

impl Envs {
    pub fn new(num_workers: usize, num_envs_per_worker: usize, mut core_pinning: bool, gamma: f32, max_len: usize, size: u8) -> Envs {
        let num_envs = num_workers * num_envs_per_worker;
        let mut workers = Vec::with_capacity(num_workers);
        let mut workers_ins = Vec::with_capacity(num_workers);
        let (master_sender, workers_out) = bounded(num_envs*4);
        let (eps_in, eps_out) = unbounded();
        let core_ids = core_affinity::get_core_ids().unwrap();
        let num_cores = core_ids.len();
        if num_workers > num_cores {
            println!("Warning: More workers: {} than cpu cores: {}. Deactivating thread to core pinning", num_workers, num_cores);
            core_pinning = false;
        }

        for id in 0..num_workers {
            // Channel
            let (s, r) = bounded(num_envs_per_worker*4);
            workers_ins.push(s);

            // Core
            let core_id = if core_pinning {
                Some(core_ids[id])
            } else {
                None
            };

            workers.push(Worker::new(id, num_envs_per_worker, r, master_sender.clone(), eps_in.clone(), gamma, max_len, core_id, size));
        }

        // Collector
        let (collector_in, master_out) = bounded(num_envs*4);
        let (master_in, collector_out) = bounded(num_envs);
        let collector = Collector::new(num_envs, max_len, master_out, master_in);

        Envs {
            num_envs,
            num_envs_per_worker,
            num_pending_request: 0,
            workers,
            workers_ins,
            workers_out,
            collector,
            collector_in,
            collector_out,
            eps_out,
        }
    }

    /// Reset env
    pub fn reset(&mut self) -> (Obss, Infos) {
        // Send
        for (cid, c) in self.workers_ins.iter().enumerate() {
            for local_eid in 0..self.num_envs_per_worker {
                let eid = local_eid + cid * self.num_envs_per_worker;
                if c.try_send(WorkerMessageIn::Reset{eid: eid}).is_err() {
                    panic!("Error sending message RESET to worker");
                } else {
                    self.num_pending_request += 1;
                }
            }
        }

        // Receive (All)
        let mut msgs = Vec::with_capacity(self.num_envs);
        for _ in 0..self.num_envs {
            match self.workers_out.recv() {
                Ok(msg) => {
                    msgs.push(msg);
                },
                _ => panic!("Error receiving message RESET from worker"),
            }
        }
        let num_msgs = msgs.len();
        self.num_pending_request -= num_msgs;
        msgs.sort_by_key(|m| m.eid);

        // Process
        let mut obs = Vec::with_capacity(num_msgs);
        let mut pid = Vec::with_capacity(num_msgs);
        let mut eid = Vec::with_capacity(num_msgs);
        let mut legal_act = Vec::with_capacity(num_msgs);
        for msg in msgs.into_iter() {
            obs.push(msg.obs);
            pid.push(msg.info.pid);
            eid.push(msg.eid);
            legal_act.push(msg.info.legal_act);
        }

        let obs = stack(Axis(0), &obs.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();

        // Create info
        let pid = Array::from_vec(pid);
        let eid = Array::from_vec(eid);
        let legal_act = stack(Axis(0), &legal_act.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
        let info = Infos{pid, eid, legal_act};

        (obs, info)
    }

    /// Execute next action
    pub fn step(&mut self, act: Vec<Action>, eid: Vec<usize>, pol_id: Vec<usize>, num_wait: usize) -> (Obss, Array<f32, Ix1>, Array<bool, Ix1>, Infos)  {
        // Send
        for (a, e, p) in izip!(&act, &eid, &pol_id) {
            let cid = e / self.num_envs_per_worker;
            let c = &self.workers_ins[cid];
            if c.try_send(WorkerMessageIn::Step{act: *a, eid: *e, pol_id: *p}).is_err() {
                panic!("Error sending message STEP to worker");
            } else {
                self.num_pending_request += 1;
            }
        }

        // Receive
        let num_wait = num_wait.min(self.num_pending_request);
        while self.workers_out.len() < num_wait { }
        let mut msgs: Vec<WorkerMessageOut> = self.workers_out.try_iter().collect();
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
        for (cid, c) in self.workers_ins.iter().enumerate() {
            for local_eid in 0..self.num_envs_per_worker {
                let eid = local_eid + cid * self.num_envs_per_worker;
                if c.try_send(WorkerMessageIn::Render{eid: eid}).is_err() {
                    panic!("Error sending message RENDER to worker");
                }
            }
        }
    }

    /// Get legal actions
    pub fn legal_actions(&self) {
    }

    /// Close environment
    pub fn close(&mut self) {
        for c in self.workers_ins.iter() {
            if c.try_send(WorkerMessageIn::Shutdown).is_err() {
                panic!("Error sending message SHUTDOWN to worker");
            }
        }
        for worker in self.workers.iter_mut() {
            worker.close();
        }
    }

    /// Get episodes
    pub fn get_episodes(&self) -> Vec<Episode> {
        self.eps_out.try_iter().collect()
    }
}
