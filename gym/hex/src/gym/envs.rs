use crate::gym::{Action, Collector, Obss, Infos, Worker, WorkerMessageIn, WorkerMessageOut, CollectorMessageIn, CollectorMessageOut};
use numpy::ndarray::{Array, Ix1, stack, Axis};
use crossbeam::channel::{unbounded, bounded, Sender, Receiver};

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
}

impl Envs {
    pub fn new(num_workers: usize, num_envs_per_worker: usize, mut core_pinning: bool, size: u8) -> Envs {
        let num_envs = num_workers * num_envs_per_worker;
        let mut workers = Vec::with_capacity(num_workers);
        let mut workers_ins = Vec::with_capacity(num_workers);
        let (master_sender, workers_out) = bounded(num_envs*4);
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

            workers.push(Worker::new(id, num_envs_per_worker, r, master_sender.clone(), core_id, size));
        }

        // Collector
        let (collector_in, master_out) = bounded(num_envs*16);
        let (master_in, collector_out) = bounded(num_envs);
        let collector = Collector::new(master_out, master_in);

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
                }
            }
        }
        if self.collector_in.try_send(CollectorMessageIn::Clear).is_err() {
            panic!("Error sending message CLEAR to collector");
        }

        // Receive
        let mut msgs = Vec::with_capacity(self.num_envs);
        for _ in 0..self.num_envs {
            match self.workers_out.recv() {
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
            // Collect
            if self.collector_in.try_send(CollectorMessageIn::Add { sample: msg.clone() }).is_err() {
                panic!("Error sending message ADD to collector");
            }

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
    pub fn step(&mut self, acts: Vec<(usize, Action)>, num_wait: usize) -> (Obss, Array<f32, Ix1>, Array<bool, Ix1>, Infos)  {
        // Send
        for (eid, act) in acts {
            let cid = eid / self.num_envs_per_worker;
            let c = &self.workers_ins[cid];
            if c.try_send(WorkerMessageIn::Step{eid: eid, act: act}).is_err() {
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
            // Collect
            if self.collector_in.try_send(CollectorMessageIn::Add { sample: msg.clone() }).is_err() {
                panic!("Error sending message ADD to collector");
            }

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
    pub fn get_episodes(&mut self) -> Vec<CollectorMessageOut> {
        self.collector_out.try_iter().collect()
    }
}
