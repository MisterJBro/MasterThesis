use crate::gym::{Action, Obs};
use numpy::ndarray::{Array, Ix1};

/// Representation of a episode
#[derive(Clone)]
pub struct Episode {
    pub obs: Vec<Obs>,
    pub act: Vec<Action>,
    pub rew: Vec<f32>,
    pub done: Vec<bool>,
    pub pid: Vec<u8>,
    pub legal_act: Vec<Array<bool, Ix1>>,
}

impl Episode {
    pub fn new(max_len: usize) -> Episode {
        Episode{
            obs: Vec::with_capacity(max_len),
            act: Vec::with_capacity(max_len),
            rew: Vec::with_capacity(max_len),
            done: Vec::with_capacity(max_len),
            pid: Vec::with_capacity(max_len),
            legal_act: Vec::with_capacity(max_len),
        }
    }

    #[inline]
    pub fn push(&mut self, obs: Obs, act: Action, rew: f32, done: bool, pid: u8, legal_act: Array<bool, Ix1>) {
        self.obs.push(obs);
        self.act.push(act);
        self.rew.push(rew);
        self.done.push(done);
        self.pid.push(pid);
        self.legal_act.push(legal_act);
    }

    /// Clear trajectory
    #[inline]
    pub fn clear(&mut self) {
        self.obs.clear();
        self.act.clear();
        self.rew.clear();
        self.done.clear();
        self.pid.clear();
        self.legal_act.clear();
    }
}
