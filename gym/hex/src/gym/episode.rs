use crate::{PyEpisode};
use crate::gym::{Action, Obs, Obss};
use numpy::ndarray::{Array, Ix1, Ix2, stack, Axis};

/// Representation of a episode
#[derive(Clone)]
pub struct Episode {
    pub obs: Vec<Obs>,
    pub act: Vec<Action>,
    pub rew: Vec<f32>,
    pub done: Vec<bool>,
    pub pid: Vec<u8>,
    pub legal_act: Vec<Array<bool, Ix1>>,
    pub ret: Option<Vec<f32>>
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
            ret: None,
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
        self.ret = None;
    }

    /// Process trajectory by calculating the return
    #[inline]
    pub fn process(&mut self, gamma: f32) {
        let mut ret = vec![0.0; self.rew.len()];
        let mut ret_sum = 0.0;
        for i in (0..self.rew.len()).rev() {
            ret_sum = self.rew[i] - gamma * ret_sum;
            ret[i] = ret_sum;
        }
        self.ret = Some(ret);
    }

    pub fn to_python(self) -> PyEpisode {
        let obs = stack(Axis(0), &self.obs.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
        let act = Array::from_vec(self.act);
        let rew = Array::from_vec(self.rew);
        let done = Array::from_vec(self.done);
        let pid = Array::from_vec(self.pid);
        let legal_act = stack(Axis(0), &self.legal_act.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap();
        let ret = Array::from_vec(self.ret.expect("Episode: Expected return to be already calculated before converting"));

        PyEpisode {
            obs,
            act,
            rew,
            done,
            pid,
            legal_act,
            ret,
        }
    }

    /// Create new episodes by data augmenting the current one e.g. rotating the board
    pub fn create_augmented_eps(&self) -> Vec<Episode> {
        /*let mut eps = Vec::new();
        let mut obs = self.obs.clone();
        let mut act = self.act.clone();
        let mut rew = self.rew.clone();
        let mut done = self.done.clone();
        let mut pid = self.pid.clone();
        let mut legal_act = self.legal_act.clone();
        let mut ret = self.ret.clone();

        // Rotate 90 degrees
        obs = obs.iter().map(|x| x.rot90()).collect();
        act = act.iter().map(|x| x.rot90()).collect();
        legal_act = legal_act.iter().map(|x| x.rot90()).collect();
        eps.push(Episode{obs, act, rew, done, pid, legal_act, ret});

        // Rotate 180 degrees
        obs = obs.iter().map(|x| x.rot90()).collect();
        act = act.iter().map(|x| x.rot90()).collect();
        legal_act = legal_act.iter().map(|x| x.rot90()).collect();
        eps.push(Episode{obs, act, rew, done, pid, legal_act, ret});

        // Rotate 270 degrees
        obs = obs.iter().map(|x| x.rot90()).collect();
        act = act.iter().map(|x| x.rot90()).collect();
        legal_act = legal_act.iter().map(|x| x.rot90()).collect();
        eps.push(Episode{obs, act, rew, done, pid, legal_act, ret});

        // Flip horizontally
        obs = obs.iter().map(|x| x.flip_horiz()).collect();
        act = act.iter().map(|x| x.flip_horiz()).collect();
        legal_act = legal_act.iter().map(|x| x.flip_horiz()).collect();
        eps.push(Episode{obs, act, rew, done, pid, legal_act, ret});

        // Flip vertically
        obs = obs.iter().map(|x| x.flip_vert()).collect();
        act = act.iter().map(|x| x.flip_vert()).collect();
        legal_act = legal_act.iter().map(|x| x.flip_vert()).collect();
        eps.push(Episode{obs, act, rew, done, pid, legal_act, ret});

        eps*/
        vec![]
    }
}
