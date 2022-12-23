use std::fmt::{Debug};
use crate::gym::{Env, Action, Obs, Info};


/// State description
#[derive(Clone, Debug)]
pub struct State {
    pub obs: Obs,
    pub rew: f32,
    pub done: bool,
    pub info: Info,
    pub env: Env,
}

impl State {
    pub fn new(obs: Obs, rew: f32, done: bool, info: Info, env: Env) -> State {
        State {
            obs,
            rew,
            done,
            info,
            env,
        }
    }

    /// Get number of legal actions
    #[inline]
    pub fn num_actions(&self) -> usize {
        self.info.legal_act.iter().filter(|&x| *x).count()
    }

    /// Transition to the next state with the given action
    #[inline]
    pub fn transition(&self, action: Action) -> State {
        let mut next_env = self.env.clone();
        let (obs, rew, done, info) = next_env.step(action);
        State::new(obs, rew, done, info, next_env)
    }

    /// Transition inplace
    #[inline]
    pub fn transition_inplace(&mut self, action: Action) {
        (self.obs, self.rew, self.done ,self.info) = self.env.step(action);
    }
}