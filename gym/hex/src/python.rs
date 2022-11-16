use crate::gym::{Env, Envs, Action, Obs, Info, Infos};
use numpy::ToPyArray;
use numpy::{PyArray1, PyArray3, PyArray4};
use pyo3::prelude::*;
use pyo3::types::{PyDict};
use serde::{Deserialize, Serialize};

/// Env Interface for use in Python
#[pyclass(name = "RustEnv")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyEnv(pub Env);

#[pymethods]
impl PyEnv {
    #[new]
    #[args(size="9")]
    pub fn new(size: u8) -> PyEnv {
        PyEnv(Env::new(size))
    }

    /// Fields
    #[getter]
    fn size(&self) -> u8 {
        self.0.size
    }

    /// Reset the environment
    fn reset<'py>(&mut self, py: Python<'py>) -> (&'py PyArray3<f32>, Info) {
        let (obs, info) = self.0.reset();
        (obs.to_pyarray(py), info)
    }

    /// Execute the current actions and update the board
    fn step<'py>(&mut self, py: Python<'py>, act: u16) -> (&'py PyArray3<f32>, f32, bool, Info) {
        let (obs, rew, done, info) = self.0.step(act);
        (obs.to_pyarray(py), rew, done, info)
    }

    /// Render env
    fn render(&self) {
        self.0.render();
    }

    /// Close env
    fn close(&self) {
        self.0.close();
    }

    // Get all legal actions
    fn legal_actions<'py>(&self, py: Python<'py>) -> &'py PyArray1<bool> {
        self.0.legal_actions().to_pyarray(py)
    }


    /// Python built-in functions
    fn __repr__(&self) -> String {self.0.to_string()}
    fn copy(&self) -> Self {self.clone()}
    fn __copy__(&self) -> Self {self.clone()}
    fn __deepcopy__(&self, _memo: &PyDict) -> Self {self.clone()}

    fn to_pickle(&self) -> Vec<u8> {
        let serialized = serde_pickle::to_vec(&self, Default::default()).unwrap();
        return serialized
    }

    fn from_pickle(&mut self, serialized: Vec<u8>) {
        let deserialized: Self = serde_pickle::from_slice(&serialized, Default::default()).unwrap();
        self.0.game = deserialized.0.game;
    }
}

/// (Multiple) Envs Interface for use in Python
#[pyclass(name = "RustEnvs")]
#[derive()]
pub struct PyEnvs(pub Envs);

#[pymethods]
impl PyEnvs {
    #[new]
    #[args(num_workers="2", num_envs_per_worker = "2", core_pinning="false", size="9")]
    pub fn new(num_workers: usize, num_envs_per_worker: usize, core_pinning: bool, size: u8) -> PyEnvs {
        PyEnvs(Envs::new(num_workers, num_envs_per_worker, core_pinning, size))
    }

    /// Reset the environment
    fn reset<'py>(&mut self, py: Python<'py>) -> (&'py PyArray4<f32>, Infos) {
        let (obs, info) = self.0.reset();
        (obs.to_pyarray(py), info)
    }

    /// Execute the current actions and update the board
    #[args(num_waits="1")]
    fn step<'py>(&mut self, py: Python<'py>, act: Vec<(usize, u16)>, num_waits: usize) -> (&'py PyArray4<f32>, &'py PyArray1<f32>, &'py PyArray1<bool>, Infos) {
        let (obs, rew, done, info) = self.0.step(act, num_waits);
        (obs.to_pyarray(py), rew.to_pyarray(py), done.to_pyarray(py), info)
    }

    /// Render env
    fn render(&self) {
        self.0.render();
    }

    /// Close env
    fn close(&mut self) {
        self.0.close();
    }
}
