use crate::gym::{Env, Envs, Action, Obs, Info, Infos, CollectorMessageOut, Episode};
use numpy::ToPyArray;
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray2};
use numpy::ndarray::{Array, Ix1, Ix2, Ix3, Ix4, stack, Axis};
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
    #[args(num_workers="2", num_envs_per_worker="2", core_pinning="false", gamma="1.0", max_len="81", size="9")]
    pub fn new(num_workers: usize, num_envs_per_worker: usize, core_pinning: bool, gamma: f32, max_len: usize, size: u8) -> PyEnvs {
        PyEnvs(Envs::new(num_workers, num_envs_per_worker, core_pinning, gamma, max_len, size))
    }

    /// Reset the environment
    fn reset<'py>(&mut self, py: Python<'py>) -> (&'py PyArray4<f32>, Infos) {
        let (obs, info) = self.0.reset();
        (obs.to_pyarray(py), info)
    }

    /// Execute the current actions and update the board   dist: Vec<Vec<f32>,
    #[args(num_waits="1")]
    fn step<'py>(&mut self, py: Python<'py>, act: Vec<Action>, eid: Vec<usize>, dist: PyReadonlyArray2<f32>, pol_id: Vec<usize>, num_waits: usize) -> (&'py PyArray4<f32>, &'py PyArray1<f32>, &'py PyArray1<bool>, Infos) {
        let dist = dist.as_array().into_owned();
        let (obs, rew, done, info) = self.0.step(act, eid, dist, pol_id, num_waits);
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

    /// Get episodes
    fn get_episodes(&self) -> Vec<PyEpisode> {
        let episodes = self.0.get_episodes();

        episodes.into_iter().map(|eps| eps.to_python()).collect()
    }

    fn get_envs(&self, eid: Vec<usize>) -> Vec<PyEnv> {
        let envs = self.0.get_envs(eid);
        envs.into_iter().map(|env| PyEnv(env)).collect()
    }
}

/// Python Episode Interface
#[pyclass(name = "RustEpisode")]
#[derive(Clone)]
pub struct PyEpisode {
    pub obs: Array<f32, Ix4>,
    pub act: Array<u16, Ix1>,
    pub rew: Array<f32, Ix1>,
    pub done: Array<bool, Ix1>,
    pub pid: Array<u8, Ix1>,
    pub legal_act: Array<bool, Ix2>,
    pub dist: Array<f32, Ix2>,
    pub pol_id: Array<usize, Ix1>,
    pub ret: Array<f32, Ix1>,
}

#[pymethods]
impl PyEpisode {
    #[getter]
    fn obs<'py>(&self, py: Python<'py>) -> &'py PyArray4<f32> {
        self.obs.to_pyarray(py)
    }

    #[getter]
    fn act<'py>(&self, py: Python<'py>) -> &'py PyArray1<u16> {
        self.act.to_pyarray(py)
    }

    #[getter]
    fn rew<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.rew.to_pyarray(py)
    }

    #[getter]
    fn done<'py>(&self, py: Python<'py>) -> &'py PyArray1<bool> {
        self.done.to_pyarray(py)
    }

    #[getter]
    fn pid<'py>(&self, py: Python<'py>) -> &'py PyArray1<u8> {
        self.pid.to_pyarray(py)
    }

    #[getter]
    fn legal_act<'py>(&self, py: Python<'py>) -> &'py PyArray2<bool> {
        self.legal_act.to_pyarray(py)
    }

    #[getter]
    fn dist<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        self.dist.to_pyarray(py)
    }

    #[getter]
    fn pol_id<'py>(&self, py: Python<'py>) -> &'py PyArray1<usize> {
        self.pol_id.to_pyarray(py)
    }

    #[getter]
    fn ret<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.ret.to_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.rew.shape()[0]
    }

    fn __repr__(&self) -> String {
        format!("RustEpisode({})", self.__len__())
    }
}
