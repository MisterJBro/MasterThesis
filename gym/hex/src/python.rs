use crate::{Env};
use numpy::ToPyArray;
use numpy::{PyArray3};
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
    pub fn new(size: u8) -> PyEnv {
        PyEnv(Env::new(size))
    }

    /// Fields
    #[getter]
    fn size(&self) -> u8 {
        self.0.size
    }

    /// Reset the environment
    fn reset<'py>(&mut self, py: Python<'py>) -> (&'py PyArray3<f32>, Vec<u16>) {
        let (obs, legal_act) = self.0.reset();
        (obs.to_pyarray(py), legal_act)
    }

    /// Execute the current actions and update the board
    fn step<'py>(&mut self, py: Python<'py>, action: u16) -> (&'py PyArray3<f32>, f32, bool, Vec<u16>) {
        let (obs, rew, done, legal_act) = self.0.step(action);
        (obs.to_pyarray(py), rew, done, legal_act)
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
    fn legal_actions(&self) -> Vec<u16> {
        self.0.legal_actions()
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
