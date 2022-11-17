use numpy::ndarray::{Array, Ix1, Ix2};
use numpy::{IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict};


/// Configuration
#[derive(Clone, Debug)]

pub struct Config {
    pub discount_factor: f32,
    pub num_envs: usize,
    pub max_len: usize,
    pub cpu_pinning : bool,
}
impl IntoPy<PyObject> for Config {
    fn into_py(self, py: Python) -> PyObject {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("discount_factor", self.discount_factor.into_py(py)),
            ("num_envs", self.num_envs.into_py(py)),
            ("max_len", self.max_len.into_py(py)),
            ("cpu_pinning", self.cpu_pinning.into_py(py)),
        ];
        let dict = key_vals.into_py_dict(py);
        dict.to_object(py)
    }
}