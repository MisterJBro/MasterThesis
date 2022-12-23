use pyo3::types::{IntoPyDict};
use numpy::ndarray::{Array, Ix1};
use numpy::{IntoPyArray};
use pyo3::prelude::*;


/// Result of search algorithms
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub pi: Array<f32, Ix1>,
    pub q: Array<f32, Ix1>,
    pub v: f32,
}
impl IntoPy<PyObject> for SearchResult {
    fn into_py(self, py: Python) -> PyObject {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("pi", self.pi.into_pyarray(py).to_object(py)),
            ("q", self.q.into_pyarray(py).to_object(py)),
            ("v", self.v.to_object(py)),
        ];
        let dict = key_vals.into_py_dict(py);
        dict.to_object(py)
    }
}