use numpy::ndarray::{Array, Ix1, Ix2};
use numpy::{IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict};


/// Extra Information, for one env step
#[derive(Debug)]

pub struct Info {
    pub pid: u8,
    pub legal_act: Array<bool, Ix1>,
}
impl IntoPy<PyObject> for Info {
    fn into_py(self, py: Python) -> PyObject {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("pid", self.pid.to_object(py)),
            ("legal_act", self.legal_act.into_pyarray(py).to_object(py)),
        ];
        let dict = key_vals.into_py_dict(py);
        dict.to_object(py)
    }
}

/// Extra information for several env.step with each env id
#[derive(Debug)]
pub struct Infos {
    pub pid: Array<u8, Ix1>,
    pub eid: Array<usize, Ix1>,
    pub legal_act: Array<bool, Ix2>,
}
impl IntoPy<PyObject> for Infos {
    fn into_py(self, py: Python) -> PyObject {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("pid", self.pid.into_pyarray(py).to_object(py)),
            ("eid", self.eid.into_pyarray(py).to_object(py)),
            ("legal_act", self.legal_act.into_pyarray(py).to_object(py)),
        ];
        let dict = key_vals.into_py_dict(py);
        dict.to_object(py)
    }
}