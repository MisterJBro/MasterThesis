use crate::{Coords};
use crate::{Game, Status};
use numpy::ToPyArray;
use numpy::{PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyBytes, PyTuple};
use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};

/// Game Interface for use in Python
#[pyclass(name = "HexGame")]
#[derive(Clone, Serialize, Deserialize)]
pub struct HexGame {
    pub size: u8,
    pub game: Game,
}

#[pymethods]
impl HexGame {
    #[new]
    pub fn new(size: u8) -> HexGame {
        HexGame {
            size,
            game: Game::new(size),
        }
    }

    /// Reset the environment
    fn reset<'py>(&mut self, py: Python<'py>) -> &'py PyArray3<f32> {
        self.game = Game::new(self.size);
        return self.get_obs(py)
    }

    /// Execute the current actions and update the board
    fn step<'py>(&mut self, py: Python<'py>, action: u16) -> (&'py PyArray3<f32>, f32, bool) {
        let mut done = false;
        let mut rew = 0f32;

        match self.game.get_status() {
            Status::Ongoing(_current_player) => {
                let coords = Coords::from_u16(action, self.size as u16);
                self.game.play(coords).expect("Invalid action");

                if let Status::Finished(_winner) = self.game.get_status() {
                    rew = 1f32; //if winner == current_player { 1f32 } else { -1f32 };
                    done = true;
                }
            }
            Status::Finished(_color) => {
                done = true;
            }
        }

        let obs = self.get_obs(py);
        return (obs, rew, done)
    }

    /// Board representation
    fn __repr__(&self) -> String {
        format!("{}", &self.game.get_board())
    }

    /// Get observation
    fn get_obs<'py>(&self, py: Python<'py>) -> &'py PyArray3<f32> {
        let matrix = self.game.get_board().to_ndarray();
        matrix.to_pyarray(py)
    }

    // Get all still possible actions, where the cell is empty, Returns PyList
    fn available_actions(&self) -> Vec<u32> {
        let mut actions = Vec::new();
        for x in 0..self.size {
            for y in 0..self.size {
                let color = self.game.get_board().get_color(Coords::new(x as u8, y as u8));

                if color.is_none() {
                    actions.push(y as u32 + self.size as u32 * x as u32);
                }
            }
        }
        return actions
    }



    /// Generic functions, just to copy/deepcopy objects within python
    fn copy(&self) -> Self {self.clone()}
    fn __copy__(&self) -> Self {self.clone()}
    fn __deepcopy__(&self, _memo: &PyDict) -> Self {self.clone()}
    fn to_pickle(&self) -> Vec<u8> {
        let serialized = serde_pickle::to_vec(&self, Default::default()).unwrap();
        return serialized
    }

    fn from_pickle(&mut self, serialized: Vec<u8>) {
        let deserialized: Self = serde_pickle::from_slice(&serialized, Default::default()).unwrap();
        self.game = deserialized.game;
    }
}
