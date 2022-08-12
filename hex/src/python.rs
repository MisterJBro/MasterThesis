use crate::{Coords};
use crate::{Game, Status};
use numpy::ToPyArray;
use numpy::{PyArray3};
use pyo3::prelude::*;

/// Game Interface for use in Python
#[pyclass(name = "HexGame")]
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
    fn reset(&mut self) {
        self.game = Game::new(self.size);
    }

    /// Execute the current actions and update the board
    fn step<'py>(&mut self, py: Python<'py>, action: u16) -> (&'py PyArray3<f32>, f32, bool) {
        let mut done = false;
        let mut rew = 0f32;

        match self.game.get_status() {
            Status::Ongoing(current_player) => {
                let coords = Coords::from_u16(action, self.size as u16);
                self.game.play(coords).expect("Invalid action");

                if let Status::Finished(winner) = self.game.get_status() {
                    rew = if winner == current_player { 1f32 } else { -1f32 };
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
}
