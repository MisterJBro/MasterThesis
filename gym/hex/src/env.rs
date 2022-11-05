use crate::{Coords};
use crate::{Game, Status};
use serde::{Deserialize, Serialize};
use numpy::ndarray::{Array, Ix3};

// Basic types
type Action = u16;
type Obs = Array<f32, Ix3>;

/// The Environment
#[derive(Clone, Serialize, Deserialize)]
pub struct Env {
    pub size: u8,
    pub game: Game,
}

impl Env {
    pub fn new(size: u8) -> Env {
        Env {
            size,
            game: Game::new(size),
        }
    }

    /// Reset env
    #[inline]
    pub fn reset(&mut self) -> (Obs, Vec<Action>) {
        self.game = Game::new(self.size);
        (self.get_obs(), self.legal_actions())
    }

    /// Execute next action
    #[inline]
    pub fn step(&mut self, action: Action) -> (Obs, f32, bool, Vec<Action>) {
        let mut done = false;
        let mut rew = 0f32;

        match self.game.get_status() {
            Status::Ongoing(_current_player) => {
                let coords = Coords::from_u16(action, self.size as u16);
                self.game.play(coords).expect("Invalid action");

                if let Status::Finished(_winner) = self.game.get_status() {
                    rew = 1f32;
                    done = true;
                }
            }
            Status::Finished(_color) => {
                done = true;
            }
        }

        (self.get_obs(), rew, done, self.legal_actions())
    }

    /// Board representation
    pub fn to_string(&self) -> String {
        format!("{}", &self.game.get_board())
    }

    /// Render
    pub fn render(&self) {
        println!("{}", &self.game.get_board());
    }

    /// Get observation
    #[inline]
    pub fn get_obs(&self) -> Obs {
        self.game.get_board().to_ndarray()
    }

    // Get legal actions
    #[inline]
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut actions = Vec::new();
        for x in 0..self.size {
            for y in 0..self.size {
                let color = self.game.get_board().get_color(Coords::new(x as u8, y as u8));

                if color.is_none() {
                    actions.push(y as u16 + self.size as u16 * x as u16);
                }
            }
        }
        actions
    }

    // Close environment
    pub fn close(&self) { }
}

