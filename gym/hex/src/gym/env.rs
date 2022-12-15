use crate::{Coords, Color, Game, Status};
use crate::gym::{Action, Obs, Info};
use serde::{Deserialize, Serialize};
use numpy::ndarray::{Array, Ix1};
use std::fmt::{Debug, Formatter, Result};


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
    pub fn reset(&mut self) -> (Obs, Info) {
        self.game = Game::new(self.size);
        (self.get_obs(), self.get_info())
    }

    /// Execute next action
    #[inline]
    pub fn step(&mut self, action: Action) -> (Obs, f32, bool, Info) {
        let mut done = false;
        let mut rew = 0f32;

        match self.game.get_status() {
            Status::Ongoing(_current_player) => {
                let current_player = self.game.get_current_player();
                let is_black = if let Some(player) = current_player { player } else { Color::Black } == Color::Black;
                let coords = Coords::from_u16(action, self.size as u16, is_black);
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
        (self.get_obs(), rew, done, self.get_info())
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
        let current_player = self.game.get_current_player();
        let is_black = if let Some(player) = current_player { player } else { Color::Black } == Color::Black;
        self.game.get_board().to_ndarray(is_black)
    }

    /// Get current player id
    #[inline]
    pub fn get_pid(&self) -> u8 {
        let current_player = self.game.get_current_player();

        if let Some(player) = current_player {
            match player {
                Color::Black => 0,
                Color::White => 1,
            }
        } else {
            2
        }
    }

    /// Get legal actions
    #[inline]
    pub fn legal_actions(&self) -> Array<bool, Ix1> {
        let current_player = self.game.get_current_player();
        let is_black = if let Some(player) = current_player { player } else { Color::Black } == Color::Black;
        let mut actions = vec![false; self.size as usize * self.size as usize];
        for x in 0..self.size {
            for y in 0..self.size {
                let color = self.game.get_board().get_color(Coords::new(x as u8, y as u8));

                if color.is_none() {
                    let index = if is_black {
                        y + self.size * x
                    } else {
                        x + self.size * y
                    };
                    actions[index as usize] = true;
                }
            }
        }
        Array::from_vec(actions)
    }

    /// Get extra info
    #[inline]
    pub fn get_info(&self) -> Info {
        Info {
            pid: self.get_pid(),
            legal_act: self.legal_actions(),
        }
    }

    /// Close environment
    pub fn close(&self) { }
}

impl Debug for Env {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.to_string())
    }
}

