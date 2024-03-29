/*!
This is a simple and performant implementation of the Hex board game.

While it was mainly written to be used in Monte-Carlo Tree Search bots for Hex, it aims to be a general-purpose library for Hex.

Features:

* rules of the game (without swap rule, see below),
* serialize/deserialize to/from JSON,
* some analysis functions that may be helpful when writing bots: `get_neighbors`, `get_empty_cells`, `find_attacked_bridges`. See the `Board` struct for more information.

# The Game of Hex

The rules of Hex are very simple: Hex is played by two players (Black and White) on a board of hexagons. The size is variable, typical sizes are 11x11 or 19x19.
Starting with Black, the players take turns to place stones of their color on an empty space. The task of Black is to connect the top and bottom edges.
White needs to connect left and right edges. The first player to connect their edges wins the game.

In the following example, Black (●) has won the game after connecting the top edge to the bottom edge.
```text
 a  b  c  d  e
1\.  .  .  ○  ●\1
 2\.  .  ○  ●  .\2
  3\.  .  ●  ●  .\3
   4\.  .  ○  ●  .\4
    5\.  ○  ●  ○  .\5
       a  b  c  d  e
```

Hex was invented in the 1940s by Piet Hein and, independently, by John Nash. Please read the [Wikipedia](https://en.wikipedia.org/wiki/Hex_(board_game)) page for more information.

## The Swap Rule

As starting player, Black has a huge advantage. In real games, this advantage is circumvented by the so-called swap rule: After Black has placed the first Stone, the second player can choose to either continue the game normally or to swap colors.

This library does not yet implement the swap rule, mostly because it was not necessary for the MCTS-research using this library.

# How to use this library

The most important struct is `Game`. To play the game, you also need `Coords`. The game keeps track of the current player automatically.
```
use hexgame::{Coords, Game};

let mut game = Game::new(19); // size of the board

game.play(Coords::new(3, 5));  // (row, column) and zero-based, i.e. f4

// Or use human-readable coordinates
game.play("d5".parse().unwrap());
```

`game.board` can be used to access the cells of the board (e.g. `get_color(coords)`).

## Serialization

Serialization functionality requires `use hexgame::Serialization;`.

To serialize a game, use `game.save_to_string()` or `game.save_to_json()`, which serializes to a [Serde](https://serde.rs/) value.
`Game::load_from_str` or `Game::load_from_json` can be used to create a game from a JSON string or value.

# Playing Hex on CLI

While this package is mostly a library, it also contains a small command-line interface to player Hex.
```text
cargo run
```
Then type the coordinates of the space where you would like to place your next stone, e.g. "c2" and press Enter.

Optionally, you can specify the size of the board like in `cargo run 7`.
*/
extern crate core_affinity;

mod attacked_bridges;
mod board;
mod color;
mod coords;
mod edges;
mod errors;
mod format;
mod game;
mod hex_cells;
mod neighbors;
mod serialize;
mod union_find;
pub mod gym;
pub mod search;
mod python;

pub use crate::board::{Board, StoneMatrix, MAX_BOARD_SIZE, MIN_BOARD_SIZE};
pub use crate::color::Color;
pub use crate::coords::{CoordValue, Coords};
pub use crate::edges::{CoordsOrEdge, Edge};
pub use crate::errors::{InvalidBoard, InvalidMove};
pub use crate::game::{Game, Status};
pub use crate::serialize::Serialization;
pub use crate::python::{PyEnv, PyEnvs, PyEpisode, PyMCTS, PyState};
use pyo3::prelude::*;

/// Register rust translator functions
#[pymodule]
fn hexgame(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEnv>()?;
    m.add_class::<PyEnvs>()?;
    m.add_class::<PyEpisode>()?;
    m.add_class::<PyMCTS>()?;
    m.add_class::<PyState>()?;
    Ok(())
}