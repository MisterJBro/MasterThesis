use hexgame::search::{MCTS, State};
use hexgame::gym::{Env};

fn main () {
    // Env
    let mut env = Env::new(5);
    let (mut obs, mut info) = env.reset();
    env.render();

    // MCTS
    let mcts = MCTS::new(3);
    let state = State { obs, rew: 0.0, done: false, info, env };
    let result = mcts.search(state, 100);

    println!("Result: {:?}", result);
}

