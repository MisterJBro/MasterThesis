use hexgame::search::{MCTS, State};
use hexgame::gym::{Action, Env};
use ndarray_stats::QuantileExt;

fn get_action(state: State) -> Action {
    let start = std::time::Instant::now();

    // MCTS
    let mcts = MCTS::new(3);
    let result = mcts.search(state, 1_000);
    let pi = result.pi;

    // Get argmax of pi
    let action = pi.argmax().unwrap() as u16;
    println!("Action {}  Time {:.3}s", action, start.elapsed().as_secs_f32());

    action
}

fn main() {
    let mut env = Env::new(5);
    let mut rew = 0f32;
    let mut done = false;

    let (mut obs, mut info) = env.reset();
    env.render();

    while !done {
        let act = get_action(State { obs, rew, done, info, env: env.clone() });
        (obs, rew, done, info) = env.step(act);
        env.render();
    }
    env.close();
}
