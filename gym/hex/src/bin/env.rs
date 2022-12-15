use hexgame::gym::{Env, Action, Obs, Info, Infos};
use rand::prelude::SliceRandom;
use numpy::ndarray::{Array, Ix1};

fn random_action(info: Info) -> Action {
    let legal_act = info.legal_act;
    let acts = legal_act.iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| i as Action).collect::<Vec<Action>>();
    let mut rng = rand::thread_rng();
    *acts.choose(&mut rng).unwrap()
}

fn main() {
    let mut env = Env::new(5);
    let mut done = false;

    let (mut obs, mut info) = env.reset();
    env.render();

    while !done {
        let act = random_action(info);
        (obs, _, done, info) = env.step(act);
        env.render();
    }
    env.close();
}