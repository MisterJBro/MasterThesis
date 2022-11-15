use hexgame::{Envs};
use rand::seq::SliceRandom;
use std::time::{Duration, Instant};

fn main() {
    let num_cpus = 3;
    let num_envs_per_worker = 6;
    let size = 9;
    let mut envs = Envs::new(num_cpus, num_envs_per_worker, false, size);
    let start = Instant::now();

    let (obs, info) = envs.reset();
    let mut legal_act = info.legal_act;
    let mut rng = rand::thread_rng();

    for _ in 0..1_000 {
        // Get act by choosing randomly from legal act, which is an array where each action which is legal is false
        let act = legal_act.outer_iter().map(|x| *x.iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| i as u16).collect::<Vec<u16>>().choose(&mut rng).unwrap()).collect::<Vec<_>>();
        let act = act.iter().enumerate().map(|(i, &x)| (i as usize, x)).collect::<Vec<_>>();
        let (obs, rew, done, info) = envs.step(act, num_cpus*num_envs_per_worker);
        legal_act = info.legal_act;
    }
    let elapsed = start.elapsed();
    println!("Elapsed: {} ms", elapsed.as_millis());

    envs.close();
}