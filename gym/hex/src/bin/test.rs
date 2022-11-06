use hexgame::{Envs};
use std::{thread, time};

fn main() {
    let num_cpus = 2;
    let num_envs_per_worker = 2;
    let size = 3;
    let envs = Envs::new(num_cpus, num_envs_per_worker, size);

    envs.reset();
    println!("{:?}", envs.step(vec![0, 1, 2, 3]));
    envs.render();

    // Wait a bit
    thread::sleep(time::Duration::from_millis(3_000));

    println!("Hello World! from main thread");
    envs.close();
}