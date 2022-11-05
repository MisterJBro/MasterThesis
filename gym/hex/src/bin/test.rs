use hexgame::{Envs};
use std::{thread, time};

fn main() {
    let num_cpus = 3;
    let num_envs_per_worker = 5;
    let size = 9;
    let envs = Envs::new(num_cpus, num_envs_per_worker, size);

    envs.reset();

    // Wait a bit
    thread::sleep(time::Duration::from_millis(3_000));

    println!("Hello World! from main thread");
    envs.close();
}