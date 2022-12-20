use hexgame::gym::{Envs};
use rand::seq::SliceRandom;
use std::time::{Duration, Instant};
use fixed::{types::extra::U13, FixedI64};

pub fn from_stats(stats: usize) -> (u32, f32) {
    // Get sum returns from fixed point float to ieee float
    let num_visits = (stats >> 40) as u32;// ((stats << 40) >> 40) as u32;

    let sum_returns_bits = ((stats as i64) << 24) >> 24;
    let sum_returns_fixed = FixedI64::<U13>::from_bits(sum_returns_bits);
    let sum_returns = sum_returns_fixed.saturating_to_num::<f32>();

    (num_visits, sum_returns)
}

pub fn to_stats(num_visits: u32, sum_returns: f32) -> usize {
    let num_visits = (num_visits as usize) << 40;

    // Get sum returns from ieee float to fixed point float
    let sum_returns_fixed = FixedI64::<U13>::saturating_from_num(sum_returns);
    let sum_returns_bits = sum_returns_fixed.to_bits() as usize;
    let sum_returns_bits_trim = (sum_returns_bits << 24) >> 24;

    let stats = sum_returns_bits_trim | num_visits;
    stats
}

fn main() {
    let a = to_stats(2, 13.4);
    let b = to_stats(4, 1000000004.2);
    let c = a + b;

    println!("flags: {:#066b} = {}", c, c);

    let (num_visits, sum_returns) = from_stats(c);

    println!("num_visits: {}", num_visits);
    println!("sum_returns: {}", sum_returns);

}