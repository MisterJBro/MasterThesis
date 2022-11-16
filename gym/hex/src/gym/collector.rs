
use std::thread;
use core_affinity::CoreId;
use crossbeam::channel::{bounded, Sender, Receiver};
use crate::gym::{WorkerMessage};

/// Collector Message
#[derive(Debug)]
pub struct CollectorMessage {
}

/// Sample Collector and processor
pub struct Collector {
    thread: Option<thread::JoinHandle<()>>,
}

impl Collector {
    fn new(in_channel: Receiver<WorkerMessage>, out_channel: Sender<WorkerMessage>) -> Collector {
        let buffer: Vec<WorkerMessage> = Vec::with_capacity(1000);

        let thread = thread::spawn(move || {
            loop {
                if let Ok(message) = in_channel.recv() {
                }
            }
        });

        Collector {
            thread: Some(thread),
        }
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.thread.take() {
            handle.join().expect("Could not join collector thread!");
        }
    }
}