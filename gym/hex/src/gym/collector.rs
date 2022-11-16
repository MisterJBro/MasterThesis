
use std::thread;
use crossbeam::channel::{Sender, Receiver};
use crate::gym::{WorkerMessageOut};

// Collector Messages
#[derive(Debug)]
pub enum CollectorMessageIn {
    Add{sample: WorkerMessageOut},
    Clear,
    Get,
}
#[derive(Debug)]
pub struct CollectorMessageOut {
}

/// Sample Collector and processor
pub struct Collector {
    thread: Option<thread::JoinHandle<()>>,
}

impl Collector {
    pub fn new(in_channel: Receiver<CollectorMessageIn>, out_channel: Sender<CollectorMessageOut>) -> Collector {
        let mut buffer = Vec::with_capacity(10_000);

        let thread = thread::spawn(move || {
            loop {
                if let Ok(message) = in_channel.recv() {
                    match message {
                        CollectorMessageIn::Add{sample} => {
                            // Add samples to buffer
                            buffer.push(sample);
                        },
                        CollectorMessageIn::Clear => {
                            // Clear buffer
                            buffer.clear();
                        },
                        CollectorMessageIn::Get => {
                            // Get samples from buffer
                            let msg = CollectorMessageOut{};
                            let episodes = vec![1];
                            // Send
                            if out_channel.try_send(msg).is_err() {
                                panic!("Error sending message to master");
                            }
                        },
                    }
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

/// Process single episode of data
fn process_episode() {

}