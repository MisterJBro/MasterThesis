use std::thread;
use crossbeam::channel::{Sender, Receiver};
use numpy::ndarray::{Array, Ix1, stack, Axis};
use crate::gym::{Action, Obs, Info, Env, Episode};
use std::sync::Arc;
use ort::tensor::{FromArray, InputTensor};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, SessionBuilder};
use std::ops::Deref;
use ndarray_stats::QuantileExt;
use std::time::Duration;

pub enum EvalMessageIn {
    Update {path: &'static str},
    Close,
}

pub struct EvalWorkerMessageIn {
    obs: Obs,
    idx: usize,
}

pub struct EvalWorkerMessageOut {
    pi: Array<f32, Ix1>,
    v: f32,
}

/// Evaluation Service, manages the policy/networks which typically live on the GPU. Bundles requests from workers for more efficient inference.
pub struct Evaluator {
    thread: Option<thread::JoinHandle<()>>,
}

impl Evaluator {
    pub fn new(path: &'static str, exec_provider: ExecutionProvider, master_channel: Receiver<EvalMessageIn>, eval_in: Receiver<EvalWorkerMessageIn>, eval_out: Vec<Sender<EvalWorkerMessageOut>>) -> Evaluator {
        // Environment
        let environment = Environment::builder()
			.with_name("onnx_env")
			.with_log_level(LoggingLevel::Warning)
			.with_execution_providers([exec_provider])
			.build().unwrap();

        // Session
        let mut session = SessionBuilder::new(&Arc::new(environment)).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
            .with_parallel_execution(false).unwrap()
            .with_intra_threads(1).unwrap()
            .with_model_from_file(path).unwrap();

        let thread = thread::spawn(move || {
            loop {
                // Check master every 50 milliseconds
                if let Ok(master_msg) = master_channel.try_recv() {

                }

                // Check worker
                if let Ok(first_msg) = eval_in.recv_timeout(Duration::from_millis(50)) {
                    let start = std::time::Instant::now();

                    // Stack messages
                    let msg: Vec<_> = eval_in.try_iter().chain(std::iter::once(first_msg)).collect();
                    let msg: Vec<_> = msg.iter().map(|m| m.obs.view()).collect();
                    let msg = stack(Axis(0), &msg).unwrap();

                    // Inference
                    let input = InputTensor::from_array(msg.into_dyn());
	                let out: Vec<_> = session.run([input]).unwrap();

                    // Run inference
                    let (pi, v) = session.run(&[("input", &input_tensor)], &["pi", "v"]).unwrap();

                    // Send messages
                    for i in 0..num_msgs {
                        let pi = pi.get(i).unwrap().deref().clone();
                        let v = v.get(i).unwrap().deref().clone();
                        let msg = EvalWorkerMessageOut {
                            pi,
                            v,
                        };
                        eval_out[idxs[i]].send(msg).unwrap();
                    }
                }
            }
        });

        Evaluator {
            thread: Some(thread),
        }
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.thread.take() {
            handle.join().expect("Could not join env Evaluator thread!");
        }
    }
}