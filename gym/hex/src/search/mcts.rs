use crate::gym::{Env, Envs, Action, Obs, Info, Infos, CollectorMessageOut, Episode};

struct MCTS {
    env: Env,
    root: Node,
    n: usize,
    c: f32,
}

impl MCTS {
    fn new(env: Env, n: usize, c: f32) -> MCTS {
        MCTS {
            env,
            root: Node::new(),
            n,
            c,
        }
    }

    fn search(&mut self) {
        for _ in 0..self.n {
            let mut env = self.env.clone();
            let mut node = &mut self.root;
            let mut path = vec![];

            // Selection
            while !node.is_leaf() {
                let (child, action) = node.select(self.c);
                path.push((node, action));
                node = child;
                env.step(action);
            }

            // Expansion
            if !node.is_expanded() {
                node.expand(&env);
            }

            // Simulation
            let (rew, info) = node.simulate(&env);

            // Backpropagation
            for (node, action) in path {
                node.update(action, rew, info);
            }
        }
    }

    fn get_action(&self) -> Action {
        self.root.select_best()
    }
}
