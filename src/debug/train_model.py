from src.networks.residual_model import ValueEquivalenceModel
import torch
import numpy as np
import torch
import numpy as np
from torch.multiprocessing import freeze_support
from src.train.config import create_config
from src.env.hex import HexEnv
import pickle

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 7
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "device": "cuda:0",

        "model_lr": 1e-3,
        "model_weight_decay": 1e-5,
        "model_iters": 10,
        "model_unroll_len": 5,
        "model_num_res_blocks": 10,
        "model_num_filters": 128,
        "model_batch_size": 512,
    })

    class PythonEps():
        def __init__(self, eps):
            self.obs = eps["obs"]
            self.act = np.concatenate((eps["act"], np.random.randint(0, size*size, 1)))
            self.rew = np.zeros(len(self.act))
            self.dist = eps["pi"]

    # Import data and model
    model = ValueEquivalenceModel(config)
    #/work/scratch/jb66zuhe/eps7x7.pkl
    with open('checkpoints/eps7x7.pkl', 'rb') as fp:
        data = pickle.load(fp)

    # Prepare data
    eps = []
    val = []
    for i in range(len(data)):
        eps.append(PythonEps(data[i]))
        val.append(data[i]["v"])

    # Print V for start obs
    start_obs = torch.zeros((1, 2, size, size)).float().to(config["device"])
    start_state = model.representation(start_obs)
    start_dist, start_val = model.prediction(start_state)
    print(start_val.item())

    # Train
    print(f"(Before) Model loss: {model.test(eps, val):.04f}")
    model.loss(eps, val)
    print(f"(After) Model loss: {model.test(eps, val):.04f}")

    # Print v after
    start_obs = torch.zeros((1, 2, size, size)).float().to(config["device"])
    start_state = model.representation(start_obs)
    start_dist, start_val = model.prediction(start_state)
    print(start_val.item())

    # Save model
    model.save("checkpoints/m_7x7_10_128.pt")