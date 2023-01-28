from src.networks.residual_model import ValueEquivalenceModel
import torch
import numpy as np
import torch
import numpy as np
from torch.multiprocessing import freeze_support
from src.train.config import create_config
from src.env.hex import HexEnv
import pickle
from random import shuffle


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 5
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "device": "cuda:0",

        "model_lr": 1e-4,
        "model_weight_decay": 0,
        "model_iters": 1,
        "model_unroll_len": 5,
        "model_num_res_blocks": 2,
        "model_num_filters": 128,
        "model_batch_size": 512,
        "use_se": True,
    })

    class PythonEps():
        def __init__(self, eps):
            self.obs = eps["obs"]
            self.act = np.concatenate((eps["act"], np.random.randint(0, size*size, 1)))
            self.rew = np.zeros(len(self.act))
            self.dist = eps["pi"]

    # Import data and model
    losses = [1000]
    model = ValueEquivalenceModel(config)
    #model.load("/work/scratch/jb66zuhe/m_7x7_14_128.pt")

    # Get train and test data
    #/work/scratch/jb66zuhe/eps7x7.pkl
    with open('checkpoints/eps5x5.pkl', 'rb') as fp:
        data = pickle.load(fp)
    shuffle(data)
    n = len(data)
    train_data = data[:int(0.9*n)]
    test_data = data[int(0.9*n):]

    def get_eps(data):
        # Prepare data
        eps = []
        val = []
        for i in range(len(data)):
            eps.append(PythonEps(data[i]))
            val.append(data[i]["v"])
        return eps, val

    eps_train, val_train = get_eps(train_data)
    eps_test, val_test = get_eps(test_data)

    # Print V for start obs
    start_obs = torch.zeros((1, 2, size, size)).float().to(config["device"])
    start_state = model.representation(start_obs)
    start_dist, start_val = model.prediction(start_state)
    print("Start obs val: ", start_val.item())
    print(f"(Before) Test loss: {model.test(eps_test, val_test):.04f}")

    for i in range(10):
        # Train
        model.loss(eps_train, val_train)
        test_loss = model.test(eps_test, val_test)
        print(f"(After) Test loss: {test_loss:.04f}")

        if min(losses) < 0.1 and test_loss > losses[-1]:
            break
        losses.append(test_loss)

        # Print v after
        start_obs = torch.zeros((1, 2, size, size)).float().to(config["device"])
        start_state = model.representation(start_obs)
        start_dist, start_val = model.prediction(start_state)
        print("Start obs val: ", start_val.item())

        # Save model
        model.save(f"/work/scratch/jb66zuhe/m_{size}x{size}_{config['model_num_res_blocks']}_{config['model_num_filters']}.pt")