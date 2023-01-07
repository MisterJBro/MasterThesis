import torch
import numpy as np
import pickle

if __name__ == '__main__':
    eps = []

    for i in range(128):
        with open(f'checkpoints/7x7/eps_{i+1}.pkl', 'rb') as fp:
            data = pickle.load(fp)
            eps.extend(data)
    print(len(eps))

    # Save
    with open('checkpoints/7x7/eps.pkl', 'wb') as fp:
        pickle.dump(eps, fp)
