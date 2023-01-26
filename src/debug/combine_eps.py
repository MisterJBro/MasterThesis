import torch
import numpy as np
import pickle
import os, os.path

if __name__ == '__main__':
    eps = []
    path = 'data/all'

    # Number of files
    num_files = len([n for n in os.listdir(f'{path}') if os.path.isfile(path + "/" + n)])
    print("Number files:", num_files)
    if num_files == 0:
        exit()

    # Iterate episodes
    for i in range(num_files):
        with open(f'{path}/eps_{i+1}.pkl', 'rb') as fp:
            data = pickle.load(fp)
            eps.extend(data)
    print("Number eps: ", len(eps))

    # Save
    with open(f'{path}/eps.pkl', 'wb') as fp:
        pickle.dump(eps, fp)
