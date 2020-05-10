from data.dataLoader import CartPoleDataLoader
from torch.utils.data import DataLoader
import torch.nn as nn
from DeepEnsembles import DeepEnsembles
from math import sqrt
import IPython
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="CNN", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    mode = args.mode
    if(mode == "CNN"):
        loader = CartPoleDataLoader(True, True, batch_size = 32)
        ens = DeepEnsembles(regressor="CNN")
        ens.train(loader)
    else:
        loader = CartPoleDataLoader(True, False, batch_size = 32)
        ens = DeepEnsembles(regressor="MLP")
        ens.train(loader)
    ens.save()
    os.system('spd-say "Finished learning model" ')
    means_diff = [0,0,0,0]
    if(mode == "MLP"):
        loader = CartPoleDataLoader(False, True, batch_size = 1)
    else:
        loader = CartPoleDataLoader(False, False, batch_size = 1)
    for i in range(10):
        if(regressor=="CNN"):
            state, delta, img = loader.next_batch()
            mean, var = ens.ensemble_mean_var(img)
        else:
            state, delta = loader.next_batch()
            mean, var = ens.ensemble_mean_var(x)
        means_diff[0] += (mean[0][0] - delta[0][0])**2
        means_diff[1] += (mean[0][1] - delta[0][1])**2
        means_diff[2] += (mean[0][2] - delta[0][2])**2
        means_diff[3] += (mean[0][3] - delta[0][3])**2
    for i in range(4):
        means_diff[i] = sqrt(means_diff[i]/50)
    print("Averaged diff")
    print(means_diff)
    IPython.embed()