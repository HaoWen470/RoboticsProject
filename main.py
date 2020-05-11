from data.dataLoader import CartPoleDataLoader
from torch.utils.data import DataLoader
import torch.nn as nn
from DeepEnsembles import DeepEnsembles
from math import sqrt
# import IPython
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="CNN", type=str)
args = parser.parse_args()

constant =  0.5*np.log(2*np.pi)

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
    means_RMSE = [0,0,0,0]
    means_NLL = [0, 0, 0, 0]
    if(mode == "CNN"):
        loader = CartPoleDataLoader(False, True, batch_size = 1)
    else:
        loader = CartPoleDataLoader(False, False, batch_size = 1)
    for i in range(10):
        if(mode == "CNN"):
            state, delta, img = loader.next_batch()
            mean, var = ens.ensemble_mean_var(img)
        else:
            state, delta = loader.next_batch()
            mean, var = ens.ensemble_mean_var(state)
        means_RMSE[0] += (mean[0][0] - delta[0][0])**2
        means_RMSE[1] += (mean[0][1] - delta[0][1])**2
        means_RMSE[2] += (mean[0][2] - delta[0][2])**2
        means_RMSE[3] += (mean[0][3] - delta[0][3])**2
        means_NLL[0] += 0.5 * var[0][0] + 0.5 * (mean[0][0] - delta[0][0])**2 / var[0][0] + constant
        means_NLL[1] += 0.5 * var[0][1] + 0.5 * (mean[0][1] - delta[0][1])**2 / var[0][1] + constant
        means_NLL[2] += 0.5 * var[0][2] + 0.5 * (mean[0][2] - delta[0][2])**2 / var[0][2] + constant
        means_NLL[3] += 0.5 * var[0][3] + 0.5 * (mean[0][3] - delta[0][3])**2 / var[0][3] + constant
    for i in range(4):
        means_RMSE[i] = sqrt(means_RMSE[i]/50)
        means_NLL[i] = means_NLL[i]/50
    print("RMSE")
    print(means_RMSE)
    print("Deep Ensemble")
    print(means_NLL)
    IPython.embed()