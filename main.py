from data.dataLoader import CartPoleDataLoader
from torch.utils.data import DataLoader
import torch.nn as nn
from DeepEnsembles import DeepEnsembles
from math import sqrt
import IPython
import os

if __name__ == "__main__":
    loader = CartPoleDataLoader(True, batch_size = 32)
    ens = DeepEnsembles()
    ens.train(loader)
    ens.save()
    os.system('spd-say "Finished learning model" ')
    means_diff = [0,0,0,0]
    loader = CartPoleDataLoader(False, batch_size = 1)
    for i in range(10):
        state, delta, img = loader.next_batch()
        mean, var = ens.ensemble_mean_var(img)
        means_diff[0] += (mean[0][0] - delta[0][0])**2
        means_diff[1] += (mean[0][1] - delta[0][1])**2
        means_diff[2] += (mean[0][2] - delta[0][2])**2
        means_diff[3] += (mean[0][3] - delta[0][3])**2
        #print("prediction mean: ")
        #print(mean[0])
        #print("ground truth")
        #print(delta[0])
    for i in range(4):
        means_diff[i] = sqrt(means_diff[i]/50)
    print("Averaged diff")
    print(means_diff)
    IPython.embed()