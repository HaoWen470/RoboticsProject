from data.dataLoader import CartPoleDataLoader
from torch.utils.data import DataLoader
import torch.nn as nn
from DeepEnsembles import DeepEnsembles

if __name__ == "__main__":
    loader = CartPoleDataLoader(batch_size = 32)
    ens = DeepEnsembles()
    ens.train(loader)
    ens.save()
    state, delta, _ = loader.next_batch()
    mean, var = ens.ensemble_mean_var(state)
    print("prediction mean: ")
    print(mean[0])
    print("ground truth")
    print(delta[0])
    IPython.embed()