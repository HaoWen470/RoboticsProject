import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sys import platform
from hw1.hw1.cartpole.policy import SwingUpAndBalancePolicy


constant = 0.5 * np.log(2 * np.pi)
loss_fn = torch.nn.MSELoss()

from torch.utils.data import Dataset, DataLoader
import os
import cv2

NUM_DATAPOINTS_PER_EPOCH = 50
ROOT_PATH = "data/train_data/"


def augmented_state(state):
    """
    :param state: cartpole state
    :param action: action applied to state
    :return: an augmented state for training GP dynamics
    """
    return np.stack([state[..., 0],  # dtheta
                     state[..., 1],  # dx
                     np.sin(state[..., 2]),  # sin(theta)
                     np.cos(state[..., 2]),  # cos(theta)
                     state[..., 3],  # x
                     state[..., 4]], axis=-1)  # action


def loadData(img_stack=1, train_type="both", augmented=True):
    """
    this is the same function with Data/dataLoader.py/load.data function
    :param img_stack:consequent img number as input
    :param train_type: for behavior cloning, aloways "swingup"
    :param augmented: for bahavior cloning, always "False"
    :return: state
    """
    # change your root path here
    img_data = []
    random_state = np.load(ROOT_PATH + "random_state.npy")
    swingup_state = np.load(ROOT_PATH + "swingup_state.npy")
    start = img_stack - 1
    random_state_data = random_state[..., start:-1, :-1]
    random_delta_state = random_state[..., start + 1:, :-1] - random_state_data
    swingup_state_data = swingup_state[..., start:-1, :-1]
    swingup_delta_state = swingup_state[..., start + 1:, :-1] - swingup_state_data
    if augmented:
        random_state_final = augmented_state(random_state[..., start:-1, :])
        swingup_state_final = augmented_state(swingup_state[..., start:-1, :])
    else:
        random_state_final = random_state[..., start:-1, :]
        swingup_state_final = swingup_state[..., start:-1, :]

    random_img = []
    swingup_img = []

    random_filenames = os.listdir(ROOT_PATH + "random_img/")
    random_filenames.sort(key=lambda x: (int((x.split('.')[0]).split('-')[0]), int((x.split('.')[0]).split('-')[1])))
    swingup_filenames = os.listdir(ROOT_PATH + "random_img/")
    swingup_filenames.sort(key=lambda x: (int((x.split('.')[0]).split('-')[0]), int((x.split('.')[0]).split('-')[1])))

    for i in random_filenames:
        step = int((i.split('.')[0]).split('-')[1])
        img = cv2.imread(ROOT_PATH + "random_img/" + i, 0)
        img = np.array(img)
        random_img.append(img)
    for i in swingup_filenames:
        step = int((i.split('.')[0]).split('-')[1])
        img = cv2.imread(ROOT_PATH + "swingup_img/" + i, 0)
        img = np.array(img)
        swingup_img.append(img)
    if train_type == "both":
        return np.concatenate([random_state_final, swingup_state_final]), \
               np.concatenate([random_delta_state, swingup_delta_state]), np.concatenate([random_img, swingup_img])
    elif train_type == "swingup":
        return swingup_state_final, swingup_delta_state, swingup_img
    elif train_type == "random":
        return random_state_final, random_delta_state, random_img




class CartPoleDataset(Dataset):
    def __init__(self, need_img=False, img_stack=1, augmented=False):
        state, _, _ = loadData(3, train_type="swingup", augmented=False)
        self.state = state[..., :-1]
        self.action = state[..., -1]
        self.traj_num, self.datapoints, _ = self.state.shape

    def __len__(self):
        return self.traj_num * self.datapoints

    def __getitem__(self, idx):
        i = idx // self.datapoints
        j = idx % self.datapoints
        state = self.state[i, j, ...]
        action = torch.FloatTensor(self.action[i, j, ...])
        state = torch.FloatTensor(normalize_state(state))
        return state, action


def normalize_state(state):
        # Convert the state representation to the one used by the Gym Env CartpoleSwingUp
        theta_dot, x_dot, theta, x_pos = state
        theta += np.pi
        result = np.array([x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot])
        return result


class CartPoleDataLoader():
    def __init__(self, need_img=False, img_stack=3, batch_size=32, augmented=False):
        dataset = CartPoleDataset(need_img=False, augmented=augmented)
        self.loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)
        self.it = iter(self.loader)

    def next_batch(self):
        try:
            batch = next(self.it)
        except:
            self.it = iter(self.loader)
            batch = next(self.it)
        return batch


class MLPGaussianRegressor(nn.Module):
    def __init__(self, sizes):
        """
        The first number in sizes is the number of input nodes
        The last number in sizes is the number of output nodes
        """
        super(MLPGaussianRegressor, self).__init__()
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)
        self.out = sizes[-1]

    def forward(self, x):
        return self.net(x)


class BehaviorCloning():
    """
    feel free to change the sizes of this MPL network, but make sure the first size is 5 and last is 1.
    """
    def __init__(self, sizes=[5, 16,32, 64,128,64,16, 1]):
        self.regressor = MLPGaussianRegressor(sizes)
        self.optimizers = torch.optim.Adam(self.regressor.parameters(), lr=0.001)

    # change the max_iter for more precise prediction
    def train(self, data_loader, max_iter=3500, alpha=0.5, eps=5e-3):
        for it in range(max_iter):
            x, target = data_loader.next_batch()
            x = torch.FloatTensor(x)
            target = torch.FloatTensor(target).reshape((-1, 1))
            x.required_grad = True
            y = self.regressor(x)
            self.optimizers.zero_grad()

            # loss = 0
            # for i in range(len(y)):
            #     if (y[i]<0 and target[i]>0) or (y[i]>0 and target[i]<0):
            #         loss += (y[i]-target[i])**2*10
            #     else:
            #         loss += (y[i]-target[i])**2
            # loss = torch.sqrt(loss)/len(y)
            lo = loss_fn(y, target)
            # lo.data = loss

            lo.backward()
            self.optimizers.step()

            if it % 50 == 0:
                print("iter : %d; loss : %2.3f" % (it, lo.data.item()))

    def save(self):
        if not os.path.exists("weights/"):
            os.makedirs("weights/")
        file_name = "weights/BehaviorClone2.pt"
        torch.save(self.regressor.state_dict(), file_name)
        print("save model to " + file_name)

    def load(self):
        try:
            file_name = "weights/BehaviorClone.pt"
            checkpoint = torch.load(file_name)
            self.regressor.load_state_dict(checkpoint)
            print("load model from " + file_name)
        except:
            print("fail to load model")


class BehaviorCloneEstimator:
    def __init__(self, size=[5, 16,32,64,128,64,16, 1]):
        self.model = BehaviorCloning()
        self.model.load()

    def predict(self, x, normalized = False):
        with torch.no_grad():
            if not normalized:
                x = torch.FloatTensor(normalize_state(x))
            output = self.model.regressor(x)
        return output


if __name__ == "__main__":
    loader = CartPoleDataLoader(batch_size=32)
    bc = BehaviorCloning()
    bc.train(loader)
    bc.save()
    state, action = loader.next_batch()
    es = BehaviorCloneEstimator()
    predict = es.predict(state, True)
    print("prediction mean: ")
    print(predict)
    print("ground truth")
    print(action)
