from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import cv2

NUM_DATAPOINTS_PER_EPOCH = 50
ROOT_PATH = "train_data/"

def augmented_state(state):
    """
    :param state: cartpole state
    :param action: action applied to state
    :return: an augmented state for training GP dynamics
    """
    return np.stack([state[..., 0], # dtheta
                     state[..., 1], # dx
                     np.sin(state[..., 2]), # sin(theta)
                     np.cos(state[..., 2]), # cos(theta)
                     state[..., 3], # x
                     state[..., 4]], axis = -1)  # action

def loadData(img_stack = 3, train_type = "both"):
    # change your root path here
    img_data = []
    random_state = np.load(ROOT_PATH+"random_state.npy")
    swingup_state = np.load(ROOT_PATH+"swingup_state.npy")
    start = img_stack-1
    random_state_data = random_state[..., start:-1, :-1]
    random_delta_state = random_state[...,start+1:, :-1]-random_state_data
    random_state_augmented = augmented_state(random_state[..., start:-1, :])
    swingup_state_data = swingup_state[..., start:-1, :-1]
    swingup_delta_state = swingup_state[..., start + 1:, :-1] - swingup_state_data
    swingup_state_augmented = augmented_state(swingup_state[..., start:-1, :])

    # augment state
    

    random_img = []
    swingup_img = []

    random_filenames = os.listdir(ROOT_PATH+"random_img/")
    random_filenames.sort(key = lambda x : (int((x.split('.')[0]).split('-')[0]),int((x.split('.')[0]).split('-')[1])))
    swingup_filenames = os.listdir(ROOT_PATH + "random_img/")
    swingup_filenames.sort(key=lambda x: (int((x.split('.')[0]).split('-')[0]), int((x.split('.')[0]).split('-')[1])))

    for i in random_filenames:
        step = int((i.split('.')[0]).split('-')[1])
        img = cv2.imread(ROOT_PATH+"random_img/"+i)
        img = np.array(img)
        random_img.append(img)
    for i in swingup_filenames:
        step = int((i.split('.')[0]).split('-')[1])
        img = cv2.imread(ROOT_PATH+"swingup_img/"+i)
        img = np.array(img)
        swingup_img.append(img)
    if train_type=="both":
        return np.concatenate([random_state_augmented,swingup_state_augmented]), \
               np.concatenate([random_delta_state, swingup_delta_state]), np.concatenate([random_img, swingup_img])
    elif train_type=="swingup":
        return swingup_state_augmented, swingup_delta_state, swingup_img
    elif train_type == "random":
        return random_state_augmented, random_delta_state, random_img

class CartPoleDataset(Dataset):
    def __init__(self,  need_img = False, img_stack = 3):
        state_augmented, delta_state, img_data = loadData(3)
        self.state_augmented = state_augmented
        self.delta_state = delta_state
        self.need_img = need_img
        self.img_data = img_data
        self.img_stack = 3
        self.traj_num, self.datapoints, _ = self.state_augmented.shape
        self.img_datapoints = self.datapoints + self.img_stack - 1

    def augment_state(self, state, action):
        return 

    def __len__(self):
        return self.traj_num * self.datapoints

    def __getitem__(self, idx):
        i = idx//self.datapoints
        j = idx%self.datapoints
        state = torch.tensor(self.state_augmented[i, j, ...])
        delta = torch.tensor(self.delta_state[i, j,...])
        if self.need_img:
            imgs = torch.tensor(self.img_data[i*self.img_datapoints+j : i*self.img_datapoints+j+self.img_stack])
            return (state, delta, imgs)
        else:
            return (state, delta)

# example of using this data loader
if __name__ == "__main__":
    #loader = CartPoleDataset(need_img=True)
    dataset = CartPoleDataset(need_img = True)
    loader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers=4)
    # during training
    for step, (state, delta, imgs) in enumerate(loader):
        # do something here
        print("at %d iteration: " % step)
        print(state.shape)
        print(delta.shape)
        print(imgs.shape)
    print("finish data loading")

