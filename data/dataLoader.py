from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
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
    return np.stack([state[..., 0], # dtheta
                     state[..., 1], # dx
                     np.sin(state[..., 2]), # sin(theta)
                     np.cos(state[..., 2]), # cos(theta)
                     state[..., 3], # x
                     state[..., 4]], axis = -1)  # action

def loadData(img_stack = 3, train_type = "both", augmented=True):
    # change your root path here
    img_data = []
    random_state = np.load(ROOT_PATH+"random_state.npy")
    swingup_state = np.load(ROOT_PATH+"swingup_state.npy")
    start = img_stack-1
    random_state_data = random_state[..., start:-1, :-1]
    random_delta_state = random_state[...,start+1:, :-1]-random_state_data
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

    random_filenames = os.listdir(ROOT_PATH+"random_img/")
    random_filenames.sort(key = lambda x : (int((x.split('.')[0]).split('-')[0]),int((x.split('.')[0]).split('-')[1])))
    swingup_filenames = os.listdir(ROOT_PATH + "random_img/")
    swingup_filenames.sort(key=lambda x: (int((x.split('.')[0]).split('-')[0]), int((x.split('.')[0]).split('-')[1])))

    for i in random_filenames:
        step = int((i.split('.')[0]).split('-')[1])
        img = cv2.imread(ROOT_PATH+"random_img/"+i, 0)
        img = np.array(img)
        random_img.append(img)
    for i in swingup_filenames:
        step = int((i.split('.')[0]).split('-')[1])
        img = cv2.imread(ROOT_PATH+"swingup_img/"+i, 0)
        img = np.array(img)
        swingup_img.append(img)
    if train_type=="both":
        return np.concatenate([random_state_final,swingup_state_final]), \
               np.concatenate([random_delta_state, swingup_delta_state]), np.concatenate([random_img, swingup_img])
    elif train_type=="swingup":
        return swingup_state_final, swingup_delta_state, swingup_img
    elif train_type == "random":
        return random_state_final, random_delta_state, random_img

def wrapAngle(x):
    return (x+np.pi) % (2*np.pi) - np.pi


class CartPoleDataset(Dataset):
    def __init__(self,  need_img = False, img_stack = 3, augmented = True, seq_num = 1):
        state_augmented, delta_state, img_data = loadData(img_stack, augmented = augmented)
        self.state_augmented = state_augmented
        self.state_augmented[..., 2] = wrapAngle(self.state_augmented[..., 2])
        self.delta_state = delta_state
        self.need_img = need_img


        self.img_data = (255.0 - img_data) / 255.0
        self.img_stack = img_stack
        self.seq_num = seq_num
        self.traj_num, self.datapoints, _ = self.state_augmented.shape
        self.datapoints = self.datapoints - self.seq_num + 1
        self.img_datapoints = self.datapoints + self.img_stack - 1
        

    def __len__(self):
        return self.traj_num * self.datapoints

    def __getitem__(self, idx):
        i = idx//self.datapoints
        j = idx%self.datapoints
        if self.seq_num == 1:
            state = torch.FloatTensor(self.state_augmented[i, j, ...])
            delta = torch.FloatTensor(self.delta_state[i, j,...])
            imgs = torch.FloatTensor(self.img_data[i*self.img_datapoints+j : i*self.img_datapoints+j+self.img_stack])
        else:
            state = torch.FloatTensor(self.state_augmented[i, j:j+self.seq_num, ...])
            delta = torch.FloatTensor(self.delta_state[i, j:j+self.seq_num, ...])
            imgs = []
            for idx in range(j, j+self.seq_num):
                imgs.append(self.img_data[i*self.img_datapoints+idx : i*self.img_datapoints+idx+self.img_stack])
            imgs = torch.FloatTensor(imgs)

        if self.need_img:
            return (state, delta, imgs)
        else:
            return (state, delta)

class CartPoleDataLoader():
    def __init__(self, need_img = True, img_stack = 3, batch_size = 32, augmented = True):
        dataset = CartPoleDataset(need_img = True, augmented = augmented)
        self.loader = DataLoader(dataset, batch_size, shuffle = True, num_workers=1)
        self.it = iter(self.loader)

    def next_batch(self):
        try:
            batch = next(self.it)
        except:
            self.it = iter(self.loader)
            batch = next(self.it)
        return batch

# example of using this data loader
if __name__ == "__main__":
    #loader = CartPoleDataset(need_img=True)
    dataset = CartPoleDataset(need_img = True)
    loader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers=1)
    # during training
    for step, (state, delta, imgs) in enumerate(loader):
        # do something here
        print("at %d iteration: " % step)
        print(state.shape)
        print(delta.shape)
        print(imgs.shape)
    print("finish data loading")

