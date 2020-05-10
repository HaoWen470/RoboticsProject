import numpy as np
import os

from cartpole_sim import CartpoleSim
from policy import SwingUpAndBalancePolicy, RandomPolicy
from visualization import Visualizer2
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--dp', default=50, type=int)
args = parser.parse_args()

NUM_DATAPOINTS_PER_EPOCH = args.dp
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05

def sim_rollout(sim, policy, n_steps, dt, init_state):
    """
    :param sim: the simulator
    :param policy: policy that generates rollout
    :param n_steps: number of time steps to run
    :param dt: simulation step size
    :param init_state: initial state

    :return: times:   a numpy array of size [n_steps + 1]
             states:  a numpy array of size [n_steps + 1 x 4]
             actions: a numpy array of size [n_steps]
                        actions[i] is applied to states[i] to generate states[i+1]
    """
    states = []
    state = init_state
    actions = []
    for i in range(n_steps):
        states.append(state)
        action = policy.predict(state)
        actions.append(action)
        state = sim.step(state, [action], noisy=True)

    states.append(state)
    times = np.arange(n_steps + 1) * dt
    return times, np.array(states), np.array(actions)


if __name__ == "__main__":
    rng = np.random.RandomState(12345)
    init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
    # num of train data epoch
    random_epoch = args.epoch
    swingup_epoch = args.epoch
    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    random_policy = RandomPolicy(seed=12831)
    sim = CartpoleSim(dt=DELTA_T)
    random_data = []
    swingup_data = []

    vis = Visualizer2(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH))
    root = "test_data/"
    if not os.path.exists(root):
        os.makedirs(root)

    for epoch in range(random_epoch):
        subpath = "random_img/"
        if not os.path.exists(root+subpath):
            os.makedirs(root+subpath)
        policy = random_policy
        init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

        ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        conca_data = np.concatenate((state_traj, np.append(action_traj, 0).reshape(-1, 1)), axis=1)
        random_data.append(conca_data)


        # make random data
        for i in range(len(state_traj)-1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])

            # if i == 0:
            #     vis.set_gp_cartpole_state(state_traj[i][3], state_traj[i][2])
            #     vis.set_gp_cartpole_rollout_state([state_traj[i][3]] * NUM_TRAJ_SAMPLES,
            #                                       [state_traj[i][2]] * NUM_TRAJ_SAMPLES)

            vis_img = vis.draw(redraw=(i == 0))

            filename = str(epoch) + "-" + str(i) + ".png"

            # do crop and resize
            h = vis_img.shape[0]
            w = vis_img.shape[1]
            cropped = vis_img[h // 3:h * 2 // 3, ]
            resized = cv2.resize(cropped, (256, 64))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(root+subpath+filename, resized)
            
    for epoch in range(swingup_epoch):
        subpath = "swingup_img/"
        if not os.path.exists(root + subpath):
            os.makedirs(root + subpath)
        policy = swingup_policy
        init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

        ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        conca_data = np.concatenate((state_traj, np.append(action_traj, 0).reshape(-1, 1)), axis=1)
        swingup_data.append(conca_data)

        # make random data
        for i in range(len(state_traj)-1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])

            # if i == 0:
            #     vis.set_gp_cartpole_state(state_traj[i][3], state_traj[i][2])
            #     vis.set_gp_cartpole_rollout_state([state_traj[i][3]] * NUM_TRAJ_SAMPLES,
            #                                       [state_traj[i][2]] * NUM_TRAJ_SAMPLES)

            vis_img = vis.draw(redraw=(i == 0))

            filename = str(epoch) + "-" + str(i) + ".png"

            # do crop and resize
            h = vis_img.shape[0]
            w = vis_img.shape[1]
            cropped = vis_img[h // 3:h * 2 // 3, ]
            resized = cv2.resize(cropped, (256, 64))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(root + subpath + filename, resized)

    np.save(root+'random_state.npy', np.array(random_data))
    np.save(root + 'swingup_state.npy', np.array(swingup_data))
