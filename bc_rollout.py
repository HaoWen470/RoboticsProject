import numpy as np
import torch

NUM_TRAINING_EPOCHS = 12
NUM_DATAPOINTS_PER_EPOCH = 50
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05
rng = np.random.RandomState(12345)
np.set_printoptions(suppress=True)


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


def augmented_state2(state, action):
    """
    :param state: cartpole state
    :param action: action applied to state
    :return: an augmented state for training GP dynamics
    """
    dtheta, dx, theta, x = state
    return dtheta, dx, np.sin(theta), np.cos(theta), x, action


def bc_rollout(sim, bc, de, n_steps, dt, init_state):
    """
    :param sim: the simulator
    :param bc: behavior cloning model
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
    state_truth = state
    for i in range(n_steps):
        states.append(state)
        if i == 0:
            action = 0
        else:
            action = bc.predict(state)
        action_truth = policy.predict(state_truth)
        state_truth = sim.step(state, [action], noisy=True)
        mean_xhat = np.array(augmented_state2(state, action))[None]
        mean, var = de.predict(mean_xhat)
        delta_predict = np.squeeze(mean.detach().numpy())
        state = state + delta_predict
        # state = mean

        actions.append(action)
        # aug_state = augmented_state(full_state).reshape((1,-1))

        # state, _ = de.predict(torch.FloatTensor(aug_state))
        # state = np.squeeze(state)

        # state_truth = sim.step(state, [action_truth], noisy=True)

    states.append(state)
    times = np.arange(n_steps + 1) * dt
    return times, np.array(states), np.array(actions)


if __name__ == "__main__":
    from cartpole_sim import CartpoleSim
    from BCvisualization import Visualizer
    from BehaviorClone import BehaviorCloneEstimator
    from cartpole_test import sim_rollout
    from policy import SwingUpAndBalancePolicy
    from DeepEnsembles import DeepEnsemblesEstimator
    import cv2

    vis = Visualizer(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH))
    bc = BehaviorCloneEstimator()
    de = DeepEnsemblesEstimator(regressor="MLP")
    sim = CartpoleSim(dt=DELTA_T)
    policy = SwingUpAndBalancePolicy('policy.npz')

    for epoch in range(NUM_TRAINING_EPOCHS):
        vis.clear()

        # Use learned policy every 4th epo

        init_state = np.array([0.01, 0.015, 0.055, 0.075]) * rng.randn(4)
        ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        if epoch == 1:
            ts2, bc_state, bc_action = bc_rollout(sim, bc, de, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        else:
            ts2, bc_state, bc_action = bc_rollout(sim, bc, de, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        for i in range(len(state_traj) - 1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])

            if i == 0:
                vis.set_gp_cartpole_state(state_traj[i][3], state_traj[i][2])
                vis.set_gp_cartpole_rollout_state([state_traj[i][3]] * NUM_TRAJ_SAMPLES,
                                                  [state_traj[i][2]] * NUM_TRAJ_SAMPLES)
            else:
                vis.set_gp_cartpole_state(bc_state[i][3], bc_state[i][2])
                vis.set_gp_cartpole_rollout_state([bc_state[i][3]] * NUM_TRAJ_SAMPLES,
                                                  [bc_state[i][2]] * NUM_TRAJ_SAMPLES)

            # vis.set_info_text('epoch: %d\npolicy: %s' % (epoch, policy_type))

            vis_img = vis.draw(redraw=(i == 0))
            cv2.imshow('vis', vis_img)

            if epoch == 0 and i == 0:
                # First frame
                video_out = cv2.VideoWriter('cartpole.mp4',
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                            int(1.0 / DELTA_T),
                                            (vis_img.shape[1], vis_img.shape[0]))

            video_out.write(vis_img)
            cv2.waitKey(int(1000 * DELTA_T))
