import numpy as np
from math import exp, sqrt, log
#from sklearn.gaussian_process.kernels import RBF
from DeepEnsembles import DeepEnsemblesEstimator

# Global variables
NUM_TRAINING_EPOCHS = 12
NUM_DATAPOINTS_PER_EPOCH = 50
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05
rng = np.random.RandomState(12345)
constant =  0.5*np.log(2*np.pi)

# State representation
# dtheta, dx, theta, x

kernel_length_scales = np.array([[240.507, 242.9594, 218.0256, 203.0197],
                                 [175.9314, 176.8396, 178.0185, 33.0219],
                                 [7.4687, 7.3903, 13.0914, 34.6307],
                                 [0.8433, 1.0499, 1.2963, 2.3903],
                                 [0.781, 0.9858, 1.7216, 31.2894],
                                 [23.1603, 24.6355, 49.9782, 219.185]])
kernel_scale_factors = np.array([3.5236, 1.3658, 0.7204, 1.1478])
noise_sigmas = np.array([0.0431, 0.0165, 0.0145, 0.0143])


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


def augmented_state(state, action):
    """
    :param state: cartpole state
    :param action: action applied to state
    :return: an augmented state for training GP dynamics
    """
    dtheta, dx, theta, x = state
    return dtheta, dx, np.sin(theta), np.cos(theta), x, action
    #return x, dx, dtheta, np.sin(theta), np.cos(theta), action


def make_training_data(state_traj, action_traj, delta_state_traj):
    """
    A helper function to generate training data.
    """
    x = np.array([augmented_state(state, action) for state, action in zip(state_traj, action_traj)])
    y = delta_state_traj
    return x, y

def compute_squared_exponential_kernel(x_1, x_2, M_kernel, sigma2_f):
    N = x_1.shape[0]
    M = x_2.shape[0]
    kernel_m = np.ndarray((N,M))
    i = 0
    j = 0
    while i < N:
        while j < M:
            # k(x_i,x_j) = \sigma_f^2 * e^(-(x_i-x_j)M(x_i-x_j)/2)
            diff = x_1[i].__add__(-x_2[j])
            kernel_m[i][j] = sigma2_f * exp(-(np.matmul(np.matmul(np.transpose(diff), M_kernel), diff))/2)
            j += 1
        j = 0
        i += 1
    return kernel_m

def predict_gaussian_process(train_x, train_y, test_x, i, M_kernel, kernel_sigma, pre_mean):
    """
    Helper function computing mean and variance for each test_x
    """
    
    kernel_combined = compute_squared_exponential_kernel(test_x, train_x, M_kernel, kernel_scale_factors[i]**2) 
    mean = np.matmul(kernel_combined, pre_mean)
    product_variance = np.matmul(np.matmul(kernel_combined, kernel_sigma), np.transpose(kernel_combined))
    variance = kernel_scale_factors[i]**2 - product_variance[0]
    return mean, variance

def learn_gaussian_process(train_x, train_y, i):
    """
    Helper function computing matrices from training_data

    :return:
        M_kernel: matrix with kernel length scales for kernel
        kernel_sigma: inverted (kernel with data - noise_sigmas) (K - \sigma*I)^{-1}
        pre_mean: inverted (kernel with data - noise_sigmas) times y
                (K - \sigma*I)^{-1}y
    """

    M_kernel = np.identity(6)
    M_kernel[0][0] = 1/kernel_length_scales[0][i]**2
    M_kernel[1][1] = 1/kernel_length_scales[1][i]**2
    M_kernel[2][2] = 1/kernel_length_scales[2][i]**2
    M_kernel[3][3] = 1/kernel_length_scales[3][i]**2
    M_kernel[4][4] = 1/kernel_length_scales[4][i]**2
    M_kernel[5][5] = 1/kernel_length_scales[5][i]**2
    N = train_x.shape[0]
    sigma_id = np.identity(N)*noise_sigmas[i]**2
    kernel_matrix = compute_squared_exponential_kernel(train_x, train_x, M_kernel, kernel_scale_factors[i]**2)
    kernel_sigma = np.linalg.inv(kernel_matrix.__add__(sigma_id))
    pre_mean = np.matmul(kernel_sigma, train_y)

    return M_kernel, kernel_sigma, pre_mean

def predict_gp(train_x, train_y, init_state, action_traj):
    """
    Let M be the number of training examples
    Let H be the length of an epoch (NUM_DATAPOINTS_PER_EPOCH)
    Let N be the number of trajectories (NUM_TRAJ_SAMPLES)

    NOTE: Please use rng.normal(mu, sigma) to generate Gaussian random noise.
          https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.normal.html


    :param train_x: a numpy array of size [M x 6]
    :param train_y: a numpy array of size [M x 4]
    :param init_state: a numpy array of size [4]. Initial state of current epoch.
                       Use this to generate rollouts.
    :param action_traj: a numpy array of size [H]

    :return: 
             # This is the mean rollout 
             pred_gp_mean: a numpy array of size [H x 4]
                           This is mu_t[k] in Algorithm 1 in the HW1 PDF.
             pred_gp_variance: a numpy array of size [H x 4]. 
                               This is sigma_t[k] in Algorithm 1 in the HW1 PDF.
             rollout_gp: a numpy array of size [H x 4]
                         This is x_t[k] in Algorithm 1 in the HW1 PDF.
                         It should start from t=1, i.e. rollout_gp[0,k] = x_1[k]
        
             # These are the sampled rollouts
             pred_gp_mean_trajs: a numpy array of size [N x H x 4]
                                 This is mu_t^j[k] in Algorithm 2 in the HW1 PDF.
             pred_gp_variance_trajs: a numpy array of size [N x H x 4]
                                     This is sigma_t^j[k] in Algorithm 2 in the HW1 PDF.
             rollout_gp_trajs: a numpy array of size [N x H x 4]
                               This is x_t^j[k] in Algorithm 2 in the HW1 PDF.
                               It should start from t=1, i.e. rollout_gp_trajs[j,0,k] = x_1^j[k]
    """
    M = train_x.shape[0]
    K = train_x.shape[1]
    #initialize arrays for pre-computed values
    M_kernel = np.array([np.zeros((K,K)), np.zeros((K,K)), np.zeros((K,K)), np.zeros((K,K))])
    kernel_sigma = [np.zeros((M,M)), np.zeros((M,M)), np.zeros((M,M)), np.zeros((M,M))]
    pre_mean = np.array([np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M))])
    #pre-compute predicting values to improve performance
    i = 0
    while(i<4):
        M_kernel[i], kernel_sigma[i], pre_mean[i] = learn_gaussian_process(train_x, train_y[:,i], i)
        i += 1

    pred_gp_mean = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_gp_variance = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_gp = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    x_t = np.zeros((NUM_DATAPOINTS_PER_EPOCH + 1, 4))
    #Algorithm 1
    x_t[0] = init_state
    i = 0
    while (i < NUM_DATAPOINTS_PER_EPOCH):
        #create augmented state
        x_hat = augmented_state(x_t[i], action_traj[i])
        j = 0
        while (j < 4):
            #mean, variance = gf(x_hat, u_t)
            pred_gp_mean[i][j], pred_gp_variance[i][j] = predict_gaussian_process(
                train_x, train_y[:,j], np.array([x_hat]), j, M_kernel[j], kernel_sigma[j], pre_mean[j])
            #compute next state
            x_t[i+1][j] = x_t[i][j] + pred_gp_mean[i][j]
            j += 1
        i += 1
    rollout_gp = np.delete(x_t, 0, 0)

    pred_gp_mean_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_gp_variance_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_gp_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    x_t = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH + 1, 4))
    #Algorithm 2
    for z in range(0, NUM_TRAJ_SAMPLES):
        x_t[z][0] = init_state
    j = 0
    while(j < NUM_TRAJ_SAMPLES):
        t = 0
        while (t < NUM_DATAPOINTS_PER_EPOCH):
            #create augmented state
            x_hat = augmented_state(x_t[j][t], action_traj[t])
            k = 0
            while (k < 4):
                #mean, variance = gf(x_hat, u_t)
                pred_gp_mean_trajs[j][t][k], pred_gp_variance_trajs[j][t][k] = predict_gaussian_process(
                    train_x, train_y[:,k], np.array([x_hat]), k, M_kernel[k], kernel_sigma[k], pre_mean[k])
                #sample state
                s = rng.normal(pred_gp_mean_trajs[j][t][k], pred_gp_variance_trajs[j][t][k], 1)
                #compute next sampled state
                x_t[j][t+1][k] = x_t[j][t][k] + s
                k += 1
            t += 1
        j += 1
    rollout_gp_trajs = np.delete(x_t, 0, 1)

    # TODO: Compute these variables.

    return pred_gp_mean, pred_gp_variance, rollout_gp, pred_gp_mean_trajs, pred_gp_variance_trajs, rollout_gp_trajs

def predict_de(model, init_state, action_traj):
    pred_de_mean = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_de_variance = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_de = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))

    pred_de_mean_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_de_variance_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_de_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))

    mean_state = init_state
    sample_state = np.tile(init_state, (NUM_TRAJ_SAMPLES, 1))
    for t in range(NUM_DATAPOINTS_PER_EPOCH):
        mean_xhat = np.array(augmented_state(mean_state, action_traj[t]))[None]
        sample_xhat = np.zeros((NUM_TRAJ_SAMPLES, 6))
        for j in range(NUM_TRAJ_SAMPLES):
            sample_xhat[j, :] = np.array(augmented_state(sample_state[j, :], action_traj[t]))

        # mean roll out
        mean, var = model.predict(mean_xhat)
        mean = mean.detach().numpy()
        var = var.detach().numpy()
        pred_de_mean[t] = mean
        pred_de_variance[t] = var
        rollout_de[t] = mean_state + mean

        # sample roll out
        mean_sample, var_sample = model.predict(sample_xhat)
        mean_sample = mean_sample.detach().numpy()
        var_sample = var_sample.detach().numpy()
        pred_de_mean_trajs[:, t, :] = mean_sample
        pred_de_variance_trajs[:, t, :] = var_sample
        rollout_de_trajs[:, t, :] = sample_state + rng.normal(mean_sample, np.sqrt(var_sample))

        mean_state = rollout_de[t, :]
        sample_state = rollout_de_trajs[:, t, :]

    return pred_de_mean, pred_de_variance, rollout_de, pred_de_mean_trajs, pred_de_variance_trajs, rollout_de_trajs

def pred_cnn(model, init_state, delta_state_traj):

    pred_cnn_mean = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_cnn_variance = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_cnn = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))

    pred_cnn_mean_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_cnn_variance_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_cnn_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))

    mean_state = init_state
    sample_state = np.tile(init_state, (NUM_TRAJ_SAMPLES, 1))
    for t in range(NUM_DATAPOINTS_PER_EPOCH):
        mean_img = np.array(make_img(mean_state, mean_state + delta_state_traj[t]))[None]
        sample_img = np.zeros((NUM_TRAJ_SAMPLES, 2, 64, 256))
        for j in range(NUM_TRAJ_SAMPLES):
            sample_img[j, :] = np.array((make_img(sample_state[j, :], sample_state[j, :] + delta_state_traj[t])))

        # mean roll out
        mean, var = model.predict(mean_img)
        mean = mean.detach().numpy()
        var = var.detach().numpy()
        pred_cnn_mean[t] = mean
        pred_cnn_variance[t] = var
        rollout_cnn[t] = mean_state + mean

        # sample roll out
        mean_sample, var_sample = model.predict(sample_img)
        mean_sample = mean_sample.detach().numpy()
        var_sample = var_sample.detach().numpy()
        pred_cnn_mean_trajs[:, t, :] = mean_sample
        pred_cnn_variance_trajs[:, t, :] = var_sample
        rollout_cnn_trajs[:, t, :] = sample_state + rng.normal(mean_sample, np.sqrt(var_sample))

        mean_state = rollout_cnn[t, :]
        sample_state = rollout_cnn_trajs[:, t, :]


    return pred_cnn_mean, pred_cnn_variance, rollout_cnn, pred_cnn_mean_trajs, pred_cnn_variance_trajs, rollout_cnn_trajs

def make_img(this_state, next_state):
    vis2 = Visualizer2(cartpole_length=1.5, x_lim=(0.0, DELTA_T * 2))
    sim = CartpoleSim(dt=DELTA_T)
    
    vis2.set_gt_cartpole_state(this_state[3], this_state[2])
    vis_img = vis2.draw(redraw=True)

    h = vis_img.shape[0]
    w = vis_img.shape[1]
    cropped = vis_img[h // 3:h * 2 // 3, ]
    resized = cv2.resize(cropped, (256, 64))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    this = np.array(resized)

    vis2.set_gt_cartpole_state(this_state[3], this_state[2])
    vis_img = vis2.draw(redraw=False)

    h = vis_img.shape[0]
    w = vis_img.shape[1]
    cropped = vis_img[h // 3:h * 2 // 3, ]
    resized = cv2.resize(cropped, (256, 64))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    nex = np.array(resized)

    return this, nex

    



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from cartpole_sim import CartpoleSim
    from policy import SwingUpAndBalancePolicy, RandomPolicy
    from visualization import Visualizer
    from data.visualization import Visualizer2
    import cv2
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="CNN", type=str)
    args = parser.parse_args()
    mode = args.mode

    vis = Visualizer(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH))
    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    random_policy = RandomPolicy(seed=12831)
    sim = CartpoleSim(dt=DELTA_T)

    # Initial training data used to train GP for the first epoch
    init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
    ts, state_traj, action_traj = sim_rollout(sim, random_policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
    delta_state_traj = state_traj[1:] - state_traj[:-1]
    train_x, train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)

    if(mode == "CNN"):
        model = DeepEnsemblesEstimator(regressor="CNN")
    else:
        model = DeepEnsemblesEstimator(regressor="MLP")

    for epoch in range(NUM_TRAINING_EPOCHS):
        vis.clear()

        # Use learned policy every 4th epoch
        if (epoch + 1) % 4 == 0:
            policy = swingup_policy
            init_state = np.array([0.01, 0.01, 0.05, 0.05]) * rng.randn(4)
        else:
            policy = random_policy
            init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

        ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        delta_state_traj = state_traj[1:] - state_traj[:-1]

        # TODO: change here to run our estimator
        if(mode == "MLP"):
            (pred_de_mean,
             pred_de_variance,
             rollout_de,
             pred_de_mean_trajs,
             pred_de_variance_trajs,
             rollout_de_trajs) = predict_de(model, state_traj[0], action_traj)

        else:
            (pred_de_mean,
             pred_de_variance,
             rollout_de,
             pred_de_mean_trajs,
             pred_de_variance_trajs,
             rollout_de_trajs) = pred_cnn(model, state_traj[0], delta_state_traj) 


        (pred_gp_mean,
         pred_gp_variance,
         rollout_gp,
         pred_gp_mean_trajs,
         pred_gp_variance_trajs,
         rollout_gp_trajs) = predict_gp(train_x, train_y, state_traj[0], action_traj)

        de_RMSE_0 = 0
        de_RMSE_1 = 0
        de_RMSE_2 = 0
        de_RMSE_3 = 0
        counter = 0
        de_nll_0 = 0
        de_nll_1 = 0
        de_nll_2 = 0
        de_nll_3 = 0

        gp_RMSE_0 = 0
        gp_RMSE_1 = 0
        gp_RMSE_2 = 0
        gp_RMSE_3 = 0
        counter = 0
        gp_nll_0 = 0
        gp_nll_1 = 0
        gp_nll_2 = 0
        gp_nll_3 = 0

        for i in range(len(state_traj) - 1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])
            vis.set_gt_delta_state_trajectory(ts[:i+1], delta_state_traj[:i+1])

            if i == 0:
                vis.set_gp_cartpole_state(state_traj[i][3], state_traj[i][2])
                vis.set_gp_cartpole_rollout_state([state_traj[i][3]] * NUM_TRAJ_SAMPLES,
                                                  [state_traj[i][2]] * NUM_TRAJ_SAMPLES)
            else:
                vis.set_gp_cartpole_state(rollout_gp[i-1][3], rollout_gp[i-1][2])
                vis.set_gp_cartpole_rollout_state(rollout_gp_trajs[:, i-1, 3], rollout_gp_trajs[:, i-1, 2])
                
            vis.set_gp_delta_state_trajectory(ts[:i+1], pred_gp_mean[:i+1], pred_gp_variance[:i+1])

            if i == 0:
                vis.set_de_cartpole_state(state_traj[i][3], state_traj[i][2])
                vis.set_de_cartpole_rollout_state([state_traj[i][3]] * NUM_TRAJ_SAMPLES,
                                                  [state_traj[i][2]] * NUM_TRAJ_SAMPLES)
            else:
                vis.set_de_cartpole_state(rollout_de[i-1][3], rollout_de[i-1][2])
                vis.set_de_cartpole_rollout_state(rollout_de_trajs[:, i-1, 3], rollout_de_trajs[:, i-1, 2])
                
            vis.set_de_delta_state_trajectory(ts[:i+1], pred_de_mean[:i+1], pred_de_variance[:i+1])

            de_RMSE_0 += (rollout_de[i-1][0] - state_traj[i][0])**2
            de_RMSE_1 += (rollout_de[i-1][1] - state_traj[i][1])**2
            de_RMSE_2 += (rollout_de[i-1][2] - state_traj[i][2])**2
            de_RMSE_3 += (rollout_de[i-1][3] - state_traj[i][3])**2

            de_nll_0 += 0.5 * log(pred_de_variance[i-1][0]) + 0.5 * (rollout_de[i-1][0] - state_traj[i][0])**2/ pred_de_variance[i-1][0] + constant
            de_nll_1 += 0.5 * log(pred_de_variance[i-1][1]) + 0.5 * (rollout_de[i-1][1] - state_traj[i][1])**2/ pred_de_variance[i-1][1] + constant
            de_nll_2 += 0.5 * log(pred_de_variance[i-1][2]) + 0.5 * (rollout_de[i-1][2] - state_traj[i][2])**2/ pred_de_variance[i-1][2] + constant
            de_nll_3 += 0.5 * log(pred_de_variance[i-1][3]) + 0.5 * (rollout_de[i-1][3] - state_traj[i][3])**2/ pred_de_variance[i-1][3] + constant


            gp_RMSE_0 += (rollout_gp[i-1][0] - state_traj[i][0])**2
            gp_RMSE_1 += (rollout_gp[i-1][1] - state_traj[i][1])**2
            gp_RMSE_2 += (rollout_gp[i-1][2] - state_traj[i][2])**2
            gp_RMSE_3 += (rollout_gp[i-1][3] - state_traj[i][3])**2

            gp_nll_0 += 0.5 * log(pred_gp_variance[i-1][0]) + 0.5 * (rollout_gp[i-1][0] - state_traj[i][0])**2/ pred_gp_variance[i-1][0] + constant
            gp_nll_1 += 0.5 * log(pred_gp_variance[i-1][1]) + 0.5 * (rollout_gp[i-1][1] - state_traj[i][1])**2/ pred_gp_variance[i-1][1] + constant
            gp_nll_2 += 0.5 * log(pred_gp_variance[i-1][2]) + 0.5 * (rollout_gp[i-1][2] - state_traj[i][2])**2/ pred_gp_variance[i-1][2] + constant
            gp_nll_3 += 0.5 * log(pred_gp_variance[i-1][3]) + 0.5 * (rollout_gp[i-1][3] - state_traj[i][3])**2/ pred_gp_variance[i-1][3] + constant

            if policy == swingup_policy:
                policy_type = 'swing up'
            else:
                policy_type = 'random'

            vis.set_info_text('epoch: %d\npolicy: %s' % (epoch, policy_type))

            vis_img = vis.draw(redraw=(i==0))
            cv2.imshow('vis', vis_img)

            if epoch == 0 and i == 0:
                # First frame
                video_out = cv2.VideoWriter('cartpole.mp4',
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                            int(1.0 / DELTA_T),
                                            (vis_img.shape[1], vis_img.shape[0]))

            video_out.write(vis_img)
            cv2.waitKey(int(1000 * DELTA_T))
        print('DeepEnsemble')
        de_RMSE_0 = sqrt(de_RMSE_0/(len(state_traj) - 1))
        de_RMSE_1 = sqrt(de_RMSE_1/(len(state_traj) - 1))
        de_RMSE_2 = sqrt(de_RMSE_2/(len(state_traj) - 1))
        de_RMSE_3 = sqrt(de_RMSE_3/(len(state_traj) - 1))
        print(str(de_RMSE_0) + ", " + str(de_RMSE_1) + ", " + str(de_RMSE_2) + ", " + str(de_RMSE_3))
        de_nll_0 /=(len(state_traj) - 1)
        de_nll_1 /=(len(state_traj) - 1)
        de_nll_2 /=(len(state_traj) - 1)
        de_nll_3 /=(len(state_traj) - 1)
        print(str(de_nll_0) + ", " + str(de_nll_1) + ", " + str(de_nll_2) + ", " + str(de_nll_3))

        print('GaussianProcesses')
        gp_RMSE_0 = sqrt(gp_RMSE_0/(len(state_traj) - 1))
        gp_RMSE_1 = sqrt(gp_RMSE_1/(len(state_traj) - 1))
        gp_RMSE_2 = sqrt(gp_RMSE_2/(len(state_traj) - 1))
        gp_RMSE_3 = sqrt(gp_RMSE_3/(len(state_traj) - 1))
        print(str(gp_RMSE_0) + ", " + str(gp_RMSE_1) + ", " + str(gp_RMSE_2) + ", " + str(gp_RMSE_3))
        gp_nll_0 /=(len(state_traj) - 1)
        gp_nll_1 /=(len(state_traj) - 1)
        gp_nll_2 /=(len(state_traj) - 1)
        gp_nll_3 /=(len(state_traj) - 1)
        print(str(gp_nll_0) + ", " + str(gp_nll_1) + ", " + str(gp_nll_2) + ", " + str(gp_nll_3))

        # Augment training data
        new_train_x, new_train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)
        train_x = np.concatenate([train_x, new_train_x])
        train_y = np.concatenate([train_y, new_train_y])
