import tensorflow as tf
import numpy as np
import gym
import tqdm
# from VLBM_for_envs_with_early_termination import *
from VLBM import *
import os
import tensorflow_probability as tfp
import multiprocessing as mp
import os
import d4rl
import json
import torch
from train_discriminator import TrajectoryClassifier
import argparse
from scipy.stats import spearmanr
from utils import generate_and_check_trajectory

slim = tf.contrib.slim
rnn = tf.contrib.rnn
tfd = tfp.distributions

parser = argparse.ArgumentParser()
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
parser.add_argument("-gamma", type=float, help="Set discounting factor DEFAULT=0.995", default=0.995)
parser.add_argument("-code_size", type=int, help="Set dimension of the latent space DEFAULT=16", default=16)
parser.add_argument("-env", type=str, help="Choose environment from <ant/hopper/walker2d>-<medium/medium-expert>-v2. Use the other script to evaluate on Halfcheetah. DEFAULT=halfcheetah-medium-expert-v2", default='walker2d-medium-expert-v2')
parser.add_argument("-max_episodes", type=int, help="Maximum number of episodes run for evaluation", default=50)
parser.add_argument("-path", type=str, help="Path to checkpoint folder")
# Below are some constants that would not be changed
parser.add_argument("-repeat", type=int, help="Set action repeat. Since we are training on offline trajectories, so this is not needed (always set to 1)", default=1)
parser.add_argument("-max_episode_len", type=int, help="Maximum episode length, which is always 1000 for Gym-Mujoco environments", default=100)

import numpy as np


def weighted_quantile(values, quantiles, sample_weight=None):
    """
    Compute weighted quantiles.

    Parameters
    ----------
    values : array-like
        Data array.
    quantiles : array-like
        Quantiles to compute, which must be between 0 and 1.
    sample_weight : array-like, optional
        Weights for each data point. If None, equal weight is assumed.

    Returns
    -------
    array-like
        The weighted quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)

    if sample_weight is None:
        sample_weight = np.ones_like(values)
    sample_weight = np.array(sample_weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_cdf = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_cdf /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_cdf, values)

def rollout_learned_env(policy_path, starting_state=None, scale=None): # This just generates a trajectory using the given path and the learned OPE model.
    file_appendix = ""

    env = gym.make(rl_params['env_name'])
    # np.random.seed(RANDOM_SEED)
    # tf.set_random_seed(RANDOM_SEED)
    # env.seed(RANDOM_SEED)

    env_state_dim = env.observation_space.shape[0]
    if "ant-" in rl_params['env_name']:
        env_state_dim = 27
    env_action_dim = env.action_space.shape[0]
    env_action_bound = env.action_space.high
    env_state_bound = None

    graph_ope_models = tf.Graph()

    graph_ac = tf.Graph()

    with tf.Session(config=config, graph=graph_ope_models) as sess_ope_models:

        with graph_ope_models.as_default():

            ope_model = OPE_Model(
                num_branch, graph_ope_models, sess_ope_models, .001, 1000, .997, CODE_SIZE,
                env_state_dim, env_state_bound, env_action_dim, file_appendix,
                4200, RANDOM_SEED, 64, MAX_EPISODE_LEN, 1.,
                is_training=False
            )

            ope_saver = ope_model.saver
            ope_saver.restore(sess_ope_models, os.path.join(ope_path, "ope_best.ckpt"))

            d4rl_qlearning = d4rl.qlearning_dataset(env)
            
            if "ant-" in rl_params['env_name']:

                obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)[:27]
                obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)[:27]
                
            else:
                
                obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)
                obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)

            rew_mean = d4rl_qlearning['rewards'].mean()
            rew_std = d4rl_qlearning['rewards'].std()


            class LearnedEnv(object):
                def __init__(self, model):
                    self.model = model

                def reset(self):
                    if starting_state is None:
                        s0 = self.model.init_z0_s0()
                    else:
                        s0 = self.model.init_z0_s0(start_state=starting_state)
                    self.obs = s0
                    return s0

                def step(self, u):
                    new_obs, reward = self.model.get_zt1_s2_r(np.reshape(u, (1, env_action_dim)))
                    self.obs = new_obs
                    self.model.update_zt()

                    return new_obs, reward, False, {}

            learned_env = LearnedEnv(ope_model)

            np.random.seed(RANDOM_SEED)
            tf.set_random_seed(RANDOM_SEED)

            ep_rewards = []
            policy = D4RL_Policy(policy_path)

            terminal = 0

            s = learned_env.reset()
            if starting_state is None:
                s = s.reshape(env_state_dim)*obs_std + obs_mean
            else:
                if scale is not None:
                    s = s.reshape(env_state_dim) + scale * obs_mean # Add a little tiny bit of noise
                else:
                    s = s.reshape(env_state_dim)
            # print("Initial state: " + str(s))
            # print("Obs mean: ", obs_mean)
            ep_reward = 0
            trajectory = [s] # test

            trajectory_actions = [] # test

            for j in tqdm.tqdm(range(MAX_EPISODE_LEN)):

                if j % REPEAT == 0:
                    if "ant-" in rl_params['env_name']:
                        a, _, _ = policy.act(np.concatenate([np.reshape(s, (env_state_dim,)), np.zeros(policy.fc0_w.shape[1]-27)]), np.zeros((env_action_dim,)))
                    else:
                        a, _, _ = policy.act(np.reshape(s, (env_state_dim,)), np.zeros((env_action_dim,)))
                trajectory_actions.append(a) # test
                s2, r, terminal, info = learned_env.step(a)
                r = r*rew_std + rew_mean
                s2 = s2.reshape(env_state_dim)*obs_std + obs_mean

                ep_reward += r*(GAMMA**j)

                s = s2
                trajectory.append(s) # test

                if terminal or j == MAX_EPISODE_LEN-1:
                    ep_rewards += [ep_reward]

                    return ep_reward, trajectory, trajectory_actions

def calculate_ips_product(t_t, t_a, target_policy, behavior_policy):
    ips_vals = []
    # T_t is the states
    ips_weight_target = 0
    ips_weight_behavior = 0
    for i in range(len(t_a)): # All of the actions in the trajectory
        # add exp
        ips_weight_target += np.log(target_policy.propensity_score(t_t[i], t_a[i]))
        ips_weight_behavior += np.log(behavior_policy.propensity_score(t_t[i], t_a[i]))

    return np.exp(ips_weight_target - ips_weight_behavior)

def train_discriminator(pi_b, pi_e):
    state_dim = 17
    return_dim = 1

    model = TrajectoryClassifier(state_dim + return_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    # Trajectories from Target Policy and Learned Env = 1
    first_term_trajs = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_first_term.pkl", 'rb'))

    trajectories_target = first_term_trajs['trajectories']
    returns_target = first_term_trajs['returns']

    # Each sample is [17-dim first state] + [1-dim return] flattened
    trajectory_inputs = np.asarray(list(trajectories_target))[:150, 0, :]
    return_inputs = np.asarray(returns_target)[:150]
    inputs_pi_e = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    labels_pi_e = torch.Tensor(np.ones(inputs_pi_e.shape[0]))

    # Trajectories from Original Env and Behavior Policy = 0
    behavior_trajectories_o = pickle.load(
        open('./saved_trajectories/' + str(pi_b) + "_o.pkl", 'rb'))  # This is always going to be similar?
    trajectories_behavior = behavior_trajectories_o['trajectories']
    returns_behavior = behavior_trajectories_o['returns']
    trajectory_inputs = np.asarray(list(trajectories_behavior))[:150, 0, :]
    return_inputs = np.asarray(returns_behavior)[:150]
    inputs_pi_b = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    labels_pi_b = torch.Tensor(np.zeros(inputs_pi_b.shape[0]))

    # Validation set
    trajectory_inputs = np.asarray(list(trajectories_target))[150:, 0, :]
    return_inputs = np.asarray(returns_target)[150:]
    val_pi_e = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    val_label_pi_e = torch.Tensor(np.ones(val_pi_e.shape[0]))

    trajectory_inputs = np.asarray(list(trajectories_behavior))[150:, 0, :]
    return_inputs = np.asarray(returns_behavior)[150:]
    val_pi_b = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    val_label_pi_b = torch.Tensor(np.zeros(val_pi_b.shape[0]))

    # Training Loop
    for epoch in range(300):
        preds = model(torch.cat([inputs_pi_e, inputs_pi_b])).squeeze(-1)
        loss = criterion(preds, torch.cat([labels_pi_e, labels_pi_b]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        val_inputs = torch.cat([val_pi_e, val_pi_b])
        val_labels = torch.cat([val_label_pi_e, val_label_pi_b])

        # Model predictions
        val_preds = model(val_inputs).squeeze(-1)  # (B,) instead of (B, 1)

        # Compute loss
        val_loss = criterion(val_preds, val_labels)

        # Optional: Compute accuracy (assumes sigmoid + 0.5 threshold)
        predicted_labels = (torch.sigmoid(val_preds) > 0.5).float()
        accuracy = (predicted_labels == val_labels).float().mean()

    print(f"Validation Loss: {val_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")  # This is not bad as a model.

    return model

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config = tf.ConfigProto(log_device_placement=False)


    GAMMA = args.gamma
    RANDOM_SEED = args.seed
    MAX_EPISODE_LEN = args.max_episode_len
    REPEAT = args.repeat # Action repeat is not needed since we are training on offline trajectories. So it's always set to 1.
    CODE_SIZE = args.code_size
    MAX_EPISODES = args.max_episodes
    n_tries_parallel = 100
    alpha = 0.05 # 95% coverage
    pi_e = 10
    generate_traj = True

    ENV = args.env
    ope_path = args.path
    rl_params = {
        'env_name':ENV,
    }

    with tf.io.gfile.GFile("./d4rl_policies.json", 'r') as f:
        policy_database = json.load(f)
    policy_metadatas = [i for i in policy_database if i['task.task_names'][0].find(rl_params['env_name'].split("-")[0]+"-")!=-1]

    env = gym.make(rl_params['env_name'])

    env_state_dim = env.observation_space.shape[0]
    if "ant-" in rl_params['env_name']:
        env_state_dim = 27
    env_action_dim = env.action_space.shape[0]
    env_action_bound = env.action_space.high
    env_state_bound = None

    graph_ope_models = tf.Graph()

    with graph_ope_models.as_default():
        tf.train.import_meta_graph(os.path.join(ope_path, "ope_best.ckpt.meta"))
        num_branch = np.asarray(list((set([int(v.name.split("/")[0].split("_")[-1]) for v in tf.trainable_variables() if v.name.find("Decoder_zt1_")!=-1])))).max()+1

    target_policy = D4RL_Policy(policy_metadatas[pi_e]['policy_path'])

    # Generate Trajectories Using the Learned Environment For Target Policy (first term)
    if generate_traj:
        policy_path = policy_metadatas[pi_e]['policy_path']
        print("********{}********".format(policy_metadatas[pi_e]['policy_path']))
        pool = mp.Pool(30)
        res = pool.map(rollout_learned_env, [policy_path for _ in range(n_tries_parallel)])
        t_rs, t_ts, t_as = zip(*res)
        first_term_trajs = {'returns': t_rs, 'trajectories': t_ts, 'actions': t_as}
        pickle.dump(first_term_trajs, open('./saved_trajectories/' + str(pi_e) + "_first_term.pkl", 'wb'))
        pool.close()
        pool.join()

    first_term_trajs = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_first_term.pkl", 'rb'))
    t_rs = first_term_trajs['returns']
    t_ts = first_term_trajs['trajectories']
    t_as = first_term_trajs['actions']
    first_term_target_rewards = t_rs  # Calculating the first term
    print("Calculated first term")

    # Generate Trajectories Using the Ground Truth Environment for the Behavior Policy
    if generate_traj:
        for pi_b in [i for i in range(10)]:
            behavior_policy = D4RL_Policy(policy_metadatas[pi_b]['policy_path'])
            pool = mp.Pool(30)
            res = pool.map(rollout_original_env, [behavior_policy for _ in range(n_tries_parallel)])
            b_o_rs, b_o_ts, b_o_as = zip(*res)
            pool.close()
            pool.join()
            behavior_trajectories_o = {'returns': b_o_rs, 'trajectories': b_o_ts, 'actions': b_o_as}
            pickle.dump(behavior_trajectories_o, open('./saved_trajectories/' + str(pi_b) + "_o.pkl", 'wb'))

    for pi_b in [i for i in range(10)]:
        behavior_trajectories_o = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_o.pkl", 'rb')) # This is always going to be similar?
        b_o_rs = behavior_trajectories_o['returns']
        b_o_ts = behavior_trajectories_o['trajectories']
        b_o_as = behavior_trajectories_o['actions']
        print("Offline data")

        discriminator = train_discriminator(pi_b=pi_b, pi_e=pi_e)
        weights = []
        scores = []
        for i in range(len(b_o_rs)):
            behavior_trajectory = b_o_ts[i]
            # if len(behavior_trajectories_diff[i]['returns']) == 0:  # No matched trajectories
            #     continue
            # else:
            s_o = behavior_trajectory[0].flatten() # First state of the behavior trajectory
            return_i = b_o_rs[i]
            input = np.hstack((s_o.reshape(1, -1), return_i.reshape(1, -1)))
            p_hat = discriminator(torch.Tensor(input))[0].squeeze(-1).detach().item()
            weight = p_hat / (1 - p_hat)
            weights.append(weight)
            scores.append(b_o_rs[i]) # Absolute value of differences in returns

        quantiles = weighted_quantile(scores, [alpha, 1-alpha], weights)
        print("pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " Interval: (" + str(quantiles[0]) + ", " + str(quantiles[1]) + ")")

    print("Calculate Ground Truth Value of Policy")

    true_target_rewards = []
    d4rl_qlearning = d4rl.qlearning_dataset(env)
    obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)
    obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)
    rew_mean = d4rl_qlearning['rewards'].mean()
    rew_std = d4rl_qlearning['rewards'].std()
    for _, i in enumerate(tqdm.tqdm(range(n_tries_parallel))): # Calculating the actual value using monte carlo sampling
        target_reward, _, _ = rollout_original_env(target_policy)
        true_target_rewards.append(target_reward)   # This will be about 20. something

    print("True target reward: " + str(np.mean(true_target_rewards)))



