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
import argparse
from scipy.stats import spearmanr
from utils import generate_and_check_trajectory
print("New Iteration")

# TODO: need to hyperparameter tune epsilon_r, alpha_r
# TODO: save the generated trajectories
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

def find_similar_trajectories(data_dict, new_traj, new_reward, epsilon=1.5, epsilon_reward=1.5):
    """
    data_dict: dict with keys 'returns', 'trajectories', 'actions'
        - 'returns': list of floats
        - 'trajectories': list of np.ndarrays, each of shape (T, D)
        - 'actions': list of integers or list-like
    new_traj: np.ndarray of shape (T, D)
    new_reward: float
    epsilon: float, threshold for comparing first and last state (only first 8 dims)
    epsilon_reward: float, threshold for comparing reward

    Returns:
        A filtered dict with the same keys, containing only matching elements
    """
    matched_returns = []
    matched_trajectories = []
    matched_actions = []

    for r, traj, a in zip(data_dict['returns'], data_dict['trajectories'], data_dict['actions']):
        # Compare first and last state (first 8 dims)
        first_close = np.linalg.norm(traj[0][:8] - new_traj[0][:8]) <= epsilon
        # last_close = np.linalg.norm(traj[-1][:8] - new_traj[-1][:8]) < epsilon
        reward_close = abs(r - new_reward) <= epsilon_reward

        if first_close and reward_close: # last_close
            matched_returns.append(r)
            matched_trajectories.append(traj)
            matched_actions.append(a)

    return matched_returns, matched_trajectories, matched_actions

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
    epsilon = 0.8
    n_tries = 40
    n_tries_parallel = 70
    alpha = 0.05 # 95% coverage
    pi_e = 10
    generate_traj = False
    alpha_r = 0

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
    print("Calculating first term")

    # Generate Trajectories Using the Ground Truth Environment for the Behavior Policy
    if generate_traj:
        for pi_b in range(11):
            pool = mp.Pool(30)
            behavior_policy = D4RL_Policy(policy_metadatas[pi_b]['policy_path'])
            res = pool.map(rollout_original_env, [behavior_policy for _ in range(n_tries_parallel)])
            b_o_rs, b_o_ts, b_o_as = zip(*res)
            pool.close()
            pool.join()
            behavior_trajectories_o = {'returns': b_o_rs, 'trajectories': b_o_ts, 'actions': b_o_as}
            pickle.dump(behavior_trajectories_o, open('./saved_trajectories/' + str(pi_b) + "_o.pkl", 'wb'))

    for pi_b in range(11):
        behavior_policy = D4RL_Policy(policy_metadatas[pi_b]['policy_path'])
        behavior_trajectories_o = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_o.pkl", 'rb')) # This is always going to be similar?
        b_o_rs = behavior_trajectories_o['returns']
        b_o_ts = behavior_trajectories_o['trajectories']
        b_o_as = behavior_trajectories_o['actions']
        print("Offline data")

        # Generate Trajectories Using the Learned Environment that already match in the first state, and filter to find stuff that matches in the return
        behavior_trajectories_diff = {}
        epsilon_r = 3
        if generate_traj:
            print("Generating trajectories now")
            for i, b_o_t in enumerate(b_o_ts):
                s_0 = b_o_t[0] # This is the first state in the trajectory
                s_last = b_o_t[-1] # This is the last state in the trajectory
                behavior_return = b_o_rs[i] # This is the return of the trajectory
                policy_path = policy_metadatas[pi_b]['policy_path']
                behavior_policy = D4RL_Policy(policy_path)
                scales = [np.random.normal(loc=0, scale=0.5) for _ in range(5)]
                pool = mp.Pool(5)
                res = pool.starmap(rollout_learned_env, [(policy_path, s_0, scales[ll]) for ll in range(5)]) # Used because function takes two arguments
                gen_returns, gen_trajs, gen_actions = zip(*res)
                pool.close()
                pool.join()
                behavior_trajectories_diff[i] = {'returns':[], 'trajectories':[], 'actions':[]}
                for j in range(len(gen_returns)):
                    if np.abs(gen_returns[j] - behavior_return) < epsilon_r: # If the rewards are pretty close
                        behavior_trajectories_diff[i]['returns'].append(gen_returns[j])
                        behavior_trajectories_diff[i]['trajectories'].append(gen_trajs[j])
                        behavior_trajectories_diff[i]['actions'].append(gen_actions[j])
                if len(behavior_trajectories_diff[i]['returns']) == 0:
                    print("Trajectory: " + str(i) + " had no matched trajectories")
                pickle.dump(behavior_trajectories_diff, open('./saved_trajectories/' + str(pi_b) + "_diff.pkl", 'wb'))
        behavior_trajectories_diff = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_diff.pkl", 'rb')) # For every sample in the behavior dataset, there are > 0 trajectories that match
        print("Finished generating matching trajectories")


        weights = []

        for i in range(len(b_o_ts)): # For all trajectories in the behavior dataset
            # Get the trajectories that already match
            if len(behavior_trajectories_diff[i]['returns']) == 0: # No matched trajectories
                weights += [0] # We don't have matched trajectories
                continue
            else:
                matched_trajs = behavior_trajectories_diff[i]
                expectation_over_matched_trajs = []
                for j in range(len(matched_trajs['returns'])):
                    traj = matched_trajs['trajectories'][j]
                    traj_actions = matched_trajs['actions'][j]
                    expectation_over_matched_trajs.append(calculate_ips_product(traj, traj_actions, target_policy, behavior_policy))
                expectation_over_matched_trajs = np.asarray(expectation_over_matched_trajs)
                w_hat = np.nanmean(expectation_over_matched_trajs[~np.isnan(expectation_over_matched_trajs) & ~np.isinf(
                    expectation_over_matched_trajs)])  # This is the expectation over the filtered trajectories
                weights += [w_hat]

        # Normalize all the weights
        weights = np.asarray(weights)/np.sum(weights)
        all_weights = []
        scores = []
        new_weighted_errors = []
        for i in range(len(b_o_rs)):
            if len(behavior_trajectories_diff[i]['returns']) == 0:  # No matched trajectories
                continue
            else:
                matched_trajectories = behavior_trajectories_diff[i]
                # For every trajectory that matched
                for j in range(len(matched_trajectories['returns'])):
                    all_weights.append(weights[i])
                    scores.append(np.abs(b_o_rs[i] - matched_trajectories['returns'][j]))
                    new_weighted_errors.append(weights[i] * np.abs(b_o_rs[i] - matched_trajectories['returns'][j])) # This should be the absolute value of the difference between returns

        quantiles = weighted_quantile(scores, [1 - alpha, alpha], all_weights)

        #   TODO: double check that this is not symmetric?
        print("Behavior Policy=" + str(pi_b) + " Target Policy:" + str(pi_e) + " Interval: (" + str(np.mean(first_term_target_rewards) - quantiles[0]) + ", " + str(
            (np.mean(first_term_target_rewards) - quantiles[1])) + ")")


    true_target_rewards = []
    d4rl_qlearning = d4rl.qlearning_dataset(env)
            
    obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)
    obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)
    rew_mean = d4rl_qlearning['rewards'].mean()
    rew_std = d4rl_qlearning['rewards'].std()

    for _, i in enumerate(tqdm.tqdm(range(n_tries))): # Calculating the actual value using monte carlo sampling
        target_reward, _, _ = rollout_original_env(target_policy)
        true_target_rewards.append(target_reward)   # This will be about 20. something

    print("True target reward: " + str(np.mean(true_target_rewards)))



