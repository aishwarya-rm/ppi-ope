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

def evaluate(target_policy_path): # This just generates a trajectory using the given path and the learned OPE model. 
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

            # class LearnedEnv(object):
            #     def __init__(self, model):

            #         self.model = model

            #     def reset(self):
            #         self.model.init_z0_s0()
            #         s0 = self.model.sess.run(self.model.decoder_state_sample, 
            #                            feed_dict={self.model.decoder_zt_holder:self.model.zt}).reshape(-1)

            #         self.obs = s0
            #         return s0

            #     def step(self, u):
            #         new_obs, reward, terminal = self.model.get_zt1_s2_r(np.reshape(u, (1, env_action_dim)))
            #         self.obs = new_obs
            #         self.model.update_zt()

            #         return new_obs, reward, terminal, {}

            class LearnedEnv(object):
                def __init__(self, model):
                    self.model = model

                def reset(self):
                    s0 = self.model.init_z0_s0()

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
            policy = D4RL_Policy(target_policy_path)

            terminal = 0

            s = learned_env.reset()
            s = s.reshape(env_state_dim)*obs_std + obs_mean
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
    for i in range(len(t_a)): # All of the actions in the trajectory
        # add exp
        ips_weight = np.log(target_policy.propensity_score(t_t[i], t_a[i])) - np.log(
            behavior_policy.propensity_score(t_t[i], t_a[i]))
        if np.isinf(ips_weight):  # when divide by zero happens
            continue
        else:
            ips_vals.append(ips_weight)
    return np.exp(np.sum(ips_vals)) # add exp

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
    epsilon = 0.25
    n_tries = 100
    n_tries_parallel = 70
    alpha = 0.05 # 95% coverage

    ENV = args.env

    # assert "halfcheetah" not in ENV, "This script only work for Ant, Hopper, Walker2d which consider early termination of episodes. To train on Halfcheetah please use the other script."

    ope_path = args.path

    rl_params = {
        'env_name':ENV,
    }

    with tf.io.gfile.GFile("./d4rl_policies.json", 'r') as f:
        policy_database = json.load(f)
    policy_metadatas = [i for i in policy_database if i['task.task_names'][0].find(rl_params['env_name'].split("-")[0]+"-")!=-1]

    # Determine number of branches of VLBM 
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

    with graph_ope_models.as_default():
        tf.train.import_meta_graph(os.path.join(ope_path, "ope_best.ckpt.meta"))
        num_branch = np.asarray(list((set([int(v.name.split("/")[0].split("_")[-1]) for v in tf.trainable_variables() if v.name.find("Decoder_zt1_")!=-1])))).max()+1

    preds = []
    truths = []

    for i in [10]: # test (Generate trajectories from policy 1)
        print("target policy start generation")

        policy_path = policy_metadatas[i]['policy_path']
        target_policy = D4RL_Policy(policy_path)
        
        print("********{}********".format(policy_metadatas[i]['policy_path']))

        truths += [np.loadtxt("./truth_discounted/" + policy_path + ".txt")[0]]


        pool = mp.Pool(30)
        res = pool.map(evaluate, [policy_path for _ in range(n_tries_parallel)])
        t_rs, t_ts, t_as = zip(*res)
        pool.close()
        pool.join()

    for i in [9]: # test # Generate trajectories from policy 0
        print("behavior policy start generation")

        policy_path = policy_metadatas[i]['policy_path']
        behavior_policy = D4RL_Policy(policy_path)
        
        print("********{}********".format(policy_metadatas[i]['policy_path']))

        truths += [np.loadtxt("./truth_discounted/" + policy_path + ".txt")[0]]

        pool = mp.Pool(30)
        res = pool.map(evaluate, [policy_path for _ in range(n_tries_parallel)])
        b_rs, b_ts, b_as = zip(*res)
        pool.close()
        pool.join()

    results_t = []
    results_b = []
    # Search among the trajectories that you have generated to create a calbiration dataset.
    for i in range(n_tries_parallel):
        b_r, b_t, b_a = b_rs[i], b_ts[i], b_as[i]
        t_r, t_t, t_a = t_rs[i], t_ts[i], t_as[i]
        if (np.linalg.norm(t_t[0][:8] - b_t[0][:8]) < epsilon):
            if (np.linalg.norm(t_t[-1][:8] - b_t[-1][:8]) < epsilon):
                obj = {'tr':t_r, 't_t':t_t, 't_a':t_a, 'b_r':b_r, 'b_t':b_t, 'b_a':b_a}
                if not os.path.exists("./calibration_dataset/"):
                    os.mkdir("./calibration_dataset/")
                pickle.dump(obj, open("./calibration_dataset/" + str(i) + ".pkl", 'wb'))
                results_t += [(t_r, t_t, t_a)]
                results_b += [(b_r, b_t, b_a)]

    assert len(results_t) > 0, "results_t length is zero!"
    assert len(results_b) > 0, "results_b length is zero!"
    print("len of found matched trajectories: ", len(results_t))

    filtered_t_rs, filtered_t_ts, filtered_t_as = zip(*results_t)
    filtered_b_rs, filtered_b_ts, filtered_b_as = zip(*results_b)

    target_returns = filtered_t_rs
    behavior_returns = filtered_b_rs
    predicted_ips_weights = []
    for i in range (len(behavior_returns)): # Should only be on the calibration dataset.
        b_t = filtered_b_ts[i]
        b_a = filtered_b_as[i]
        predicted_ips_weights += [calculate_ips_product(b_t, b_a, target_policy, behavior_policy)] # Calculates the weights for all terms in the behavior trajectory
    
    first_term_target_rewards = t_rs # Calculating the first term
    ips_weighted_errors = []
    regular_errors = []
    for i in range(len(target_returns)):
        # should norm or absolute: actual_returns[i] - predicted_returns[i]
        ips_weighted_errors.append(np.abs(predicted_ips_weights[i] * behavior_returns[i] - target_returns[i])) # This should be the absolute value of the difference between returns
        regular_errors.append(np.abs(behavior_returns[i] - target_returns[i]))
    print("predicted_ips_weights ", predicted_ips_weights)
    print("ips_weighted_errors ", ips_weighted_errors)
    print("target_returns ", target_returns)
    print("behavior_returns ", behavior_returns)
    print("unweighted_errors ", regular_errors)


    true_target_rewards = []
    d4rl_qlearning = d4rl.qlearning_dataset(env)
            
    obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)
    obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)
    rew_mean = d4rl_qlearning['rewards'].mean()
    rew_std = d4rl_qlearning['rewards'].std()

    for _, i in enumerate(tqdm.tqdm(range(n_tries))): # Calculating the actual value using monte carlo sampling
        target_reward, _, _ = generate_trajectory(target_policy, env, obs_std, obs_mean, MAX_EPISODE_LEN, env_state_dim, env_action_dim, rew_std, rew_mean)
        true_target_rewards.append(target_reward)    

    # import ipdb; ipdb.set_trace()
    print("With IPS weighting")
    print(np.mean(first_term_target_rewards), np.mean(first_term_target_rewards) - np.quantile(ips_weighted_errors, 1 - alpha), (np.mean(first_term_target_rewards) - np.quantile(ips_weighted_errors, alpha)), np.mean(true_target_rewards))

    print("Without IPS weighting")
    print(np.mean(first_term_target_rewards), np.mean(first_term_target_rewards) - np.quantile(regular_errors, 1 - alpha), (np.mean(first_term_target_rewards) - np.quantile(regular_errors, alpha)), np.mean(true_target_rewards))


    # preds = np.asarray(preds)
    # truths = np.asarray(truths)
    # print ("MAE:", np.mean(np.abs((preds - truths))))
    #
    # rank, _ = spearmanr(preds, truths)
    # print ("Rank:", rank)
    #
    # print("Regret:", (np.max(truths) - truths[np.argmax(preds)])/np.max(truths))
    



