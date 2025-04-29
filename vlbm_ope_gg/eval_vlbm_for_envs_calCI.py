import tensorflow as tf
import numpy as np
import tqdm
# from VLBM_for_envs_with_early_termination import *
from VLBM import *
import os
import tensorflow_probability as tfp
import multiprocessing as mp
import os
import json
import argparse
from scipy.stats import spearmanr
from utils import generate_trajectory

# TODO: may need to take out randomness in trajectory generation if IPS weights get too big.
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
parser.add_argument("-max_episode_len", type=int, help="Maximum episode length, which is always 1000 for Gym-Mujoco environments", default=20)

def evaluate(policy): # This just generates a trajectory using the given path and the learned OPE model.
    file_appendix = ""

    # env = gym.make(rl_params['env_name'])
    # np.random.seed(RANDOM_SEED)
    # tf.set_random_seed(RANDOM_SEED)
    # env.seed(RANDOM_SEED)

    env_state_dim = 1
    env_action_dim = 1
    env_action_bound = 10
    env_state_bound = 10

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

            inventory_data = dict(np.load("data/inventory_data.npz", allow_pickle=True))
            print(inventory_data.keys())

            obs_mean = np.mean(inventory_data['observations'])
            obs_std = np.std(inventory_data['observations'])

            rew_mean = np.mean(inventory_data['rewards'])
            rew_std = np.std(inventory_data['rewards'])


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

            terminal = 0

            s = learned_env.reset()
            s = s.reshape(env_state_dim)*obs_std + obs_mean
            ep_reward = 0
            trajectory = [] 
            trajectory_actions = [] 

            for j in tqdm.tqdm(range(MAX_EPISODE_LEN)):

                if j % REPEAT == 0:
                    a = policy.act(np.reshape(s, (env_state_dim,)))
                trajectory_actions.append(a) 
                s2, r, terminal, info = learned_env.step(a)
                r = r*rew_std + rew_mean # This is the reward in the original space
                s2 = s2.reshape(env_state_dim)*obs_std + obs_mean

                ep_reward += r*(GAMMA**j)
                trajectory.append(s*obs_std + obs_mean) # This is the state in the original space
                s = s2

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
            # ips_vals.append(ips_weight)
            ips_vals.append(min(10, max(1e-8, ips_weight))) # Clamping the IPS weights
    return np.exp(np.sum(ips_vals)) # add exp

def find_similar_trajectories(data_dict, new_traj, new_reward, epsilon=0.8, epsilon_reward=1.5):
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
        first_close = np.linalg.norm(traj[0][:1] - new_traj[0][:1]) < epsilon
        # last_close = np.linalg.norm(traj[-1][:8] - new_traj[-1][:8]) < epsilon
        reward_close = abs(r - new_reward) < epsilon_reward

        if first_close and reward_close: # last_close
            matched_returns.append(r)
            matched_trajectories.append(traj)
            matched_actions.append(a)

    return matched_returns, matched_trajectories, matched_actions
    

class uniform_policy(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def act(self, s):
        a = np.random.uniform(self.a, self.b, (1,))
        return a

class sS_policy(object):
    def __init__(self, s, S):
        self.a = s
        self.b = S

    def act(self, s):
        if s < self.a:
            a = self.b - s
        else:
            a = np.random.uniform(0, 0, (1,))
        return a
        

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
    n_tries = 100
    n_tries_parallel = 1000
    alpha = 0.05 # 95% coverage
    alpha_r = 6

    ENV = "inventory"
    ope_path = "./saved_model/VLBM_inventory_1000iter_0.0001_1000_0.997_16_1.0_2599/"
    rl_params = {
        'env_name':ENV,
    }

    # with tf.io.gfile.GFile("./d4rl_policies.json", 'r') as f:
    #     policy_database = json.load(f)
    # policy_metadatas = [i for i in policy_database if i['task.task_names'][0].find(rl_params['env_name'].split("-")[0]+"-")!=-1]

    # env = gym.make(rl_params['env_name'])
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    # env.seed(RANDOM_SEED)

    env_state_dim = 1
    env_action_dim = 1
    env_action_bound = 10
    env_state_bound = 10

    graph_ope_models = tf.Graph()

    with graph_ope_models.as_default():
        tf.train.import_meta_graph(os.path.join(ope_path, "ope_best.ckpt.meta"))
        num_branch = np.asarray(list((set([int(v.name.split("/")[0].split("_")[-1]) for v in tf.trainable_variables() if v.name.find("Decoder_zt1_")!=-1])))).max()+1

    pi_u_0_10 = uniform_policy(0, 10)
    pi_u_0_5 = uniform_policy(0, 5)
    pi_u_3_8 = uniform_policy(3, 8)
    pi_u_0_0 = uniform_policy(0, 0)
    pi_u_6_6 = uniform_policy(6, 6)

    pi_sS_3_10 = sS_policy(3, 10)
    pi_sS_3_6 = sS_policy(3, 6)
    pi_sS_7_10 = sS_policy(7, 10)
    pi_sS_1_5 = sS_policy(1, 5)
    pi_sS_5_8 = sS_policy(5, 8)

    policies = [pi_u_0_10, pi_u_0_5, pi_u_3_8, pi_u_0_0, pi_u_6_6,
            pi_sS_3_10, pi_sS_3_6, pi_sS_7_10, pi_sS_1_5, pi_sS_5_8]
    
    pi_e = 9
    pi_b = 0
        
    preds = []
    with open("./data/inventory_values.json", 'r') as f:
        truths = dict(json.load(f))
    # truths_list = []
    # Generate Trajectories Using the Learned Environment For Target Policy (first term)
    generate_traj = True
    if generate_traj:
        for i in [pi_e]:
            print("target policy start generation, policy=" + str(i))

            target_policy = policies[i]

            # truths_list += [truths['sS_policy_5_8']]

            pool = mp.Pool(30)
            res = pool.map(evaluate, [target_policy for _ in range(n_tries_parallel)])
            t_rs, t_ts, t_as = zip(*res)
            first_term_trajs = {'returns': t_rs, 'trajectories': t_ts, 'actions': t_as}
            pickle.dump(first_term_trajs, open('./saved_trajectories/' + str(pi_e) + "_first_term.pkl", 'wb+'))
            pool.close()
            pool.join()

    first_term_trajs = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_first_term.pkl", 'rb'))
    t_rs = first_term_trajs['returns']
    t_ts = first_term_trajs['trajectories']
    t_as = first_term_trajs['actions']
    first_term_target_rewards = t_rs  # Calculating the first term

    # Generate Trajectories Using the Learned Environment For Behavior Policy
    generate_traj = True
    if generate_traj:
        for i in [pi_b]:
            print("behavior policy start generation, policy=" + str(i))
            
            behavior_policy = policies[i]
            # truths += []
            pool = mp.Pool(30)
            res = pool.map(evaluate, [behavior_policy for _ in range(n_tries_parallel)])
            b_gen_rs, b_gen_ts, b_gen_as = zip(*res)
            pool.close()
            pool.join()
            behavior_trajectories_diff = {'returns':b_gen_rs, 'trajectories':b_gen_ts, 'actions':b_gen_as}
            pickle.dump(behavior_trajectories_diff, open('./saved_trajectories/' + str(pi_b) + "_diff.pkl", 'wb+'))
    behavior_trajectories_diff = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_diff.pkl", 'rb'))
    b_gen_rs = behavior_trajectories_diff['returns']
    b_gen_ts = behavior_trajectories_diff['trajectories']
    b_gen_as = behavior_trajectories_diff['actions']


    # Generate Trajectories Using the Ground Truth Environment for the Behavior Policy
    generate_traj = False
    if generate_traj:
        pool = mp.Pool(30)
        res = pool.map(generate_trajectory, [policies[pi_b] for _ in range(n_tries_parallel)])
        b_o_rs, b_o_ts, b_o_as = zip(*res)
        pool.close()
        pool.join()
        behavior_trajectories_o = {'returns': b_o_rs, 'trajectories': b_o_ts, 'actions': b_o_as}
        pickle.dump(behavior_trajectories_o, open('./saved_trajectories/' + str(pi_b) + "_o.pkl", 'wb+'))
    behavior_trajectories_o = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_o.pkl", 'rb')) # This is always going to be similar?
    b_o_rs = behavior_trajectories_o['returns']
    b_o_ts = behavior_trajectories_o['trajectories']
    b_o_as = behavior_trajectories_o['actions']

    results_b_o = []
    results_b_gen = []
    print("Creating Calibration Dataset")
    generate_cal = False
    if generate_cal:
        # Search among the trajectories that you have generated to create a calbiration dataset.
        for i in range(70):
            b_o_r, b_o_t, b_o_a = b_o_rs[i], b_o_ts[i], b_o_as[i]
            for j in range(n_tries_parallel):
                b_gen_r, b_gen_t, b_gen_a = b_gen_rs[j], b_gen_ts[j], b_gen_as[j]
                if (np.linalg.norm(b_o_t[0][:1] - b_gen_t[0][:1]) < epsilon):
                    obj = {'b_o_r':b_o_r, 'b_o_t':b_o_t, 'b_o_a':b_o_a, 'b_gen_r':b_gen_r, 'b_gen_t':b_gen_t, 'b_gen_a':b_gen_a}
                    if not os.path.exists("./calibration_dataset/"):
                        os.mkdir("./calibration_dataset/")
                    pickle.dump(obj, open("./calibration_dataset/" + str(i) + ".pkl", 'wb+'))
                    results_b_o += [(b_o_r, b_o_t, b_o_a)]
                    results_b_gen += [(b_gen_r, b_gen_t, b_gen_a)]

        assert len(results_b_o) > 0, "results_b_o length is zero!"
        assert len(results_b_gen) > 0, "results_b_gen length is zero!"
        print("len of found matched trajectories: ", len(results_b_o))
    else:
        # Load the calibration dataset
        results_b_o = []
        results_b_gen = []
        for i in range(70):
            obj = pickle.load(open("./calibration_dataset/" + str(i) + ".pkl", 'rb'))
            b_o_r, b_o_t, b_o_a = obj['b_o_r'], obj['b_o_t'], obj['b_o_a']
            b_gen_r, b_gen_t, b_gen_a = obj['b_gen_r'], obj['b_gen_t'], obj['b_gen_a']
            results_b_o += [(b_o_r, b_o_t, b_o_a)]
            results_b_gen += [(b_gen_r, b_gen_t, b_gen_a)]
            print("len of found matched trajectories: ", len(results_b_o))

    filtered_b_o_rs, filtered_b_o_ts, filtered_b_o_as = zip(*results_b_o)
    filtered_b_gen_rs, filtered_b_gen_ts, filtered_b_gen_as = zip(*results_b_gen)

    # Create a large set of trajectories from the original enviornment (TODO may need to change to the learned environment idk)
    print("Creating pi_b dataset to calculate weights")
    generate_traj = False
    if generate_traj:
        pool = mp.Pool(30)
        res = pool.map(generate_trajectory, [policies[pi_b] for _ in range(400)])
        behavior_dataset_returns, behavior_dataset_trajectories, behavior_dataset_actions = zip(*res)
        pool.close()
        pool.join()
        if not os.path.exists('./saved_trajectories'):
            os.mkdir('./saved_trajectories')
        behavior_trajectories_all = {'returns':behavior_dataset_returns, 'trajectories':behavior_dataset_trajectories, 'actions':behavior_dataset_actions}
        pickle.dump(behavior_trajectories_all, open('./saved_trajectories/' + str(pi_b) + ".pkl", 'wb+'))
    behavior_trajectories_all = pickle.load(open('./saved_trajectories/' + str(pi_b) + ".pkl", 'rb'))
    
    # Calculate weights for all pairs in the calibration dataset
    b_o_returns = filtered_b_o_rs
    b_gen_returns = filtered_b_gen_rs
    weights = []

    behavior_policy = policies[pi_b]
    target_policy = policies[pi_e]
    print("Calculating Weights")
    for i in range (len(b_gen_returns)): # Should only be on the calibration dataset.
        b_o = filtered_b_o_ts[i] # Original env trajectory in match
        b_gen = filtered_b_gen_ts[i] # Generated trajectory in match
        b_o_r = b_o_returns[i]
        b_gen_r = b_gen_returns[i]
        matched_returns, matched_trajectories, matched_actions = find_similar_trajectories(behavior_trajectories_all, b_gen, b_gen_r)
        assert len(matched_returns) > 0, "no matched trajectories when calculating weights!"
        all_terms = []
        for t, a, r in zip(matched_trajectories, matched_actions, matched_returns):
            all_terms.append(calculate_ips_product(t, a, target_policy, behavior_policy))
        w_hat = np.mean(all_terms) # This is the expectation over the filtered trajectories
        weights += [w_hat]
    print("IPS weights, unnormalized, " + str(weights)) # All nans, because the IPS products are super low?

    # Normalize all the weights
    weights = np.asarray(weights)/np.sum(weights)

    new_weighted_errors = []
    for i in range(len(b_o_returns)):
        new_weighted_errors.append(weights[i] * (b_o_returns[i] - b_gen_returns[i])) # This should be the absolute value of the difference between returns
    print("weights ", weights)
    print("new_weighted_errors ", new_weighted_errors)
    print("b_o_returns ", b_o_returns)
    print("b_gen_returns ", b_gen_returns)


    true_target_rewards = []

#    TODO: double check that this is not symmetric?
    print("First Term Approximation: " + str(np.mean(first_term_target_rewards)))
    print("Interval: (" + str(np.mean(first_term_target_rewards) - np.quantile(new_weighted_errors, 1 - alpha)) + ", " + str((np.mean(first_term_target_rewards) + np.quantile(new_weighted_errors, 1-alpha))) + ")")

    for _, i in enumerate(tqdm.tqdm(range(n_tries))): # Calculating the actual value using monte carlo sampling
        target_reward, _, _ = generate_trajectory(target_policy)
        true_target_rewards.append(target_reward)   # This will be about 20. something

    print("True target reward: " + str(np.mean(true_target_rewards)))


    # preds = np.asarray(preds)
    # truths = np.asarray(truths)
    # print ("MAE:", np.mean(np.abs((preds - truths))))
    #
    # rank, _ = spearmanr(preds, truths)
    # print ("Rank:", rank)
    #
    # print("Regret:", (np.max(truths) - truths[np.argmax(preds)])/np.max(truths))
    



