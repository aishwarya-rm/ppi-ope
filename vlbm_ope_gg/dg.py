import tensorflow as tf
import numpy as np
import gym
import tqdm
# from VLBM_for_envs_with_early_termination import *
from VLBM import *
import os
import tensorflow_probability as tfp
import multiprocessing as mp
import d4rl
import json
import argparse
import numpy as np
from scipy.stats import norm
import pickle
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress all warnings other than critical ones.

slim = tf.contrib.slim
rnn = tf.contrib.rnn
tfd = tfp.distributions

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task-id', '-i',
    type=int,
    help='SLURM array index'
)
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
parser.add_argument("-gamma", type=float, help="Set discounting factor DEFAULT=0.995", default=0.995)
parser.add_argument("-code_size", type=int, help="Set dimension of the latent space DEFAULT=16", default=16)
parser.add_argument("-env", type=str,
                    help="Choose environment from <ant/hopper/walker2d>-<medium/medium-expert>-v2. Use the other script to evaluate on Halfcheetah. DEFAULT=halfcheetah-medium-expert-v2",
                    default='halfcheetah-medium-expert-v2')
parser.add_argument("-max_episodes", type=int, help="Maximum number of episodes run for evaluation", default=50)
parser.add_argument("-path", type=str, help="Path to checkpoint folder", default="./saved_model/VLBM_halfcheetah-medium-expert-v2/")
# Below are some constants that would not be changed
parser.add_argument("-repeat", type=int,
                    help="Set action repeat. Since we are training on offline trajectories, so this is not needed (always set to 1)",
                    default=1)
parser.add_argument("-max_episode_len", type=int,
                    help="Maximum episode length, which is always 1000 for Gym-Mujoco environments", default=1000)

env = gym.make("halfcheetah-medium-expert-v2")

env_state_dim = env.observation_space.shape[0]
env_action_dim = env.action_space.shape[0]
env_action_bound = env.action_space.high
env_state_bound = None
d4rl_qlearning = d4rl.qlearning_dataset(env)

obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)
obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)
rew_mean = d4rl_qlearning['rewards'].mean()
rew_std = d4rl_qlearning['rewards'].std()

# _graph   = None
# _learned_env = None

# def _init_worker(ope_path, config):
#     global _graph, _sess, _learned_env
#     # 1) build graph & session once
#     _graph = tf.Graph()
#     with tf.Session(config=config, graph=_graph) as sess_ope_models:
#         with _graph.as_default():
#             # 2) instantiate & restore your OPE_Model
#             ope_model = OPE_Model(
#                 num_branch, graph_ope_models, sess_ope_models, .001, 1000, .997, CODE_SIZE,
#                 env_state_dim, env_state_bound, env_action_dim, "",
#                 4200, RANDOM_SEED, 64, MAX_EPISODE_LEN, 1.,
#                 is_training=False
#             )
#             saver = ope_model.saver
#             saver.restore(sess_ope_models, os.path.join(ope_path, "ope_best.ckpt"))
#             # 3) wrap it in your LearnedEnv
#             _learned_env = LearnedEnv(ope_model)

def rollout_original_env(policy, starting_state=None, scale=None): # From original environment

    if starting_state is not None:
        nq = env.model.nq  # Number of position variables (qpos)
        nv = env.model.nv  # Number of velocity variables (qvel)
        _ = env.reset()
        if scale is not None:
            qpos = np.concatenate([[0.0], starting_state[:nq - 1]]) + scale * np.concatenate([[0.0], obs_mean[:nq-1]])
            qvel = starting_state[nq - 1:] + scale * obs_mean[nq-1:]
        else:
            qpos = np.concatenate([[0.0], starting_state[:nq - 1]])
            qvel = starting_state[nq - 1:]

        env.set_state(qpos, qvel)
        s = env.unwrapped._get_obs()
    else:
        s = env.reset()
    s = s.reshape(env_state_dim)
    ep_reward = 0
    trajectory_states = [s]
    trajectory_actions = []
    trajectory_rewards = []
    terminal = 0
    for j in tqdm.tqdm(range(MAX_EPISODE_LEN)):
        if j % REPEAT == 0:
            a, _, _ = policy.act(np.reshape(s, (env_state_dim,)), np.zeros((env_action_dim,))) # Avoiding randomness
        trajectory_actions.append(a)
        s2, r, terminal, info = env.step(a)
        # r = r * rew_std + rew_mean
        s2 = s2.reshape(env_state_dim)

        ep_reward += r * (GAMMA ** j)

        s = s2
        trajectory_states.append(s)
        trajectory_rewards.append(r)

        if terminal or j == MAX_EPISODE_LEN - 1:
            return ep_reward, trajectory_states, trajectory_actions, trajectory_rewards


def rollout_learned_env(policy, starting_state=None, scale=None):
    '''
    policy_path: D4RL policy that you want the generated trajectory's actions to be sampled from.
    starting_state: If you want the generated trajectory to start in a particular state, specify it.
    scale: Adds noise to the generated trajectory. Can also be set to None.
    Returns ep_reward, trajectory_states, trajectory_actions, trajectory_rewards for 1 trajectory.
    '''
    graph_ope_models = tf.Graph()
    with tf.Session(config=config, graph=graph_ope_models) as sess_ope_models:
        with graph_ope_models.as_default():
            ope_model = OPE_Model(
                num_branch, graph_ope_models, sess_ope_models, .001, 1000, .997, CODE_SIZE,
                env_state_dim, env_state_bound, env_action_dim, "",
                4200, RANDOM_SEED, 64, MAX_EPISODE_LEN, 1.,
                is_training=False
            )
            ope_saver = ope_model.saver
            ope_saver.restore(sess_ope_models, os.path.join(ope_path, "ope_best.ckpt"))

        _learned_env = LearnedEnv(ope_model)

        ep_rewards = []

        terminal = 0
        if starting_state is None:
            s = _learned_env.reset()
            s = s.reshape(env_state_dim) * obs_std + obs_mean
        else:
            s = _learned_env.reset(starting_state)
            if scale is not None:
                s = s.reshape(env_state_dim) + scale * obs_mean  # Add a little tiny bit of noise
            else:
                s = s.reshape(env_state_dim)

        ep_reward = 0
        trajectory_states = [s]
        trajectory_actions = []
        trajectory_rewards = []

        for j in tqdm.tqdm(range(MAX_EPISODE_LEN)):
            if j % REPEAT == 0:
                a, _, _ = policy.act(np.reshape(s, (env_state_dim,)), np.zeros((env_action_dim,)))
            trajectory_actions.append(a)  # test
            s2, r, terminal, info = _learned_env.step(a)
            r = r * rew_std + rew_mean
            s2 = s2.reshape(env_state_dim) * obs_std + obs_mean

            ep_reward += r * (GAMMA ** j)

            s = s2
            trajectory_states.append(s)  # test
            trajectory_rewards.append(r)
            if terminal or j == MAX_EPISODE_LEN - 1:
                ep_rewards += [ep_reward]

                return ep_reward, trajectory_states, trajectory_actions, trajectory_rewards

class LearnedEnv(object):
    def __init__(self, model):
        self.model = model

    def reset(self, starting_state=None):
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
    REPEAT = args.repeat
    CODE_SIZE = args.code_size
    task_id = args.task_id

    n = 1000
    m = 10
    pi_e = 10
    pi_b = 9
    num_processes = np.min([n, 20])

    ENV = args.env
    ope_path = args.path
    rl_params = {
        'env_name': ENV,
    }
    with tf.io.gfile.GFile("./d4rl_policies.json", 'r') as f:
        policy_database = json.load(f)
    policy_metadatas = [i for i in policy_database if
                        i['task.task_names'][0].find(rl_params['env_name'].split("-")[0] + "-") != -1]

    target_policy = D4RL_Policy(policy_metadatas[pi_e]['policy_path'])
    behavior_policy = D4RL_Policy(policy_metadatas[pi_b]['policy_path'])

    # # true, pi_b
    # pool = mp.Pool(num_processes)
    # res = pool.map(rollout_original_env, [behavior_policy for _ in range(n)])
    # b_ep_rewards, b_states, b_actions, b_rewards = zip(*res)
    # pool.close()
    # pool.join()
    # behavior_trajectories_o = {'ep_returns': b_ep_rewards, 'states': b_states, 'actions': b_actions, 'rewards':b_rewards}
    # pickle.dump(behavior_trajectories_o, open("./saved_trajectories/true_b_1.pkl", 'wb'))

    # # true, pi_e
    # pool = mp.Pool(num_processes)
    # res = pool.map(rollout_original_env, [target_policy for _ in range(n)])
    # t_ep_rewards, t_states, t_actions, t_rewards = zip(*res)
    # pool.close()
    # pool.join()
    # target_trajectories_o = {'ep_returns': t_ep_rewards, 'states': t_states, 'actions': t_actions,
    #                            'rewards': t_rewards}
    # pickle.dump(target_trajectories_o, open("./saved_trajectories/true_e.pkl", 'wb'))

    # # true, pi_e, s0
    # # State conditioned generations
    # behavior_trajectories_o = pickle.load(open("./saved_trajectories/true_b.pkl", 'rb'))
    # b_states = behavior_trajectories_o['states']  # This is just an arbitrary first state
    # s_0 = b_states[0][0]

    # pool = mp.Pool(num_processes)
    # scales = [np.random.normal(loc=0, scale=0.2) for _ in range(n)]
    # res = pool.starmap(rollout_original_env, [(target_policy, s_0, None) for ll in range(n)])
    # gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
    # target_trajs_s0 = {'ep_returns': gen_returns, 'states': gen_states, 'actions': gen_actions,
    #                        'rewards': gen_rewards}
    # pickle.dump(target_trajs_s0, open("./saved_trajectories/true_e_s0.pkl", 'wb'))
    # pool.close()
    # pool.join()

    graph_ope_models = tf.Graph()

    with graph_ope_models.as_default():
        tf.train.import_meta_graph(os.path.join(ope_path, "ope_best.ckpt.meta"))
        num_branch = np.asarray(list((set([int(v.name.split("/")[0].split("_")[-1]) for v in tf.trainable_variables() if
                                           v.name.find("Decoder_zt1_") != -1])))).max() + 1
    
    # # for each true,pi_b, generate m learned,pi_e
    # behavior_trajectories_o = pickle.load(open("./saved_trajectories/true_b_2.pkl", 'rb'))

    # b_ep_rewards = behavior_trajectories_o['ep_returns']
    # b_states = behavior_trajectories_o['states']
    # b_o_as = behavior_trajectories_o['actions']
    # b_o_tr = behavior_trajectories_o['rewards']
    # behavior_trajectories_matching_dr_ppi = {}
    # for i, b_traj_states in enumerate(b_states):
    #     s_0 = b_traj_states[0]  # This is the first state in the trajectory
    #     pool = mp.Pool(num_processes)
    #     scales = [np.random.normal(loc=0, scale=0.2) for _ in range(m)] # To add some noise
    #     res = pool.starmap(rollout_learned_env, [(target_policy, s_0, None) for _ in range(m)])
    #     gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
    #     pool.close()
    #     pool.join()

    #     behavior_trajectories_matching_dr_ppi[i] = {'ep_returns': [], 'states': [], 'actions': [], 'rewards':[]}
    #     for j in range(len(gen_returns)): # All matching trajectories
    #         behavior_trajectories_matching_dr_ppi[i]['ep_returns'].append(gen_returns[j])
    #         behavior_trajectories_matching_dr_ppi[i]['states'].append(gen_states[j])
    #         behavior_trajectories_matching_dr_ppi[i]['actions'].append(gen_actions[j])
    #         behavior_trajectories_matching_dr_ppi[i]['rewards'].append(gen_rewards[j])

    #     pickle.dump(behavior_trajectories_matching_dr_ppi, open("./saved_trajectories/matching_dr_ppi2_{}.pkl".format(task_id), 'wb'))

    
    # for each true,pi_b, generate m learned,pi_b
    behavior_trajectories_o = pickle.load(open("./saved_trajectories/true_b_2.pkl", 'rb'))

    b_ep_rewards = behavior_trajectories_o['ep_returns']
    b_states = behavior_trajectories_o['states']
    b_o_as = behavior_trajectories_o['actions']
    b_o_tr = behavior_trajectories_o['rewards']
    behavior_trajectories_matching_dr_ppi = {}
    for i, b_traj_states in enumerate(b_states):
        s_0 = b_traj_states[0]  # This is the first state in the trajectory
        pool = mp.Pool(num_processes)
        scales = [np.random.normal(loc=0, scale=0.2) for _ in range(m)] # To add some noise
        res = pool.starmap(rollout_learned_env, [(behavior_policy, s_0, None) for _ in range(m)])
        gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
        pool.close()
        pool.join()

        behavior_trajectories_matching_dr_ppi[i] = {'ep_returns': [], 'states': [], 'actions': [], 'rewards':[]}
        for j in range(len(gen_returns)): # All matching trajectories
            behavior_trajectories_matching_dr_ppi[i]['ep_returns'].append(gen_returns[j])
            behavior_trajectories_matching_dr_ppi[i]['states'].append(gen_states[j])
            behavior_trajectories_matching_dr_ppi[i]['actions'].append(gen_actions[j])
            behavior_trajectories_matching_dr_ppi[i]['rewards'].append(gen_rewards[j])

        pickle.dump(behavior_trajectories_matching_dr_ppi, open("./saved_trajectories/matching_cp_ppi2_{}.pkl".format(task_id), 'wb'))
    
    # # learned, pi_e
    # pool = mp.Pool(num_processes)
    # res = pool.map(rollout_learned_env, [target_policy for _ in range(n)])
    # t_ep_rewards, t_states, t_actions, t_rewards = zip(*res)
    # first_term_trajs = {'ep_returns': t_ep_rewards, 'states': t_states, 'actions': t_actions, 'rewards':t_rewards}
    # pickle.dump(first_term_trajs, open("./saved_trajectories/first_term_dr_ppi.pkl", 'wb'))
    # pool.close()
    # pool.join()

    # # learned, pi_e, s0
    # pool = mp.Pool(num_processes)
    # scales = [np.random.normal(loc=0, scale=0.2) for _ in range(n)]
    # res = pool.starmap(rollout_learned_env, [(target_policy, s_0, scales[ll]) for ll in range(n)])
    # gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
    # first_term_trajs_s0 = {'ep_returns': gen_returns, 'states': gen_states, 'actions': gen_actions, 'rewards': gen_rewards}
    # pickle.dump(first_term_trajs_s0, open("./saved_trajectories/first_term_cp_ppi.pkl", 'wb'))
    # pool.close()
    # pool.join()


    # # learned, pi_b
    # print("*****Generating Trajectories: Behavior Policy, Learned Environment******")
    # pool = mp.Pool(num_processes)
    # scales = [np.random.normal(loc=0, scale=0.2) for _ in range(n)]
    # res = pool.starmap(rollout_learned_env, [(behavior_policy, None, scales[ll]) for ll in range(n)])
    # gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
    # target_trajs_s0 = {'ep_returns': gen_returns, 'states': gen_states, 'actions': gen_actions,
    #                        'rewards': gen_rewards}
    # pickle.dump(target_trajs_s0, open("./saved_trajectories/augment_b.pkl", 'wb'))
    # pool.close()
    # pool.join()


