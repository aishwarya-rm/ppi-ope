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
import argparse
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress all warnings other than critical ones.

slim = tf.contrib.slim
rnn = tf.contrib.rnn
tfd = tfp.distributions

parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--task-id', '-i',
#     type=int,
#     help='SLURM array index'
# )
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
parser.add_argument("-gamma", type=float, help="Set discounting factor DEFAULT=0.995", default=0.995)
parser.add_argument("-code_size", type=int, help="Set dimension of the latent space DEFAULT=16", default=16)
parser.add_argument("-env", type=str,
                    help="Choose environment from <ant/hopper/walker2d>-<medium/medium-expert>-v2. Use the other script to evaluate on Halfcheetah. DEFAULT=halfcheetah-medium-expert-v2",
                    default='walker2d-medium-expert-v2')
parser.add_argument("-max_episodes", type=int, help="Maximum number of episodes run for evaluation", default=50)
parser.add_argument("-path", type=str, help="Path to checkpoint folder", default='./saved_model/VLBM_mimic-iv_1000iter_0.001_1000_0.997_16_1_2599/')
# Below are some constants that would not be changed
parser.add_argument("-repeat", type=int,
                    help="Set action repeat. Since we are training on offline trajectories, so this is not needed (always set to 1)",
                    default=1)
parser.add_argument("-max_episode_len", type=int,
                    help="Maximum episode length, which is always 1000 for Gym-Mujoco environments", default=100)

env_state_dim = 20
env_action_dim = 5
env_action_bound = None
env_state_bound = None

def rollout_learned_env(policy, starting_state=None):
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

        if starting_state is None:
            s = _learned_env.reset()
            s = s.reshape(env_state_dim) * obs_std + obs_mean
        else:
            s = _learned_env.reset(starting_state)
            s = s.reshape(env_state_dim)
        ep_reward = 0
        trajectory_states = [s]
        trajectory_actions = []
        trajectory_rewards = []
        terminal = 0
        for j in range(MAX_EPISODE_LEN):
            if j % REPEAT == 0:
                a  = predict_action(policy, s, num_classes=5)
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

def predict_action(policy, state, num_classes=5):
    with torch.no_grad():
        logits = policy(torch.tensor(state, dtype=torch.float32))
        predicted_class = torch.argmax(logits, dim=-1).item()
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[predicted_class] = 1.0
        return one_hot
# Learn an NN to predict the action given the state from the behavior dataset (behavior cloning)
# Map discrete actions to class indices
ACTION_TO_IDX = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


class DiscretePolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions=5, hidden_sizes=[256, 256]):
        super(DiscretePolicyNetwork, self).__init__()
        layers = []
        input_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, num_actions))  # logits over 5 actions
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # raw logits

def learn_policies(behavior_dataset, target_dataset):
    # Define the classification network

    def train_discrete_policy(states, one_hot_actions, state_dim, epochs=40, batch_size=64, lr=1e-3):
        # Convert one-hot actions to class indices
        action_indices = np.argmax(one_hot_actions, axis=1)

        dataset = TensorDataset(
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(action_indices, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        policy = DiscretePolicyNetwork(state_dim)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_action_indices in loader:
                logits = policy(batch_states)
                loss = loss_fn(logits, batch_action_indices)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

        return policy

    behavior_policy = train_discrete_policy(behavior_dataset['observations'],
                                           behavior_dataset['actions'],
                                           state_dim=behavior_dataset['observations'].shape[1])
    target_policy = train_discrete_policy(target_dataset['observations'], target_dataset['actions'], state_dim=target_dataset['observations'].shape[1])

    return behavior_policy, target_policy

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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # Now set this session as default
    tf.keras.backend.set_session(sess)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    GAMMA = args.gamma
    RANDOM_SEED = args.seed
    MAX_EPISODE_LEN = args.max_episode_len
    REPEAT = args.repeat
    CODE_SIZE = args.code_size

    n = 100
    m = 10
    pi_e = 10
    pi_b = 9
    num_processes = np.min([m, 20])

    ENV = args.env
    ope_path = args.path

    # Learn the policies
    behavior_dataset = pickle.load(open('mimic_iv_behavior_trajectories.pkl', 'rb'))
    target_dataset = pickle.load(open('mimic_iv_target_trajectories.pkl', 'rb'))
    obs_mean = behavior_dataset['observations'].mean(0).astype(np.float32)
    obs_std = behavior_dataset['observations'].std(0).astype(np.float32)
    rew_mean = behavior_dataset['rewards'].mean()
    rew_std = behavior_dataset['rewards'].std()
    behavior_policy, target_policy = learn_policies(behavior_dataset, target_dataset)


    graph_ope_models = tf.Graph()

    with graph_ope_models.as_default():
        tf.train.import_meta_graph(os.path.join(ope_path, "ope_best.ckpt.meta"))
        num_branch = np.asarray(list((set([int(v.name.split("/")[0].split("_")[-1]) for v in tf.trainable_variables() if
                                           v.name.find("Decoder_zt1_") != -1])))).max() + 1

    # for each true,pi_b, generate m learned,pi_e
    # We have a lot of trajectories here, for about 230 behavior trajectories.
    # behavior_trajectories_o = pickle.load(open("mimic_iv_trajectories_behavior.pkl", 'rb'))
    # print("<<<For each True pi_b, generate m learned Pi_e that start in the same state>>\n\n")
    # b_ep_rewards = behavior_trajectories_o['reward_sum']
    # b_states = behavior_trajectories_o['states']
    # b_o_as = behavior_trajectories_o['actions']
    # b_o_tr = behavior_trajectories_o['rewards']
    # behavior_trajectories_matching_dr_ppi = {}
    # for i, b_traj_states in enumerate(b_states):
    #     s_0 = b_traj_states[0]  # This is the first state in the trajectory
    #     # pool = mp.Pool(num_processes)
    #     # res = pool.starmap(rollout_learned_env, [(target_policy, s_0) for _ in range(m)])
    #     # gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
    #     # pool.close()
    #     # pool.join()
    #     gen_returns = []
    #     gen_states = []
    #     gen_actions = []
    #     gen_rewards = []
    #     # Option to do this without parallelization:
    #     for j in range(m):
    #         ret, states, actions, rewards = rollout_learned_env(target_policy, s_0)
    #         gen_returns.append(ret)
    #         gen_states.append(states)
    #         gen_actions.append(actions)
    #         gen_rewards.append(rewards)
    #
    #     behavior_trajectories_matching_dr_ppi[i] = {'ep_returns': [], 'states': [], 'actions': [], 'rewards':[]}
    #     for j in range(len(gen_returns)): # All matching trajectories
    #         behavior_trajectories_matching_dr_ppi[i]['ep_returns'].append(gen_returns[j])
    #         behavior_trajectories_matching_dr_ppi[i]['states'].append(gen_states[j])
    #         behavior_trajectories_matching_dr_ppi[i]['actions'].append(gen_actions[j])
    #         behavior_trajectories_matching_dr_ppi[i]['rewards'].append(gen_rewards[j])
    #     print("Dumping " + str(i))
    #     pickle.dump(behavior_trajectories_matching_dr_ppi, open("./saved_trajectories/mimic-iv/matching_dr_ppi_mimiciv.pkl", 'wb'))

    # for each true,pi_b, generate m learned,pi_b
    # print("<<<For each True pi_b, generate m learned Pi_b that start in the same state>>\n\n")
    # behavior_trajectories_o = pickle.load(open("mimic_iv_trajectories_behavior.pkl", 'rb'))
    # b_ep_rewards = behavior_trajectories_o['reward_sum']
    # b_states = behavior_trajectories_o['states']
    # b_o_as = behavior_trajectories_o['actions']
    # b_o_tr = behavior_trajectories_o['rewards']
    # behavior_trajectories_matching_dr_ppi = {}
    # for i, b_traj_states in enumerate(tqdm.tqdm(b_states[:100])): # TODO: right now, we are just doing this for 100 trajectories
    #     s_0 = b_traj_states[0]  # This is the first state in the trajectory
    #     # pool = mp.Pool(num_processes)
    #     # res = pool.starmap(rollout_learned_env, [(behavior_policy, s_0) for _ in range(m)])
    #     # gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
    #     # pool.close()
    #     # pool.join()
    #     gen_returns = []
    #     gen_states = []
    #     gen_actions = []
    #     gen_rewards = []
    #     # Option to do this without parallelization:
    #     for j in range(m):
    #         ret, states, actions, rewards = rollout_learned_env(behavior_policy, s_0)
    #         gen_returns.append(ret)
    #         gen_states.append(states)
    #         gen_actions.append(actions)
    #         gen_rewards.append(rewards)
    #
    #     behavior_trajectories_matching_dr_ppi[i] = {'ep_returns': [], 'states': [], 'actions': [], 'rewards': []}
    #     for j in range(len(gen_returns)):  # All matching trajectories
    #         behavior_trajectories_matching_dr_ppi[i]['ep_returns'].append(gen_returns[j])
    #         behavior_trajectories_matching_dr_ppi[i]['states'].append(gen_states[j])
    #         behavior_trajectories_matching_dr_ppi[i]['actions'].append(gen_actions[j])
    #         behavior_trajectories_matching_dr_ppi[i]['rewards'].append(gen_rewards[j])
    #
    #     pickle.dump(behavior_trajectories_matching_dr_ppi,
    #                 open("./saved_trajectories/mimic-iv/matching_cp_ppi_mimiciv.pkl", 'wb'))

    # # learned, pi_e
    # print("<<<Generate a bunch of trajectories from pi_e>>\n\n", flush=True)
    # pool = mp.Pool(num_processes)
    # res = pool.map(rollout_learned_env, [target_policy for _ in range(n)])
    # t_ep_rewards, t_states, t_actions, t_rewards = zip(*res)
    t_ep_rewards = []
    t_states = []
    t_actions = []
    t_rewards = []
    # Option to do this without parallelization:
    for _, j in enumerate(tqdm.tqdm(range(200))):
        ret, states, actions, rewards = rollout_learned_env(target_policy)
        t_ep_rewards.append(ret)
        t_states.append(states)
        t_actions.append(actions)
        t_rewards.append(rewards)
        first_term_trajs = {'ep_returns': t_ep_rewards, 'states': t_states, 'actions': t_actions, 'rewards':t_rewards}
        pickle.dump(first_term_trajs, open(f"./saved_trajectories/mimic-iv/first_term_dr_ppi_mimiciv_2.pkl", 'wb'))
    # pool.close()
    # pool.join()
    #
    # # # learned, pi_e, s0
    # print("<<<Generate a bunch of trajectories from pi_e that start in the same initial state>>\n\n", flush=True)
    # # pool = mp.Pool(num_processes)
    # s_0 = b_states[0][0]  # This is the first state in the trajectory
    # # res = pool.starmap(rollout_learned_env, [(target_policy, s_0) for _ in range(n)])
    # # gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
    # gen_returns = []
    # gen_states = []
    # gen_actions = []
    # gen_rewards = []
    # # Option to do this without parallelization:
    # for _, j in enumerate(tqdm.tqdm(range(n))):
    #     ret, states, actions, rewards = rollout_learned_env(target_policy, s_0)
    #     gen_returns.append(ret)
    #     gen_states.append(states)
    #     gen_actions.append(actions)
    #     gen_rewards.append(rewards)
    # first_term_trajs_s0 = {'ep_returns': gen_returns, 'states': gen_states, 'actions': gen_actions, 'rewards': gen_rewards}
    # pickle.dump(first_term_trajs_s0, open(f"./saved_trajectories/mimic-iv/first_term_cp_ppi_mimiciv.pkl", 'wb'))
    # # pool.close()
    # # pool.join()
    #
    # # # learned, pi_b
    # print("<<<Generate a bunch of trajectories from pi_b using learned environment>>\n\n", flush=True)
    # # pool = mp.Pool(num_processes)
    # # res = pool.starmap(rollout_learned_env, [(behavior_policy, None) for _ in range(n)])
    # # gen_returns, gen_states, gen_actions, gen_rewards = zip(*res)
    # gen_returns = []
    # gen_states = []
    # gen_actions = []
    # gen_rewards = []
    # # Option to do this without parallelization:
    # for _, j in enumerate(tqdm.tqdm(range(n))):
    #     ret, states, actions, rewards = rollout_learned_env(behavior_policy)
    #     gen_returns.append(ret)
    #     gen_states.append(states)
    #     gen_actions.append(actions)
    #     gen_rewards.append(rewards)
    # target_trajs_s0 = {'ep_returns': gen_returns, 'states': gen_states, 'actions': gen_actions,
    #                        'rewards': gen_rewards}
    # pickle.dump(target_trajs_s0, open(f"./saved_trajectories/mimic-iv/augment_b_mimiciv.pkl", 'wb'))
    # pool.close()
    # pool.join()

