import tensorflow as tf
import numpy as np
from collections import deque
import random
import time
import gym
from gym import wrappers
import cloudpickle as cp
from VLBM import *
import os
import tensorflow_probability as tfp
import multiprocessing as mp
import os
import d4rl
import json
from functools import partial
import pandas as pd
import argparse
import pickle
import tqdm
import dill
import collections
import concurrent.futures
from multiprocessing import Manager
from utils import generate_and_check_trajectory, LearnedEnv

mp.set_start_method("spawn")
slim = tf.contrib.slim
rnn = tf.contrib.rnn
tfd = tfp.distributions
np.random.seed(42)

# Walker2d in this file
parser = argparse.ArgumentParser()
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-lr", type=float, help="Set learning rate for training VLBM DEFAULT=0.0001", default=0.0001)
parser.add_argument("-decay_step", type=int, help="Set exponential decay step DEFAULT=1000", default=1000)
parser.add_argument("-decay_rate", type=float, help="Set exponential decay rate DEFAULT=0.997", default=0.997)
parser.add_argument("-max_iter", type=int, help="Set max number of training iterations DEFAULT=1000", default=1000)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
parser.add_argument("-gamma", type=float, help="Set discounting factor DEFAULT=0.995", default=0.995)
parser.add_argument("-batch_size", type=int, help="Set minibatch size DEFAULT=64", default=64)
parser.add_argument("-num_branch", type=int, help="Set number of branches for VLBM decoder DEFAULT=10", default=10)
parser.add_argument("-code_size", type=int, help="Set dimension of the latent space DEFAULT=16", default=16)
parser.add_argument("-beta", type=float, help="Set the constant C in the objective DEFAULT=1.0", default=1.)
parser.add_argument("-env", type=str, help="Choose environment from {halfcheetah-medium-expert-v2, halfcheetah-medium-v2}. Use the other script to train on Ant, Hopper, Walker2d. DEFAULT=halfcheetah-medium-expert-v2", default='halfcheetah-medium-expert-v2')
parser.add_argument("-val_interval", type=int, help="Validation interval DEFAULT=50", default=50)
# Below are some constants that would not be changed
parser.add_argument("-path", type=str, help="Path to checkpoint folder")
parser.add_argument("-repeat", type=int, help="Set action repeat. Since we are training on offline trajectories, so this is not needed (always set to 1)", default=1)
parser.add_argument("-max_episode_len", type=int, help="Maximum episode length, which is always 1000 for Gym-Mujoco environments", default=1000)
parser.add_argument("-buffer_size", type=int, help="Maximum buffer size. Set to 3000 to make sure it can accomodate all offline trajectories used for training", default=3000)


def sequence_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if k.find("metadata")==-1:
                data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
def evaluate(ope_eval, graph_ope_eval, sess_ope_eval, *args):
    
    # Validate and create checkpoints of VLBM during training
    
    (MAX_EPISODE_LEN, REPEAT, env_state_dim, env_action_dim, RANDOM_SEED, 
    	obs_mean, obs_std, rew_mean, rew_std, rl_params) = args
    
    with tf.io.gfile.GFile("./d4rl_policies.json", 'r') as f:
        policy_database = json.load(f)

        policy_metadatas = [i for i in policy_database if i['task.task_names'][0].find(rl_params['env_name'].split("-")[0]+"-")!=-1]
    
    truths = [np.loadtxt("./truth_discounted/" + p["policy_path"] + ".txt")[0] for p in policy_metadatas]
    
    pred = []

    learned_env = LearnedEnv(ope_eval)

    for _i in range(len(policy_metadatas)):
        with graph_ope_eval.as_default():
            ope_eval.saver.restore(sess_ope_eval, ope_eval.save_appendix)
        ep_rewards = []
        policy = D4RL_Policy(policy_metadatas[_i]['policy_path'])
        for i in range(5):

            terminal = 0

            s = learned_env.reset()
            s = s.reshape(env_state_dim)*obs_std + obs_mean
            ep_reward = 0

            for j in range(MAX_EPISODE_LEN):

                if j % REPEAT == 0:
                    a, _, _ = policy.act(np.reshape(s, (env_state_dim,)), np.zeros((env_action_dim,)))
                s2, r, terminal, info = learned_env.step(a)
                r = r*rew_std + rew_mean
                s2 = s2.reshape(env_state_dim)*obs_std + obs_mean

                ep_reward += r*(GAMMA**j)

                s = s2

                if terminal or j == MAX_EPISODE_LEN-1:
                    ep_rewards += [ep_reward]
                    break
                    
        pred += [np.mean(ep_rewards)]
    return np.mean(np.abs(np.asarray(truths)-np.asarray(pred)))

# def generate_trajectory(policy, env):
#     s = env.reset()
#     s = s.reshape(env_state_dim) * obs_std + obs_mean
#     ep_reward = 0
#     trajectory = [s]
#     trajectory_actions = []
#     terminal = 0
#     for j in range(MAX_EPISODE_LEN):
#         if j % REPEAT == 0:
#             a, _, _ = policy.act(np.reshape(s, (env_state_dim,)), np.zeros((env_action_dim,)))
#         trajectory_actions.append(a)
#         s2, r, terminal, info = env.step(a)
#         r = r * rew_std + rew_mean
#         s2 = s2.reshape(env_state_dim) * obs_std + obs_mean

#         ep_reward += r * (GAMMA ** j)

#         s = s2
#         trajectory.append(s)

#         if terminal or j == MAX_EPISODE_LEN - 1:
#             return ep_reward, trajectory, trajectory_actions
def calculate_ips_product(t_t, t_a, target_policy, behavior_policy):
    ips_vals = []
    for i in range(len(t_a)):
        ips_weight = np.log(target_policy.propensity_score(t_t[i], t_a[i])) - np.log(
            behavior_policy.propensity_score(t_t[i], t_a[i]))
        if np.isinf(ips_weight):  # when divide by zero happens
            continue
        else:
            ips_vals.append(ips_weight)
    return np.sum(ips_vals)

def calculate_policy_value(target_policy_path, behavior_policy_path, ope_model, n_tries=100):
    # Returns a list of trajectories that are calibrated for this particular behavior and target policy
    ope_path = args.path
    ope_saver = ope_model.saver
    with tf.Session(config=config, graph=graph_ope_models) as sess_ope_model:
        ope_saver.restore(sess_ope_model, os.path.join(ope_path, "ope_best.ckpt"))

    d4rl_qlearning = d4rl.qlearning_dataset(env)

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
    ENV = args.env
    rl_params = {
        'env_name': ENV,
    }
    original_env = gym.make(rl_params['env_name'])

    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    target_policy = D4RL_Policy(target_policy_path)
    behavior_policy = D4RL_Policy(behavior_policy_path)

    # Calculate the true value of the policy
    true_target_rewards = []
    for _, i in enumerate(tqdm.tqdm(range(n_tries))):
        target_reward, _, _ = generate_trajectory(target_policy, original_env)
        true_target_rewards.append(target_reward)

    # Generate a set of trajectories for the first term from target policy
    first_term_target_rewards = []
    for _, i in enumerate(tqdm.tqdm(range(n_tries))):  # TODO: Make this 100000
        ep_reward, _, _ = generate_trajectory(target_policy, learned_env)
        first_term_target_rewards.append(ep_reward)

    # Generate possible matching trajectories (this is without paralellization)
    # epsilon = 0.2
    # predicted_returns = []
    # actual_returns = []
    # # Or alternatively generate trajectories until you reach a total calibration dataset size
    # calibration_dataset_size = 100
    # attempts = 0
    # while len(predicted_returns) < calibration_dataset_size:
    #     b_r, b_t = generate_trajectory(behavior_policy, original_env)
    #     t_r, t_t = generate_trajectory(target_policy, learned_env)
    #     attempts += 1
    #     if (np.linalg.norm(t_t[0][:8] - b_t[0][:8]) < epsilon):
    #         if (np.linalg.norm(t_t[-1][:8] - b_t[-1][:8]) < epsilon):  # First and last state are the same
    #             predicted_returns.append(t_r)
    #             actual_returns.append(b_r)
    #     if attempts >= 4000:
    #         break

    # Parallelize the calibration dataset generation:
    epsilon = 0.2
    calibration_dataset_size = 100
    max_attempts = 4000

    parallelize = True
    if parallelize:
        pool = mp.Pool(processes=4)
        # Bundle policies and environments to pass to worker functions
        policies_and_envs = (target_policy, learned_env, behavior_policy, original_env)
        mp.reduction.ForkingPickler = dill.Pickler
        results = pool.map(cp.loads(cp.dumps(generate_and_check_trajectory)), [(policies_and_envs) for i in range(max_attempts)])
        pool.close()
        pool.join()
        pool.join()


        # Process valid results
        # # TODO: go through every saved trajectory?
        # (t_r, t_t, t_a), (b_r, b_t, b_a) = result
        # if len(t_t) > 0:
        #     predicted_returns.append(t_r)
        #     actual_returns.append(b_r)
        #     ips_weight = calculate_ips_product(t_t, t_a, target_policy, behavior_policy)
        #     predicted_ips_weights.append(ips_weight)
        import ipdb; ipdb.set_trace()
    else:
        print("no parallelization")
        pass
        # predicted_returns = []
        # actual_returns = []
        # predicted_ips_weights = []
        # for i in range(max_attempts):
        #     result = generate_and_check_trajectory(i)
        #     if result is not None:
        #         (t_r, t_t, t_a), (b_r, b_t, b_a) = result
        #         predicted_returns.append(t_r)
        #         actual_returns.append(b_r)
        #         predicted_ips_weights.append(calculate_ips_product(t_t, t_a))
        #     if len(predicted_returns) >= calibration_dataset_size:
        #         break

    print("Size of calibration dataset: " + str(len(predicted_returns))) # If this is around 50, we are maximally matching.

    # Option 2: nearest neighbor?
    IPS_weighting = True

    if len(predicted_returns) == 0:
        print("Cannot calculate quantiles")
        return (0, 0), np.mean(true_target_rewards)
    # Calculate quantiles over the calibration dataset rather than mean/sd
    alpha = 0.1 # 90% coverage

    if IPS_weighting:
        ips_weighted_errors = []
        for i in range(len(predicted_returns)):
            ips_weighted_errors.append(predicted_ips_weights[i] * (actual_returns[i] - predicted_returns[i]))
        import ipdb; ipdb.set_trace()
        return (np.mean(first_term_target_rewards) - np.quantile(ips_weighted_errors, 1 - alpha), (np.mean(first_term_target_rewards) - np.quantile(ips_weighted_errors, alpha))), np.mean(true_target_rewards)
    else:
        errors = []
        for i in range(len(predicted_returns)):
            errors.append(actual_returns[i] - predicted_returns[i])
        return (np.mean(first_term_target_rewards) - np.quantile(errors, 1 - alpha), (np.mean(first_term_target_rewards) - np.quantile(errors, alpha))), np.mean(true_target_rewards)


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
    MINIBATCH_SIZE_OPE = args.batch_size
    RANDOM_SEED = args.seed
    MAX_ITER = args.max_iter
    NUM_BRANCH = args.num_branch
    CODE_SIZE = args.code_size
    BETA = args.beta
    OPE_LR = args.lr
    OPE_DS = args.decay_step
    OPE_DR = args.decay_rate
    TRAIN = False

    MAX_EPISODE_LEN = args.max_episode_len
    REPEAT = args.repeat # Action repeat is not needed since we are training on offline trajectories. So it's always set to 1.
    BUFFER_SIZE_OPE = args.buffer_size
    BEST_MAE = 9999. # Used later for validation and checkpoint saving

    ENV = args.env

    assert "halfcheetah" in ENV, "This script only work for Halfcheetah which does not perform early termination of episodes. To train on Ant, Hopper, Walker2d, please use the other script."

    if "expert" in ENV:
        sequence_dataset = d4rl.sequence_dataset

    if not os.path.exists("./rl_stats"):
        os.mkdir("./rl_stats")
    if not os.path.exists("./saved_model"):
        os.mkdir("./saved_model")

    rl_params = {
        'env_name': ENV,
    }

    file_appendix = (
        "VLBM_" + rl_params['env_name'] + "_" + str(MAX_ITER)
        + "iter_"
        + str(OPE_LR) + "_"
        + str(OPE_DS) + "_"
        + str(OPE_DR) + "_"
        + str(CODE_SIZE) + "_"
        + str(BETA) + "_"
        + str(RANDOM_SEED)
    )

    graph_ope_models = tf.Graph()
    env = gym.make(rl_params['env_name'])
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    env_state_dim = env.observation_space.shape[0]
    env_action_dim = env.action_space.shape[0]
    env_action_bound = env.action_space.high
    env_state_bound = None
    if TRAIN:
        iters_already_passed = 0

        # To determine if there exist checkpoints associated with same hyper-parameters
        if os.path.exists("./rl_stats/" + file_appendix + ".txt"):
            stats_df = pd.read_csv("./rl_stats/" + file_appendix + ".txt", header=None, delimiter=" | ")
            iters_already_passed = len(stats_df.index.values)
            assert iters_already_passed < MAX_ITER, 'There already exist a model trained using same parameter, please delete its checkpoint and logs before starting a new round of training'

        graph_ope_models_eval = tf.Graph()
        with tf.Session(config=config, graph=graph_ope_models) as sess_ope_models:
            with tf.Session(config=config, graph=graph_ope_models_eval) as sess_ope_models_eval:

                d4rl_qlearning = d4rl.qlearning_dataset(env)

                obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)
                obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)

                rew_mean = d4rl_qlearning['rewards'].mean()
                rew_std = d4rl_qlearning['rewards'].std()

                with graph_ope_models.as_default():

                    ope_model = OPE_Model(
                        NUM_BRANCH, graph_ope_models, sess_ope_models, OPE_LR, OPE_DS, OPE_DR, CODE_SIZE,
                        env_state_dim, env_state_bound, env_action_dim, file_appendix,
                        BUFFER_SIZE_OPE, RANDOM_SEED, MINIBATCH_SIZE_OPE, MAX_EPISODE_LEN, BETA
                    )

                    ope_saver = ope_model.saver

                    sess_ope_models.run(tf.global_variables_initializer())

                    ope_model.replay_buffer.port_d4rl_data(
                        sequence_dataset(env),
                        obs_mean,
                        obs_std,
                        rew_mean,
                        rew_std,
                    )

                    # If exist checkpoints using same hyper-parameters, load it and train on top
                    if os.path.exists("./rl_stats/"+file_appendix+".txt"):
                        ope_model.saver.restore(
                            sess_ope_models,
                            os.path.join(
                                "./saved_model/",
                                file_appendix,
                                "ope.ckpt"
                            )
                        )

                        for _k in range(iters_already_passed*MAX_EPISODE_LEN):

                            sess_ope_models.run(ope_model.global_step_increment)

                with graph_ope_models_eval.as_default():

                    # Create a copy of VLBM to use for validation -- ensuring batch norm weights won't be updated
                    ope_model_eval = OPE_Model(
                        NUM_BRANCH, graph_ope_models_eval, sess_ope_models_eval, OPE_LR, OPE_DS, OPE_DR, CODE_SIZE,
                        env_state_dim, env_state_bound, env_action_dim, file_appendix,
                        BUFFER_SIZE_OPE, RANDOM_SEED, MINIBATCH_SIZE_OPE, MAX_EPISODE_LEN,
                        BETA, is_training=False
                    )


                actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env_action_dim))


                for i in range(iters_already_passed, MAX_ITER):

                    ep_elbo = []

                    if ope_model.replay_buffer.size > MINIBATCH_SIZE_OPE * 4:


                        batch = ope_model.replay_buffer.sample_batch(MINIBATCH_SIZE_OPE)

                        ope_model.train(batch)
                        ep_elbo += [np.mean([ope_model.elbo_evaluated for k in range(NUM_BRANCH)])]

                        # If the objective becomes to NAN, then it is likely that inappropriate beta is chosen.
                        # Need to restart training by setting beta to other values.
                        if np.isnan(ep_elbo[-1]):
                            break

                        if (i+1) % args.val_interval == 0 and ope_model.replay_buffer.size > MINIBATCH_SIZE_OPE * 4:

                            # Validate VLBM during training
                            mae = evaluate(
                                ope_model_eval,
                                graph_ope_models_eval,
                                sess_ope_models_eval,
                                MAX_EPISODE_LEN,
                                REPEAT,
                                env_state_dim,
                                env_action_dim,
                                RANDOM_SEED,
                                obs_mean,
                                obs_std,
                                rew_mean,
                                rew_std,
                                rl_params
                            )

                            # Save model checkpoints
                            if mae < BEST_MAE:
                                ope_model.saver.save(
                                    ope_model.sess,
                                    ope_model.save_appendix.replace("ope.ckpt", "ope_best.ckpt")
                                )
                                BEST_MAE = mae


                    with open("./rl_stats/"+file_appendix+".txt", "a") as myfile:
                        myfile.write(
                            '| Episode: {:d}  | ELBO: {:.4f} | \n'
                            .format(
                                i,
                                np.mean(ep_elbo),
                            )
                        )


                    print(
                        '| Episode: {:d}  | ELBO: {:.4f} | \n'
                        .format(
                            i,
                            np.mean(ep_elbo),
                        )
                    )
    else:
        with tf.io.gfile.GFile("./d4rl_policies.json", 'r') as f:
            policy_database = json.load(f)

        d4rl_qlearning = d4rl.qlearning_dataset(env)
        ope_path = args.path
        obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)
        obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)

        rew_mean = d4rl_qlearning['rewards'].mean()
        rew_std = d4rl_qlearning['rewards'].std()
        with graph_ope_models.as_default():
            with tf.Session(config=config, graph=graph_ope_models) as sess_ope_models:
                ope_model = OPE_Model(
                    NUM_BRANCH, graph_ope_models, sess_ope_models, OPE_LR, OPE_DS, OPE_DR, CODE_SIZE,
                    env_state_dim, env_state_bound, env_action_dim, file_appendix,
                    BUFFER_SIZE_OPE, RANDOM_SEED, MINIBATCH_SIZE_OPE, MAX_EPISODE_LEN, BETA
                )

                ope_saver = ope_model.saver

                sess_ope_models.run(tf.global_variables_initializer())

                ope_model.replay_buffer.port_d4rl_data(
                    sequence_dataset(env),
                    obs_mean,
                    obs_std,
                    rew_mean,
                    rew_std,
                )

                # If exist checkpoints using same hyper-parameters, load it and train on top
                if os.path.exists("./rl_stats/" + file_appendix + ".txt"):
                    ope_model.saver.restore(
                        sess_ope_models,
                        os.path.join(
                            "./saved_model/",
                            file_appendix,
                            "ope.ckpt"
                        )
                    )
                # This should calculate the calibration dataset, for a given target and behavior policy
                policy_metadatas = [i for i in policy_database if
                                    i['task.task_names'][0].find(rl_params['env_name'].split("-")[0] + "-") != -1]
                # i_abridged = [0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 6]
                # j_abridged = [2, 4, 0, 2, 7, 0, 0, 4, 8, 5, 0, 8, 0]
                # for kk in range(len(i_abridged)):
                #     i = i_abridged[kk]
                #     j = j_abridged[kk]
                #     target_policy_path = policy_metadatas[i]['policy_path']
                #     behavior_policy_path = policy_metadatas[j]['policy_path']
                #     (lower_quantile, upper_quantile), policy_value = calculate_policy_value(target_policy_path, behavior_policy_path, ope_model, 100)
                #     print("i: " + str(i) + " j: " + str(j) + " policy value: (" + str(lower_quantile) + "," + str(upper_quantile) + ") gt policy value: " + str(policy_value))

                # Experiment 3: Try to assess the value of the behavior policy
                target_policy_path = policy_metadatas[0]['policy_path']
                behavior_policy_path = policy_metadatas[4]['policy_path'] # using different policy paths now.
                (lower_quantile, upper_quantile), policy_value = calculate_policy_value(target_policy_path, behavior_policy_path, ope_model, 5)
                print("policy value: (" + str(lower_quantile) + "," + str(upper_quantile) + ") gt policy value: " + str(policy_value))

                # TODO: parallelization
                # TODO: use walker-2d
                import ipdb; ipdb.set_trace()




