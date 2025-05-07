import tensorflow as tf
import numpy as np
import tqdm
from VLBM import *
import os
import tensorflow_probability as tfp
import multiprocessing as mp
import os
import json
import torch
from train_discriminator import TrajectoryClassifier
import argparse
slim = tf.contrib.slim
rnn = tf.contrib.rnn
tfd = tfp.distributions
from scipy.stats import norm
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-env", type=str,
                    help="Choose environment from <ant/hopper/walker2d>-<medium/medium-expert>-v2. Use the other script to evaluate on Halfcheetah. DEFAULT=halfcheetah-medium-expert-v2",
                    default='halfcheetah-medium-expert-v2')
def weighted_quantile(values, alpha, sample_weight=None):
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

    if sample_weight is None:
        sample_weight = np.ones_like(values)
    sample_weight = np.array(sample_weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    low_level = 0
    high_level = 0
    low_q = 0
    high_q = 0
    for i in range(len(values)):
        low_level += sample_weight[i]
        if low_level >= alpha / 2:
            low_q = values[i - 1]
            break
    for i in range(len(values)):
        high_level += sample_weight[i]
        if high_level >= 1 - alpha / 2:
            high_q = values[i - 1]
            break
    return (low_q, high_q)

def calculate_ips_product(t_t, t_a, t_r, target_policy, behavior_policy, wis=False, pdis=False, clipping=False):
    # T_t is the states
    if wis: # This works sometimes but produces a biased estimate
        log_probs_target = []
        log_probs_behavior = []
        for i in range(len(t_a)): # All of the actions in the trajectory
            rho_e = target_policy.log_prob(t_t[i], t_a[i])
            rho_b = behavior_policy.log_prob(t_t[i], t_a[i])
            log_probs_target.append(rho_e)
            log_probs_behavior.append(rho_b)

        log_rho_traj = np.asarray(log_probs_target) - np.asarray(log_probs_behavior)
        # return np.exp(np.sum(log_rho_traj) - np.max(log_rho_traj)) # Re-normalizing by the maximum
        return np.sum(log_rho_traj)
    elif clipping: # Clipping is the only thing that works?
        log_rhos = []
        for i in range(len(t_a)):
            rho_e = target_policy.log_prob(t_t[i], t_a[i])
            rho_b = behavior_policy.log_prob(t_t[i], t_a[i])
            log_rho = rho_e - rho_b
            log_rhos.append(log_rho)
        total_log_rho = np.sum(log_rhos)
        clipped_log_rho = np.clip(total_log_rho, a_min=-10, a_max=10) # You basically lose all signal and just get the clipped values
        return np.exp(clipped_log_rho)
    elif pdis:
        cumulative_log_rhos = []
        cumulative_log_rho = 0.0

        for t in range(len(t_a)):
            log_rho_e = target_policy.log_prob(t_t[t], t_a[t])
            log_rho_b = behavior_policy.log_prob(t_t[t], t_a[t])
            cumulative_log_rho += (log_rho_e - log_rho_b)
            cumulative_log_rhos.append(cumulative_log_rho)

        # Stabilize exponentiation by subtracting the max
        max_log_rho = np.max(cumulative_log_rhos)
        pdis_estimate = 0.0
        for t in range(len(t_a)):
            stable_log_rho = cumulative_log_rhos[t] - max_log_rho
            weight = np.exp(stable_log_rho)
            weight = np.exp(cumulative_log_rhos[t])
            pdis_estimate += weight * t_r[t]
        return pdis_estimate
    else:
        log_probs_target = []
        log_probs_behavior = []
        for i in range(len(t_a)):  # All of the actions in the trajectory
            rho_e = target_policy.log_prob(t_t[i], t_a[i])
            rho_b = behavior_policy.log_prob(t_t[i], t_a[i])
            log_probs_target.append(rho_e)
            log_probs_behavior.append(rho_b)
        log_rho_traj = np.sum(np.asarray(log_probs_target) - np.asarray(log_probs_behavior))
        return np.exp(log_rho_traj)

def calculate_cppi_weights(b_gen_t, b_gen_a, b_gen_r, b_t, b_a, b_r, target_policy, behavior_policy):
    T = len(b_a) # Horizon
    log_probs_target = [] # pi_e(b_gen) * pi_e(b_t)
    log_probs_behavior = [] # pi_b(b_gen) * pi_b(b_t
    for i in range(T):  # All of the actions in the trajectory
        rho_e_squared = target_policy.log_prob(b_gen_t[i], b_gen_a[i]) * target_policy.log_prob(b_t[i], b_a[i])
        rho_b_squared = behavior_policy.log_prob(b_gen_t[i], b_gen_a[i]) * behavior_policy.log_prob(b_t[i], b_a[i])
        log_probs_target.append(rho_e_squared)
        log_probs_behavior.append(rho_b_squared)

    log_rho_traj = np.asarray(log_probs_target) - np.asarray(log_probs_behavior)
    total_log_rho = np.sum(log_rho_traj)
    # log_rho_traj = np.clip(log_rho_traj, -8, 8) # Clipping
    # return np.exp(np.sum(log_rho_traj))
    # return np.exp(np.sum(log_rho_traj) - np.max(log_rho_traj))  # Re-normalizing by the maximum --> had a bug here.
    clipped_log_rho = np.clip(total_log_rho, a_min=-0.3,
                              a_max=0.3)  # You basically lose all signal and just get the clipped values
    return clipped_log_rho

def find_matched_trajectories(behavior_trajectories, behavior_matching_cp_ppi, return_difference_ij, initial_state):
    b_o_rs = behavior_trajectories['ep_returns']
    b_o_ts = behavior_trajectories['states']
    b_o_as = behavior_trajectories['actions']
    b_o_tr = behavior_trajectories['rewards']
    epsilon_r = 3
    epsilon_s = 1.5
    weights = [] # Each element should be a weight
    for i in range(len(b_o_rs[50:])): # For the second half of the data
        # If the initial state matches
        behavior_s0 = b_o_ts[i][0]
        behavior_return = b_o_rs[i]
        if (np.linalg.norm(initial_state[:8] - behavior_s0[:8]) < epsilon_s): # the first state matches
            matched_trajs = behavior_matching_cp_ppi[i] # all the trajectories that match
            for j in range(len(matched_trajs['ep_returns'])):
                diff_proposed_match = (np.abs(behavior_return - matched_trajs['ep_returns'][j]))
                if np.abs(return_difference_ij - diff_proposed_match) < epsilon_r:
                    w_cppi = calculate_cppi_weights(matched_trajs['states'][j], matched_trajs['actions'][j], matched_trajs['rewards'][j], b_o_ts[i], b_o_as[i], b_o_tr[i],target_policy, behavior_policy)
                    weights.append(w_cppi)
    return weights

def train_discriminator(behavior_trajectories, target_trajectories):
    state_dim = 17
    return_dim = 1

    model = TrajectoryClassifier(state_dim + return_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    # Trajectories from Target Policy and Learned Env = 1
    trajectories_target = target_trajectories['states']
    returns_target = target_trajectories['ep_returns']
    n_samples = len(returns_target)

    # Each sample is [17-dim first state] + [1-dim return] flattened
    trajectory_inputs = np.asarray(list(trajectories_target))[:int(n_samples*0.75), 0, :]
    return_inputs = np.asarray(returns_target)[:int(n_samples*0.75)]
    inputs_pi_e = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    labels_pi_e = torch.Tensor(np.ones(inputs_pi_e.shape[0]))

    # Trajectories from Original Env and Behavior Policy = 0
    trajectories_behavior = behavior_trajectories['states']
    returns_behavior = behavior_trajectories['ep_returns']
    trajectory_inputs = np.asarray(list(trajectories_behavior))[:int(n_samples*0.75), 0, :]
    return_inputs = np.asarray(returns_behavior)[:int(n_samples*0.75)]
    inputs_pi_b = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    labels_pi_b = torch.Tensor(np.zeros(inputs_pi_b.shape[0]))

    # Validation set
    trajectory_inputs = np.asarray(list(trajectories_target))[int(n_samples*0.75):, 0, :]
    return_inputs = np.asarray(returns_target)[int(n_samples*0.75):]
    val_pi_e = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    val_label_pi_e = torch.Tensor(np.ones(val_pi_e.shape[0]))

    trajectory_inputs = np.asarray(list(trajectories_behavior))[int(n_samples*0.75):, 0, :]
    return_inputs = np.asarray(returns_behavior)[int(n_samples*0.75):]
    val_pi_b = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    val_label_pi_b = torch.Tensor(np.zeros(val_pi_b.shape[0]))

    # Training Loop
    for epoch in range(1000):
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
    alpha = 0.05  # 95% coverage
    pi_e = 10
    pi_b = 9

    ENV = args.env
    rl_params = {
        'env_name': ENV,
    }

    with tf.io.gfile.GFile("./d4rl_policies.json", 'r') as f:
        policy_database = json.load(f)
    policy_metadatas = [i for i in policy_database if
                        i['task.task_names'][0].find(rl_params['env_name'].split("-")[0] + "-") != -1]


    target_policy = D4RL_Policy(policy_metadatas[pi_e]['policy_path'])
    behavior_policy = D4RL_Policy(policy_metadatas[pi_b]['policy_path'])
    first_term_trajs = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_target_trajectories.pkl", 'rb')) # target policy, learned dynamics
    t_rs = first_term_trajs['ep_returns']
    behavior_trajectories_o = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_offline_dataset.pkl", 'rb')) #behavior policy, original env
    target_trajectories_o = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_target_dataset.pkl", 'rb')) # target policy, original env
    print("v(pi_e) full policy = " + str(np.mean(target_trajectories_o['ep_returns'])))
    #
    # # # Discriminator approach
    b_o_rs = behavior_trajectories_o['ep_returns']
    b_o_ts = behavior_trajectories_o['states']
    b_o_as = behavior_trajectories_o['actions']
    b_o_tr = behavior_trajectories_o['rewards']
    #
    # discriminator = train_discriminator(behavior_trajectories=behavior_trajectories_o, target_trajectories=first_term_trajs)
    # weights = []
    # scores = []
    # for i in range(len(b_o_rs)):
    #     behavior_trajectory = b_o_ts[i]
    #     s_o = behavior_trajectory[0].flatten()  # First state of the behavior trajectory
    #     return_i = b_o_rs[i]
    #     input = np.hstack((s_o.reshape(1, -1), return_i.reshape(1, -1)))
    #     p_hat = discriminator(torch.Tensor(input))[0].squeeze(-1).detach().item()
    #     weight = p_hat / ((1 - p_hat)+1e-8)
    #     weights.append(weight)
    #     scores.append(b_o_rs[i])
    # weights = np.asarray(weights) / np.sum(weights) # TOOD: compare this with the return from just v_pie(s0)
    # quantiles = weighted_quantile(scores, alpha, weights)
    # print("Discriminator: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " Interval: (" + str(quantiles[0]) + ", " + str(
    #     quantiles[1]) + ")")
    #
    # # DR-PPI
    first_term = np.mean(t_rs)  # Calculating the first term (expectation over target rewards)
    var_f = np.std(t_rs) ** 2
    behavior_trajectories_matching_dr_ppi = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_matching_dr_ppi.pkl", 'rb'))
    second_term_all = []
    rhos = []
    for _, i in enumerate(tqdm.tqdm(range(len(b_o_rs)))):  # For each behavior trajectory
        behavior_trajectory = b_o_ts[i]
        behavior_actions = b_o_as[i]
        behavior_rewards = b_o_tr[i]
        rho = calculate_ips_product(behavior_trajectory, behavior_actions, behavior_rewards, target_policy, behavior_policy, clipping=True)
        behavior_return = b_o_rs[i]
        m_term = np.mean(behavior_trajectories_matching_dr_ppi[i]['ep_returns'])
        second_term_all.append(rho * behavior_return - m_term)

    var_b = np.nanstd(second_term_all) ** 2
    z = norm.ppf(1 - alpha / 2)
    var_ppi = var_f / len(t_rs) + var_b / len(second_term_all)
    mean_ppi = first_term + np.nanmean(second_term_all)
    print("DR-PPI: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " Interval: (" + str(mean_ppi - z * np.sqrt(var_ppi)) + ", " + str(mean_ppi + z * np.sqrt(var_ppi)) + ")")
    # CP-PPI
    # True value of the state
    # target_trajectories_s0 = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_target_dataset_s0.pkl", 'rb'))
    # print("v_pie(s_0) = " + str(np.mean(target_trajectories_s0['ep_returns'])))
    #
    # behavior_trajectories = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_offline_dataset.pkl", 'rb'))
    # first_term_trajs_s0 = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_target_trajectories_s0.pkl", 'rb'))  # target policy, learned dynamics
    # t_rs = first_term_trajs_s0['ep_returns']
    #
    # b_o_rs = behavior_trajectories['ep_returns']
    # b_o_ts = behavior_trajectories['states']
    # b_o_as = behavior_trajectories['actions']
    # b_o_tr = behavior_trajectories['rewards']
    # weights = []
    # scores = []
    # first_term = np.mean(t_rs) #This is for all of the states
    # # # TODO: generate more matches here
    # behavior_trajectories_matching_cp_ppi = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_matching_dr_ppi_2.pkl", 'rb'))
    # for _, i in enumerate(tqdm.tqdm(range(len(b_o_ts[:50])))):  # For all trajectories in the behavior dataset
    #     if i not in behavior_trajectories_matching_cp_ppi.keys(): # No matching trajectories were generated
    #         continue
    #     # Get the trajectories that already match
    #     if len(behavior_trajectories_matching_cp_ppi[i]['ep_returns']) == 0:  # No matched trajectories
    #         weights += [0]  # We don't have matched trajectories
    #         continue
    #     else:
    #         matched_trajs = behavior_trajectories_matching_cp_ppi[i]
    #         for j in range(len(matched_trajs['ep_returns'])): # First states and rewards already match, we can calculate the weight now
    #             # Calculate weight using D_tr (the other 50 trajectories)
    #
    #             # Find pairs (original env, learned env) in D_tr (the other 50) --> match in first state and return difference
    #             return_difference_ij = np.abs(np.abs(b_o_rs[i] - matched_trajs['ep_returns'][j]))
    #             initial_state = b_o_ts[i][0]
    #
    #             matching_weights = find_matched_trajectories(behavior_trajectories, behavior_trajectories_matching_cp_ppi, return_difference_ij, initial_state)
    #             if len(matching_weights) == 0:
    #                 weights += [0]
    #             else:
    #                 weights += [np.mean(matching_weights)]
    #             scores.append(return_difference_ij)
    # print("weights: " + str(weights))
    # weights = np.asarray(weights) / np.sum(weights) # Should be 150 weights
    #
    # quantiles = weighted_quantile(scores, alpha, weights)
    # print("CP-PPI: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " Interval: (" + str(first_term + quantiles[0]) + ", " + str((first_term + quantiles[1])) + ")")
    # import ipdb; ipdb.set_trace()

    # IS CLT estimator
    IPS_weights_scores = []
    for _, i in enumerate(tqdm.tqdm(range(len(b_o_ts)))):
        # Calculate Rho
        rho = calculate_ips_product(b_o_ts[i], b_o_as[i], b_o_tr[i], target_policy, behavior_policy, clipping=True)
        traj_ret = b_o_rs[i]
        IPS_weights_scores += [rho * traj_ret]

    z = norm.ppf(1 - alpha / 2)
    V_IS_mean = np.nanmean(IPS_weights_scores)
    V_IS_var = np.nanvar(IPS_weights_scores)
    V_IS_lb = V_IS_mean - z * np.sqrt(V_IS_var / len(IPS_weights_scores))
    V_IS_ub = V_IS_mean + z * np.sqrt(V_IS_var / len(IPS_weights_scores))


    print("IS CLT: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " \hat{v}(\pi_e): " + str(V_IS_mean) + " Interval: (" + str(V_IS_lb) + ", " + str(V_IS_ub) + ")")
    import ipdb; ipdb.set_trace()
    # IS bootstrap
    # IPS_estimates = []
    # for _, i in enumerate(tqdm.tqdm(range(20))):
    #     ips_val = 0
    #     for j in range(len(b_o_ts)):
    #         if i == j: # Skip this sample
    #             continue
    #         rho = calculate_ips_product(b_o_ts[i], b_o_as[i], b_o_rs[i], target_policy, behavior_policy, clipping=True)#  Must do clipping here
    #         traj_ret = b_o_rs[i]
    #         ips_val += rho * traj_ret
    #     IPS_estimates += [ips_val/(len(b_o_ts) - 1)]
    # V_IS_bootstrap_mean = np.mean(IPS_estimates)
    # V_IS_bootstrap_alpha = np.quantile(IPS_estimates, alpha)
    # V_IS_bootstrap_1_alpha = np.quantile(IPS_estimates, 1 - alpha)
    # print("IS Bootstrap: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " \hat{v}(\pi_e): " + str(V_IS_bootstrap_mean) + " Interval: (" + str(V_IS_bootstrap_alpha) + ", " + str(V_IS_bootstrap_1_alpha) + ")")
    # import ipdb; ipdb.set_trace()

    # Aug IS (Bootstrap)
    # behavior_trajectories_gen = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_behavior_baseline.pkl", 'rb'))
    # b_o_rs_gen = behavior_trajectories['ep_returns']
    # b_o_ts_gen = behavior_trajectories['states']
    # b_o_as_gen = behavior_trajectories['actions']
    # b_o_tr_gen = behavior_trajectories['rewards']
    # AugIPS_estimates = []
    # for _, i in enumerate(tqdm.tqdm(range(20))):
    #     ips_val = 0
    #     for j in range(len(b_o_ts)):
    #         if i == j: # Skip this sample
    #             continue
    #         rho = calculate_ips_product(b_o_ts[i], b_o_as[i], b_o_tr[i], target_policy, behavior_policy, clipping=True)#  Must do clipping here
    #         traj_ret = b_o_rs[i]
    #         ips_val += rho * traj_ret
    #
    #     for j in range(len(b_o_ts_gen)):
    #         if i == j:
    #             continue
    #         rho = calculate_ips_product(b_o_ts_gen[i], b_o_as_gen[i], b_o_tr_gen[i], target_policy, behavior_policy,
    #                                     clipping=True)  # Must do clipping here
    #         traj_ret = b_o_rs_gen[i]
    #         ips_val += rho * traj_ret
    #
    #     AugIPS_estimates += [ips_val/(len(b_o_ts) - 1 + len(b_o_ts_gen) - 1)]
    # V_IS_bootstrap_mean = np.mean(AugIPS_estimates)
    # V_IS_bootstrap_alpha = np.quantile(AugIPS_estimates, alpha)
    # V_IS_bootstrap_1_alpha = np.quantile(AugIPS_estimates, 1 - alpha)
    # print("IS Bootstrap: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " \hat{v}(\pi_e): " + str(V_IS_bootstrap_mean) + " Interval: (" + str(V_IS_bootstrap_alpha) + ", " + str(V_IS_bootstrap_1_alpha) + ")")
    import ipdb; ipdb.set_trace()

    # Aug IS (CLT)
    # TODO: roll out trajectories from learned dynamics model behavior policy

    # DM (Bootstrap)
    # TODO: roll out trajectories from the learned dynamics model in the behavior policy

    # DR (CLT)


    # Aug DR (CLT) --> This may be difficult to do

    # Aug DM (this may be difficult to do)








    # DR estimator
    DR_val = 0




