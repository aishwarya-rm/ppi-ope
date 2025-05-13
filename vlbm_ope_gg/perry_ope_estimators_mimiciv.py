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
from torch.utils.data import DataLoader, TensorDataset

tf.disable_v2_behavior()

seed = 42

tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


ACTION_TO_IDX = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4}
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
            rho_e = np.log(target_policy[t_a[i]])
            rho_b = np.log(behavior_policy[t_a[i]])
            log_probs_target.append(rho_e)
            log_probs_behavior.append(rho_b)

        log_rho_traj = np.asarray(log_probs_target) - np.asarray(log_probs_behavior)
        # return np.exp(np.sum(log_rho_traj) - np.max(log_rho_traj)) # Re-normalizing by the maximum
        return np.sum(log_rho_traj)
    elif clipping:
        log_rhos = []
        for i in range(len(t_a)):
            action = int(np.clip((t_a[i]) // 10 * 10, 0, 40))
            action_idx = ACTION_TO_IDX[action]
            rho_e = np.log(target_policy[action_idx])
            rho_b = np.log(behavior_policy[action_idx])
            log_rho = rho_e - rho_b
            log_rhos.append(log_rho)
        total_log_rho = np.sum(log_rhos)
        clipped_log_rho = np.clip(total_log_rho, a_min=-0.5, a_max=0.5) # You basically lose all signal and just get the clipped values
        return np.exp(clipped_log_rho)
    elif pdis:
        cumulative_log_rhos = []
        cumulative_log_rho = 0.0

        for t in range(len(t_a)):
            action = int(np.clip((t_a[t]) // 10 * 10, 0, 40))
            action_idx = ACTION_TO_IDX[action]
            log_rho_e = np.log(target_policy(torch.Tensor(t_t[t]))[action_idx].detach().numpy().item())
            log_rho_b = np.log(behavior_policy(torch.Tensor(t_t[t]))[action_idx].detach().numpy().item())
            cumulative_log_rho += (log_rho_e - log_rho_b)
            cumulative_log_rhos.append(cumulative_log_rho)

        # Stabilize exponentiation by subtracting the max
        pdis_estimate = 0.0
        for t in range(len(t_a)):
            weight = np.exp(cumulative_log_rhos[t])
            weight = np.clip(weight, a_min=-10, a_max=3.6)
            pdis_estimate += weight * t_r[t]
        return pdis_estimate
    else:
        probs_target = []
        probs_behavior = []
        for i in range(len(t_a)):  # All of the actions in the trajectory
            action = int(np.clip((t_a[i]) // 10 * 10, 0, 40))
            action_idx = ACTION_TO_IDX[action]
            rho_e = target_policy(torch.Tensor(t_t[i]))[action_idx].detach().numpy().item()
            rho_b = behavior_policy(torch.Tensor(t_t[i]))[action_idx].detach().numpy().item()
            probs_target.append(rho_e)
            probs_behavior.append(rho_b)
        if np.prod(probs_behavior) == 0:
            return 0 # I think the data is bad.
        else:
            rho_traj = np.prod(np.asarray(probs_target))/ np.prod(np.asarray(probs_behavior))
            return rho_traj

def calculate_cppi_weights(b_gen_t, b_gen_a, b_gen_r, b_t, b_a, b_r, target_policy, behavior_policy):
    T = len(b_a) # Horizon
    probs_target = [] # pi_e(b_gen) * pi_e(b_t)
    probs_behavior = [] # pi_b(b_gen) * pi_b(b_t
    for i in range(T):  # All of the actions in the trajectory
        action_gen_idx = ACTION_TO_IDX[np.argmax(b_gen_a[i])]
        action_t = int(np.clip((b_a[i]) // 10 * 10, 0, 40))
        action_t_idx = ACTION_TO_IDX[action_t]
        rho_e_squared = target_policy(torch.Tensor(b_gen_t[i]))[action_gen_idx].detach().numpy().item() * target_policy(torch.Tensor(b_t[i]))[action_t_idx].detach().numpy().item()
        rho_b_squared = behavior_policy(torch.Tensor(b_gen_t[i]))[action_gen_idx].detach().numpy().item() * behavior_policy(torch.Tensor(b_t[i]))[action_t_idx].detach().numpy().item()
        probs_target.append(rho_e_squared)
        probs_behavior.append(rho_b_squared)

    if np.prod(probs_behavior) == 0:
        return 0 # This pair of trajectories is either unlikely or bad --> we don't want to consider this in the correction
    else:
        rho_traj = np.prod(probs_target) / np.prod(probs_behavior)
        # total_log_rho = np.sum(log_rho_traj)
        # log_rho_traj = np.clip(total_log_rho, -8, 8) # Clipping
        return rho_traj

def find_matched_trajectories(behavior_trajectories, behavior_matching_cp_ppi, return_difference_ij, initial_state):
    b_o_rs = behavior_trajectories['reward_sum']
    b_o_ts = behavior_trajectories['states']
    b_o_as = behavior_trajectories['actions']
    b_o_tr = behavior_trajectories['rewards']
    epsilon_r = 0.4
    epsilon_s = 0.5
    weights = [] # Each element should be a weight
    for i in range(50, 100): # For the second half of the data
        # If the initial state matches
        behavior_s0 = b_o_ts[i][0]
        behavior_return = b_o_rs[i]
        if np.linalg.norm(np.array(initial_state)/np.linalg.norm(np.array(initial_state)) - np.array(behavior_s0)/np.linalg.norm(np.array(behavior_s0))) < epsilon_s: # the first state matches
            matched_trajs = behavior_matching_cp_ppi[i] # all the trajectories that match
            for j in range(len(matched_trajs['ep_returns'])):
                diff_proposed_match = behavior_return - matched_trajs['ep_returns'][j]
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

class DiscretePolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions=5, hidden_sizes=[100, 100]):
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
        logits = self.model(x)  # raw logits
        return torch.softmax(logits, dim=-1) # Probability distribution

def train_discrete_policy(states, one_hot_actions, state_dim, epochs=100, batch_size=64, lr=1e-3):
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
    return policy

if __name__ == '__main__':
    alpha = 0.05  # 95% coverage

    first_term_trajs = pickle.load(open("./saved_trajectories/mimic-iv/first_term_dr_ppi_mimiciv.pkl", 'rb')) # target policy, learned dynamics
    t_rs = first_term_trajs['ep_returns']
    first_term_trajs2 = pickle.load(open("./saved_trajectories/mimic-iv/first_term_dr_ppi_mimiciv_2.pkl", 'rb'))
    t_rs2 = first_term_trajs2['ep_returns']
    behavior_trajectories_o = pickle.load(open("mimic_iv_trajectories_behavior.pkl", 'rb')) #behavior policy, original env
    target_trajectories_o = pickle.load(open("mimic_iv_trajectories_target.pkl", 'rb')) # target policy, original env
    _trajectories_behavior = pickle.load(open("mimic_iv_behavior_trajectories.pkl", 'rb'))
    _trajectories_target = pickle.load(open("mimic_iv_target_trajectories.pkl", 'rb'))
    behavior_policy = train_discrete_policy(_trajectories_behavior['observations'], _trajectories_behavior['actions'], state_dim=20)
    target_policy = train_discrete_policy(_trajectories_target['observations'], _trajectories_target['actions'], state_dim=20)
    print("v(pi_e) full policy = " + str(np.mean(target_trajectories_o['reward_sum']))) # Considering only the first 100 behavior trajectories.
    # # # Discriminator approach
    b_o_rs = behavior_trajectories_o['reward_sum'][:100]
    b_o_ts = behavior_trajectories_o['states'][:100]
    b_o_as = behavior_trajectories_o['actions'][:100]
    b_o_tr = behavior_trajectories_o['rewards'][:100]
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
    behavior_trajectories_matching_dr_ppi = pickle.load(open('./saved_trajectories/mimic-iv/matching_dr_ppi_mimiciv.pkl', 'rb'))
    second_term_all = []
    rhos = []
    for _, i in enumerate(tqdm.tqdm(range(len(b_o_rs)))):  # For each behavior trajectory
        behavior_trajectory = b_o_ts[i]
        behavior_actions = b_o_as[i]
        behavior_rewards = b_o_tr[i]
        rho = calculate_ips_product(behavior_trajectory, behavior_actions, behavior_rewards, target_policy, behavior_policy)
        behavior_return = b_o_rs[i]
        m_term = np.mean(behavior_trajectories_matching_dr_ppi[i]['ep_returns'])
        second_term_all.append(rho*behavior_return - m_term) # If pdis, rho includes the return of the trajectory already

    var_b = np.nanstd(second_term_all) ** 2
    z = norm.ppf(1 - alpha / 2)
    var_ppi = var_f / len(t_rs) + var_b / len(second_term_all)
    mean_ppi = first_term + np.nanmean(second_term_all)
    print("DR-PPI: Interval: (" + str(mean_ppi - z * np.sqrt(var_ppi)) + ", " + str(mean_ppi + z * np.sqrt(var_ppi)) + ")")
    # CP-PPI
    # True value of the state
    # print("v_pie(s_0) = " + str(behavior_trajectories_o['reward_sum'][0])) # This is the true value
    # behavior_trajectories_o = pickle.load(open("mimic_iv_trajectories_behavior.pkl", 'rb')) # 100
    # first_term_trajs_s0 = pickle.load(open('./saved_trajectories/mimic-iv/first_term_cp_ppi_mimiciv.pkl', 'rb'))  # target policy, learned dynamics
    # t_rs = first_term_trajs_s0['ep_returns']
    #
    # b_o_rs = behavior_trajectories_o['reward_sum']
    # b_o_ts = behavior_trajectories_o['states']
    # b_o_as = behavior_trajectories_o['actions']
    # b_o_tr = behavior_trajectories_o['rewards']
    # weights = []
    # scores = []
    # first_term = np.mean(t_rs)
    # behavior_trajectories_matching_cp_ppi = pickle.load(open('./saved_trajectories/mimic-iv/matching_cp_ppi_mimiciv.pkl', 'rb'))
    # for _, i in enumerate(tqdm.tqdm(range(len(b_o_ts[:50])))):  # For all trajectories in the behavior dataset
    #     # Get the trajectories that already match
    #     matched_trajs = behavior_trajectories_matching_cp_ppi[i]
    #     for j in range(len(matched_trajs['ep_returns'])):
    #         return_difference_ij = b_o_rs[i] - matched_trajs['ep_returns'][j]
    #         initial_state = b_o_ts[i][0] # First state
    #         matching_weights = find_matched_trajectories(behavior_trajectories_o, behavior_trajectories_matching_cp_ppi, return_difference_ij, initial_state)
    #         print("Found : " + str(len(matching_weights)) + " matches")
    #         if len(matching_weights) == 0:
    #            weights += [1]
    #         else:
    #             weights += [np.mean(matching_weights)]
    #             scores.append(return_difference_ij)
    # print("weights: " + str(weights))
    # weights = np.asarray(weights) / np.sum(weights)
    #
    # quantiles = weighted_quantile(scores, alpha, weights)
    # print("CP-PPI: Interval: (" + str(first_term + quantiles[0]) + ", " + str((first_term + quantiles[1])) + ")")
    # import ipdb; ipdb.set_trace()

    # IS CLT estimator
    IPS_weights_scores = []
    for _, i in enumerate(tqdm.tqdm(range(len(b_o_ts)))):
        # Calculate Rho
        rho = calculate_ips_product(b_o_ts[i], b_o_as[i], b_o_tr[i], target_policy, behavior_policy)
        traj_ret = b_o_rs[i]
        # IPS_weights_scores += [rho * traj_ret]
        IPS_weights_scores += [rho*traj_ret]
    z = norm.ppf(1 - alpha / 2)
    V_IS_mean = np.nanmean(IPS_weights_scores)
    V_IS_var = np.nanvar(IPS_weights_scores)
    V_IS_lb = V_IS_mean - z * np.sqrt(V_IS_var / len(IPS_weights_scores))
    V_IS_ub = V_IS_mean + z * np.sqrt(V_IS_var / len(IPS_weights_scores))
    print("IS CLT: \hat{v}(\pi_e): " + str(V_IS_mean) + " Interval: (" + str(V_IS_lb) + ", " + str(V_IS_ub) + ")")
    # IS bootstrap
    # IPS_estimates = []
    # for _, i in enumerate(tqdm.tqdm(range(20))):
    #     ips_weights_scores = []
    #     for j in range(len(b_o_ts)):
    #         if i == j: # Skip this sample
    #             continue
    #         rho = calculate_ips_product(b_o_ts[j], b_o_as[j], b_o_tr[j], target_policy, behavior_policy, pdis=True)
    #         traj_ret = b_o_rs[j]
    #         ips_weights_scores += [rho] #* traj_ret
    #     ips_mean = np.nanmean(ips_weights_scores)
    #     IPS_estimates += [ips_mean]
    # 
    # V_IS_bootstrap_mean = np.mean(IPS_estimates)
    # V_IS_bootstrap_alpha = np.quantile(IPS_estimates, alpha)
    # V_IS_bootstrap_1_alpha = np.quantile(IPS_estimates, 1 - alpha)
    # print("IS Bootstrap: \hat{v}(\pi_e): " + str(V_IS_bootstrap_mean) + " Interval: (" + str(V_IS_bootstrap_alpha) + ", " + str(V_IS_bootstrap_1_alpha) + ")")
    # 
    # # Aug IS (Bootstrap)
    # behavior_trajectories_gen = pickle.load(open('./saved_trajectories/mimic-iv/augment_b_mimiciv.pkl', 'rb'))
    # b_o_rs_gen = behavior_trajectories_gen['ep_returns']
    # b_o_ts_gen = behavior_trajectories_gen['states']
    # b_o_as_gen = behavior_trajectories_gen['actions']
    # b_o_tr_gen = behavior_trajectories_gen['rewards']
    # AugIPS_estimates = []
    # for _, i in enumerate(tqdm.tqdm(range(20))):
    #     ips_val = []
    #     for j in range(len(b_o_ts)):
    #         if i == j: # Skip this sample
    #             continue
    #         rho = calculate_ips_product(b_o_ts[j], b_o_as[j], b_o_tr[j], target_policy, behavior_policy, pdis=True)
    #         traj_ret = b_o_rs[j]
    #         ips_val += [rho] #* traj_ret
    # 
    #     for j in range(len(b_o_ts_gen)):
    #         if i == j:
    #             continue
    #         actions = [np.argmax(np.asarray(b_o_as_gen[k])) for k in range(len(b_o_as_gen))]
    #         rho = calculate_ips_product(b_o_ts_gen[j], actions, b_o_tr_gen[j], target_policy, behavior_policy, pdis=True)
    #         traj_ret = b_o_rs_gen[j]
    #         ips_val += [rho] #* traj_ret
    # 
    #     AugIPS_estimates += [np.nanmean(ips_val)]
    # V_IS_bootstrap_mean = np.mean(AugIPS_estimates)
    # V_IS_bootstrap_alpha = np.quantile(AugIPS_estimates, alpha)
    # V_IS_bootstrap_1_alpha = np.quantile(AugIPS_estimates, 1 - alpha)
    # print("AugIS Bootstrap: \hat{v}(\pi_e): " + str(V_IS_bootstrap_mean) + " Interval: (" + str(V_IS_bootstrap_alpha) + ", " + str(V_IS_bootstrap_1_alpha) + ")")
    # 
    # # Aug IS (CLT)
    # IPS_weights_scores = []
    # for _, i in enumerate(tqdm.tqdm(range(len(b_o_ts)))):
    #     rho = calculate_ips_product(b_o_ts[i], b_o_as[i], b_o_tr[i], target_policy, behavior_policy, pdis=True)
    #     traj_ret = b_o_rs[i]
    #     # IPS_weights_scores += [rho * traj_ret]
    #     IPS_weights_scores += [rho]
    # for j in range(len(b_o_ts_gen)):
    #     actions = [np.argmax(np.asarray(b_o_as_gen[i])) for i in range(len(b_o_as_gen))]
    #     rho = calculate_ips_product(b_o_ts_gen[j], actions, b_o_tr_gen[j], target_policy, behavior_policy, pdis=True)
    #     traj_ret = b_o_rs_gen[j]
    #     # IPS_weights_scores += [rho * traj_ret]
    #     IPS_weights_scores += [rho]
    # 
    # z = norm.ppf(1 - alpha / 2)
    # V_IS_mean = np.nanmean(IPS_weights_scores)
    # V_IS_var = np.nanvar(IPS_weights_scores)
    # V_IS_lb = V_IS_mean - z * np.sqrt(V_IS_var / len(IPS_weights_scores))
    # V_IS_ub = V_IS_mean + z * np.sqrt(V_IS_var / len(IPS_weights_scores))
    # 
    # print("AugIS CLT: \hat{v}(\pi_e): " + str(V_IS_mean) + " Interval: (" + str(V_IS_lb) + ", " + str(V_IS_ub) + ")")


    # TODO: DR estimators
    # DR (CLT)


    # Aug DR (CLT) --> This may be difficult to do


    # DR estimator
    DR_val = 0




