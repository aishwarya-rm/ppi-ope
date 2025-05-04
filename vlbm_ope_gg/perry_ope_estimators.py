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
from scipy.stats import norm

slim = tf.contrib.slim
rnn = tf.contrib.rnn
tfd = tfp.distributions

parser = argparse.ArgumentParser()
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-env", type=str,
                    help="Choose environment from <ant/hopper/walker2d>-<medium/medium-expert>-v2. Use the other script to evaluate on Halfcheetah. DEFAULT=halfcheetah-medium-expert-v2",
                    default='halfcheetah-medium-expert-v2')
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

def calculate_ips_product(t_t, t_a, t_r, target_policy, behavior_policy, wis=True, pdis=False, clipping=False):
    # T_t is the states
    if wis: # This works sometimes but produces a biased estimate
        ips_weight_sum = 0
        log_probs_target = []
        log_probs_behavior = []
        for i in range(len(t_a)): # All of the actions in the trajectory
            rho_e = target_policy.log_prob(t_t[i], t_a[i])
            rho_b = behavior_policy.log_prob(t_t[i], t_a[i])
            log_probs_target.append(rho_e)
            log_probs_behavior.append(rho_b)
            ips_weight_sum += (rho_e - rho_b)
        log_rho_traj = np.sum(np.asarray(log_probs_target) - np.asarray(log_probs_behavior))
        return np.exp(log_rho_traj - np.max(log_rho_traj)) # Re-normalizing by the maximum
    elif clipping: # this basically never works
        clip_max = 100
        log_clip_max = np.log(clip_max)
        log_clip_min = np.log(1.0 / clip_max)

        ips_weight_product = 1.0
        for i in range(len(t_a)):
            rho_e = target_policy.log_prob(t_t[i], t_a[i])
            rho_b = behavior_policy.log_prob(t_t[i], t_a[i])
            log_rho = rho_e - rho_b
            clipped_log_rho = np.clip(log_rho, log_clip_min, log_clip_max)
            ips_weight_product *= np.exp(clipped_log_rho)

        return ips_weight_product
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
            pdis_estimate += weight * t_r[t]

        # Optional: rescale to restore magnitude (may not be needed for relative comparisons)
        # pdis_estimate *= np.exp(max_log_rho)

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


def calculate_cppi_weights(b_gen_t, b_gen_a, b_gen_r, b_t, b_a, b_r, target_policy, behavior_policy, wis=True, pdis=False, clipping=False):
    # Basically always doing weighted importance sampling
    T = len(b_a) # Horizon
    log_probs_target = [] # pi_e(b_gen) * pi_e(b_t)
    log_probs_behavior = [] # pi_b(b_gen) * pi_b(b_t
    for i in range(T):  # All of the actions in the trajectory
        rho_e_squared = target_policy.log_prob(b_gen_t[i], b_gen_a[i]) * target_policy.log_prob(b_t[i], b_a[i])
        rho_b_squared = behavior_policy.log_prob(b_gen_t[i], b_gen_a[i]) * behavior_policy.log_prob(b_t[i], b_a[i])
        log_probs_target.append(rho_e_squared)
        log_probs_behavior.append(rho_b_squared)
    log_rho_traj = np.sum(np.asarray(log_probs_target) - np.asarray(log_probs_behavior))
    return np.exp(log_rho_traj - np.max(log_rho_traj))  # Re-normalizing by the maximum

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
    for epoch in range(10000):
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

    # Trajectories from the target policy, learned dynamics
    first_term_trajs = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_target_trajectories.pkl", 'rb'))
    # Trajectories from the behavior policy, original dynamics (offline behavior dataset)
    behavior_trajectories_o = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_offline_dataset.pkl", 'rb'))

    # True Target policy value = 20.69
    target_trajectories_o = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_target_dataset.pkl", 'rb'))
    print("v(pi_e) = " + str(np.mean(target_trajectories_o['ep_returns'])))

    # Discriminator approach (H=1000, very little data though)
    b_o_rs = behavior_trajectories_o['ep_returns']
    b_o_ts = behavior_trajectories_o['states']
    b_o_as = behavior_trajectories_o['actions']
    b_o_tr = behavior_trajectories_o['rewards']

    discriminator = train_discriminator(behavior_trajectories=behavior_trajectories_o, target_trajectories=first_term_trajs)
    weights = []
    scores = []
    for i in range(len(b_o_rs)):
        behavior_trajectory = b_o_ts[i]
        s_o = behavior_trajectory[0].flatten()  # First state of the behavior trajectory
        return_i = b_o_rs[i]
        input = np.hstack((s_o.reshape(1, -1), return_i.reshape(1, -1)))
        p_hat = discriminator(torch.Tensor(input))[0].squeeze(-1).detach().item()
        weight = p_hat / ((1 - p_hat)+1e-8)
        weights.append(weight)
        scores.append(b_o_rs[i])

    quantiles = weighted_quantile(scores, [alpha, 1 - alpha], weights)
    print("Discriminator: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " Interval: (" + str(quantiles[0]) + ", " + str(
        quantiles[1]) + ")")

    # DR-PPI
    t_rs = first_term_trajs['ep_returns']
    first_term = np.mean(t_rs)  # Calculating the first term (expectation over target rewards)
    var_f = np.std(t_rs) ** 2
    behavior_trajectories_matching_dr_ppi = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_matching_dr_ppi.pkl", 'rb'))
    second_term_all = []
    for i in range(len(b_o_rs)):  # For each behavior trajectory
        behavior_trajectory = b_o_ts[i]
        behavior_actions = b_o_as[i]
        behavior_rewards = b_o_tr[i]
        m_term = np.mean(behavior_trajectories_matching_dr_ppi[i]['ep_returns'])
        rho = calculate_ips_product(behavior_trajectory, behavior_actions, behavior_rewards, target_policy, behavior_policy)
        behavior_return = b_o_rs[i]
        second_term_all.append(rho * behavior_return - m_term)

    var_b = np.nanstd(second_term_all) ** 2
    z = norm.ppf(1 - alpha / 2)
    var_ppi = var_f / len(t_rs) + var_b / len(second_term_all)
    mean_ppi = first_term + np.nanmean(second_term_all)
    print("DR-PPI: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " Interval: (" + str(mean_ppi - z * np.sqrt(var_ppi)) + ", " + str(mean_ppi + z * np.sqrt(var_ppi)) + ")")

    # CP-PPI
    epsilon_r = 20
    weights = []
    first_term = np.mean(t_rs)
    behavior_trajectories_matching_cp_ppi = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_matching_cp_ppi.pkl", 'rb'))

    for i in range(len(b_o_ts)):  # For all trajectories in the behavior dataset
        if i not in behavior_trajectories_matching_cp_ppi.keys(): # No matching trajectories were generated
            continue
        # Get the trajectories that already match
        if len(behavior_trajectories_matching_cp_ppi[i]['ep_returns']) == 0:  # No matched trajectories
            weights += [0]  # We don't have matched trajectories
            continue
        else:
            matched_trajs = behavior_trajectories_matching_cp_ppi[i]

            w = 0
            n = 0
            for j in range(len(matched_trajs['ep_returns'])): # First states and rewards already match, we can calculate the weight now
                # Each pair is (i, j)
                traj = matched_trajs['states'][j]
                traj_actions = matched_trajs['actions'][j]
                traj_rewards = matched_trajs['rewards'][j]
                w += calculate_cppi_weights(traj, traj_actions, traj_rewards,b_o_ts[i], b_o_as[i], b_o_rs[i], target_policy, behavior_policy)
                # w += calculate_ips_product(traj, traj_actions, traj_rewards, target_policy, behavior_policy) * calculate_ips_product(b_o_ts[i], b_o_as[i], b_o_rs[i], target_policy,behavior_policy)
            w/= len(matched_trajs['ep_returns']) # TODO: the above calculation always returns 1?
            weights += [w]
    weights = np.asarray(weights) / np.sum(weights)

    all_weights = []
    scores = []
    for i in range(len(b_o_rs)):
        if i not in behavior_trajectories_matching_cp_ppi.keys(): # No matching trajectories were generated
            continue
        if len(behavior_trajectories_matching_cp_ppi[i]['ep_returns']) == 0:  # No matched trajectories
            continue
        else:
            matched_trajectories = behavior_trajectories_matching_cp_ppi[i]
            # The score should be an expectation over the difference in returns
            # The weight is the weights as calculated earlier.

            # For every trajectory that matched
            E_diff_rewards = 0
            for j in range(len(matched_trajectories['ep_returns'])):
                E_diff_rewards += (np.abs(b_o_rs[i] - matched_trajectories['ep_returns'][j]))
            E_diff_rewards /= len(matched_trajectories['ep_returns'])
            all_weights.append(weights[i])
            scores.append(E_diff_rewards)

    quantiles = weighted_quantile(scores, [1 - alpha, alpha], all_weights)

    print("CP-PPI: pi_b=" + str(pi_b) + " pi_e=" + str(pi_e) + " Interval: (" + str(first_term - quantiles[0]) + ", " + str((first_term - quantiles[1])) + ")")



