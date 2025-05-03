import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
# train VAE to learn the dynamic of inventory control problem
# s: inventory level, a: order quantity, r: reward, o: demand
# N = 10, k = 1, c = 2, z = 2, p = 4, lambda = 5.
# 0 \le a \le N
# o ~ Poisson(lambda)
# s' = max(0, min(N, s + a) - o)
# r = -k * 1(a>0) - c * (min(N, s + a) - s) - z * s + p * min(o, s+a)

class Inventory_Simulator(object):
    def __init__(self, N, k, c, z, p, lambda_, H):
        self.N = N
        self.k = k
        self.c = c
        self.z = z
        self.p = p
        self.lambda_ = lambda_
        self.H = H

    def reset(self): # return the initial state
        return np.random.uniform(0, self.N, (1,))

    def step(self, s, a, h): # transition function
        o = np.random.normal(5, 1, (1,)) # demand
        s1 = np.clip(np.clip(s + a, a_min=None, a_max=self.N) - o, 0, None)
        # s1 = np.array(s1)
        r = -self.k * int(a > 0) - self.c * (min(self.N, s + a) - s) - self.z * s + self.p * min(o, s + a)
        r = float(r)
        done = (h == self.H - 1)
        return s, a, r, s1, done


class Noisy_Inventory_Simulator(object):
    def __init__(self, N, k, c, z, p, lambda_, H):
        self.N = N
        self.k = k
        self.c = c
        self.z = z
        self.p = p
        self.lambda_ = lambda_
        self.H = H

    def reset(self): # return the initial state
        return np.random.uniform(0, self.N, (1,))

    def step(self, s, a, h): # transition function
        o = np.random.normal(5, 1, (1,)) # demand
        s1 = np.clip(np.clip(s + a + np.random.normal(0, 1, (1,)), a_min=None, a_max=self.N) - o, 0, None)
        r = -self.k * int(a > 0) - self.c * (min(self.N, s + a) - s) - self.z * s + self.p * min(o, s + a) + np.random.normal(1, 1, (1,)) # some positive bias
        r = float(r)
        done = (h == self.H - 1)
        return s, a, r, s1, done
    
# define 10 policies to evaluate
# several [a, b] uniform policies
# several (s, S) policies
# several N(mu, sigma^2) policies
from scipy.stats import norm

class uniform_policy(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def act(self, s):
        a = np.random.randint(self.a, self.b+1, (1,))
        return a
    
    def prob(self, s, a):
        return 1.0 / (self.b - self.a + 1)

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

class normal_policy(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def act(self, s):
        # sample from 0 to 10, with the probability proportional to exp(-|x-mu+s|)
        prob = [float(np.exp(-abs(x - self.mu + s))) for x in range(11)]
        # normalize the probability
        prob = [p/sum(prob) for p in prob]
        a = np.random.choice(range(11), p=prob)
        return a
    
    def prob(self, s, a):
        # calculate the probability of a given s and a
        prob = [float(np.exp(-abs(x - self.mu + s))) for x in range(11)]
        # normalize the probability
        prob = [p/sum(prob) for p in prob]
        return prob[int(a)]
        # prob = 1.0 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((a - self.mu + s) / self.sigma) ** 2)
        # # reweight for truncated normal distribution
        # # compute P(0 <= X <= 10)
        # integral_0_10 = norm.cdf(10, loc=self.mu-s, scale=self.sigma) - norm.cdf(0, loc = self.mu-s, scale=self.sigma)
        # return prob / integral_0_10

class approximate_uniform_policy(object):
    def __init__(self, a, b, eps):
        self.a = a
        self.b = b
        self.eps = eps
        self.idx_add_eps = range(5)
        self.idx_sub_eps = range(5, 10)
        self.prob_list = [1/11 for _ in range(11)]
        for i in self.idx_add_eps:
            self.prob_list[i] = self.prob_list[i] + self.eps
        for i in self.idx_sub_eps:
            self.prob_list[i] = self.prob_list[i] - self.eps

    def act(self, s):
        a = np.random.choice(range(11), p=self.prob_list)
        return a
    
    def prob(self, s, a):
        return self.prob_list[int(a)]

env = Inventory_Simulator(N=10, k=1, c=2, z=2, p=4, lambda_=5, H=20)
noisy_env = Noisy_Inventory_Simulator(N=10, k=1, c=2, z=2, p=4, lambda_=5, H=20)

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

pi_n_10_1 = normal_policy(5.5, 1)

result = pd.DataFrame(columns=['delta', 'iter', 'V_PPI', 'lower_bound', 'upper_bound'])

pi_b = pi_u_0_10
gamma = 0.9
_trajectories = []
_actions_trajectories = []
_rewards = []
for _ in range(250):
    s = env.reset()
    reward = 0
    trajectory = []
    action_trajectory = []
    for h in range(env.H):
        a = pi_b.act(s)
        s, a, r, s1, done = env.step(s, a, h)
        trajectory.append(s)
        action_trajectory.append(a)
        reward += r * (gamma ** h)
        s = s1
    _trajectories.append(trajectory)
    _actions_trajectories.append(action_trajectory)
    _rewards.append(reward)
# convert to dictionary
data = {
    'trajectories': np.array(_trajectories),
    'actions': np.array(_actions_trajectories),
    'rewards': np.array(_rewards)
}
# save as 0_o.pkl
with open('dr_ci_trajs/train_0_o.pkl', 'wb') as f:
    pickle.dump(data, f)

_trajectories = []
_actions_trajectories = []
_rewards = []
for _ in range(250):
    s = env.reset()
    reward = 0
    trajectory = []
    action_trajectory = []
    for h in range(env.H):
        a = pi_b.act(s)
        s, a, r, s1, done = env.step(s, a, h)
        trajectory.append(s)
        action_trajectory.append(a)
        reward += r * (gamma ** h)
        s = s1
    _trajectories.append(trajectory)
    _actions_trajectories.append(action_trajectory)
    _rewards.append(reward)
# convert to dictionary
data = {
    'trajectories': np.array(_trajectories),
    'actions': np.array(_actions_trajectories),
    'rewards': np.array(_rewards)
}
# save as 0_o.pkl
with open('dr_ci_trajs/first_term_0_o.pkl', 'wb') as f:
    pickle.dump(data, f)

for delta in np.linspace(0, 0.1, 10, endpoint=False):
    approximate_pi_u_0_10 = approximate_uniform_policy(0, 10, delta)
    pi_e = approximate_pi_u_0_10
    min_ratio = pi_e.prob(0, 5) / pi_b.prob(0, 5)


    for iter in range(10):

        train_trajectories = pickle.load(open("dr_ci_trajs/train_0_o.pkl", "rb"))
        _trajectories = []
        _actions_trajectories = []
        _rewards = []
        for i in range(len(train_trajectories['trajectories'])):
            s0 = train_trajectories['trajectories'][i][0]
            for _ in range(100): # M = 100
                s = s0
                reward = 0
                trajectory = []
                action_trajectory = []
                for h in range(env.H):
                    a = pi_e.act(s)
                    s, a, r, s1, done = noisy_env.step(s, a, h)
                    trajectory.append(s)
                    action_trajectory.append(a)
                    reward += r * (gamma ** h)
                    s = s1
                _trajectories.append(trajectory)
                _actions_trajectories.append(action_trajectory)
                _rewards.append(reward)
        # convert to dictionary
        data = {
            'trajectories': np.array(_trajectories),
            'actions': np.array(_actions_trajectories),
            'rewards': np.array(_rewards)
        }
        # save as 0_o.pkl
        with open('dr_ci_trajs/train_9_diff.pkl', 'wb') as f:
            pickle.dump(data, f)
        

        train_trajectories = pickle.load(open("dr_ci_trajs/first_term_0_o.pkl", "rb"))
        _trajectories = []
        _actions_trajectories = []
        _rewards = []
        for i in range(len(train_trajectories['trajectories'])):
            s0 = train_trajectories['trajectories'][i][0]
            for _ in range(100): # N_f = 250*100
                s = s0
                reward = 0
                trajectory = []
                action_trajectory = []
                for h in range(env.H):
                    a = pi_e.act(s)
                    s, a, r, s1, done = noisy_env.step(s, a, h)
                    trajectory.append(s)
                    action_trajectory.append(a)
                    reward += r * (gamma ** h)
                    s = s1
                _trajectories.append(trajectory)
                _actions_trajectories.append(action_trajectory)
                _rewards.append(reward)
        # convert to dictionary
        data = {
            'trajectories': np.array(_trajectories),
            'actions': np.array(_actions_trajectories),
            'rewards': np.array(_rewards)
        }
        with open('dr_ci_trajs/first_term_9_diff.pkl', 'wb') as f:
            pickle.dump(data, f)
        
                
        train_0_o = pickle.load(open("dr_ci_trajs/train_0_o.pkl", "rb"))
        train_9_diff = pickle.load(open("dr_ci_trajs/train_9_diff.pkl", "rb"))
        first_term_diff = pickle.load(open("dr_ci_trajs/first_term_9_diff.pkl", "rb"))

        first_term_diff_rewards = first_term_diff['rewards']
        first_term_diff_rewards = np.array(first_term_diff_rewards)

        sigma_f2 = np.var(first_term_diff_rewards)
        
        pi_b_0_trajs = train_0_o['trajectories']
        pi_b_0_actions = train_0_o['actions']
        pi_b_0_rewards = train_0_o['rewards']
        pi_e_diff_trajs = train_9_diff['trajectories']
        pi_e_diff_actions = train_9_diff['actions']
        pi_e_diff_rewards = train_9_diff['rewards']

        s0_pairs = []
        for i in range(len(pi_b_0_trajs)):
            for j in range(len(pi_e_diff_trajs)):
                if pi_b_0_trajs[i][0] == pi_e_diff_trajs[j][0]:
                    s0_pairs.append((i, j))
        
        weights = np.zeros(len(pi_b_0_trajs))
        for i in range(len(pi_b_0_trajs)):
            w = 1
            for t in range(len(pi_b_0_trajs[i])):
                w *= pi_e.prob(pi_b_0_trajs[i][t], pi_b_0_actions[i][t]) / pi_b.prob(pi_b_0_trajs[i][t], pi_b_0_actions[i][t])
            weights[i] = w
        
        # E_diff_rewards = np.zeros(len(pi_b_0_trajs))
        # for i, j in s0_pairs:  
        #     diff = weights[i]*pi_b_0_rewards[i] - pi_e_diff_rewards[j]
        #     E_diff_rewards[i] += diff
        # E_diff_rewards = E_diff_rewards / (len(pi_e_diff_trajs)/len(pi_b_0_trajs))

        # sigma_b2 = np.var(E_diff_rewards)

        # V_PPI = np.mean(first_term_diff_rewards) + np.mean(E_diff_rewards)
        # sigma_V2 = sigma_f2/len(first_term_diff_rewards) + sigma_b2/len(E_diff_rewards)
        # alpha = 0.1
        # # calculate the confidence interval using normal approximation
        # z = norm.ppf(1 - alpha/2)
        # lower_bound = V_PPI - z * np.sqrt(sigma_V2)
        # upper_bound = V_PPI + z * np.sqrt(sigma_V2)

        start_s = 5
        eps_s = 0.3

        s0_pairs_s = []
        for i, j in s0_pairs:
            if pi_b_0_trajs[i][0] > start_s - eps_s and pi_b_0_trajs[i][0] < start_s + eps_s:
                s0_pairs_s.append((i, j))
        
        # TODO: WIS?
        weights_s = []
        E_diff_rewards_s = []
        has_visited = []
        for i, j in s0_pairs_s:
            if i not in has_visited:
                weights_s.append(weights[i])
                has_visited.append(i)
        E_diff_rewards = np.zeros(len(pi_b_0_trajs))
        for i, j in s0_pairs:  
            diff = weights[i]/np.sum(weights_s) * pi_b_0_rewards[i] - pi_e_diff_rewards[j]
            E_diff_rewards[i] += diff
        E_diff_rewards = E_diff_rewards / (len(pi_e_diff_trajs)/len(pi_b_0_trajs))
        has_visited = []
        for i, j in s0_pairs_s:
            if i not in has_visited:
                E_diff_rewards_s.append(E_diff_rewards[i])
                has_visited.append(i)
        E_diff_rewards_s = np.array(E_diff_rewards_s)
        sigma_b2_s = np.var(E_diff_rewards_s)

        _trajectories = []
        _actions_trajectories = []
        _rewards = []
        for _ in range(500):
            s = np.array(start_s).reshape(1,)
            reward = 0
            trajectory = []
            action_trajectory = []
            for h in range(env.H):
                a = pi_e.act(s)
                s, a, r, s1, done = noisy_env.step(s, a, h)
                trajectory.append(s)
                action_trajectory.append(a)
                reward += r * (gamma ** h)
                s = s1
            _trajectories.append(trajectory)
            _actions_trajectories.append(action_trajectory)
            _rewards.append(reward)

        # convert to dictionary
        import pickle
        data = {
            'trajectories': np.array(_trajectories),
            'actions': np.array(_actions_trajectories),
            'rewards': np.array(_rewards)
        }

        # save as 9_first_term.pkl
        with open('dr_ci_trajs/9_first_term_s.pkl', 'wb') as f:
            pickle.dump(data, f)

        pi_e_diff = pickle.load(open("dr_ci_trajs/9_first_term_s.pkl", "rb"))

        pi_e_diff_trajs = pi_e_diff['trajectories']
        pi_e_diff_actions = pi_e_diff['actions']
        pi_e_diff_rewards = pi_e_diff['rewards']

        first_term_diff_rewards_s = np.array(pi_e_diff_rewards)

        sigma_f2_s = np.var(first_term_diff_rewards_s)

        V_PPI_s = np.mean(first_term_diff_rewards_s) + np.mean(E_diff_rewards_s)
        sigma_V2_s = sigma_f2_s/len(first_term_diff_rewards_s) + sigma_b2_s/len(E_diff_rewards_s)
        alpha = 0.05
        # calculate the confidence interval using normal approximation
        z = norm.ppf(1 - alpha/2)
        lower_bound_s = V_PPI_s - z * np.sqrt(sigma_V2_s)
        upper_bound_s = V_PPI_s + z * np.sqrt(sigma_V2_s)

        # save a row to result
        result = pd.concat([result, pd.DataFrame({'delta': [delta], 'iter': [iter], 'V_PPI': [V_PPI_s], 'lower_bound': [lower_bound_s], 'upper_bound': [upper_bound_s]})], ignore_index=True)
        print(f"delta: {delta}, iter: {iter}, V_PPI: {V_PPI_s}, lower_bound: {lower_bound_s}, upper_bound: {upper_bound_s}")

# save the result
result.to_csv('dr_ci_trajs/result_per_state_5_WIS.csv', index=False)