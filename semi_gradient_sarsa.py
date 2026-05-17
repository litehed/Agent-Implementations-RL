import numpy as np
import random
from tiles3 import IHT, tiles


class SemiGradientBase:
    def __init__(self, actions, iht_size,
                 state_low,
                 state_high,
                 num_tilings=8,
                 tiles_per_dim=8, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.iht = IHT(iht_size)
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.alpha = alpha / num_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.w = np.zeros(iht_size)

    def scale(self, state):
        return list((state - self.state_low) / (self.state_high - self.state_low) * self.tiles_per_dim)

    def get_features(self, state, action):
        return tiles(self.iht, self.num_tilings, self.scale(state), ints=[action])

    def q_hat(self, state, action):
        feature_vector = self.get_features(state, action)
        return float(np.sum(self.w[feature_vector]))

    def q_hat_all(self, state):
        all_q = np.zeros(self.actions)
        for a in range(self.actions):
            all_q[a] = self.q_hat(state, a)
        return all_q

    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() <= epsilon:
            return random.randint(0, self.actions - 1)
        q_vals = self.q_hat_all(state)
        max_q = np.max(q_vals)
        best = []
        for a in range(self.actions):
            if q_vals[a] == max_q:
                best.append(a)
        
        return random.choice(best)

    def best_path(self, env):
        state, _ = env.reset()
        path = [tuple(state)]
        done = False
        best_reward = 0
        max_steps = 3000
        while not done:
            action = self.choose_action(state, epsilon=0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            best_reward += reward
            path.append(state)
            if len(path) > max_steps:
                break
        return path, best_reward
    
    
class SemiGradientSARSA(SemiGradientBase):
    def train(self, env, episodes=1000):
        reward_history = [] 
        for episode in range(episodes):
            state, _ = env.reset()
            action = self.choose_action(state)
            total_reward = 0
            while True:
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                feature_vec = self.get_features(state, action)
                if terminated:
                    self.w[feature_vec] += self.alpha * (reward - self.q_hat(state, action))
                    break
                
                next_action = self.choose_action(next_state)
                td_error = reward + self.gamma * self.q_hat(next_state, next_action) - self.q_hat(state, action)
                self.w[feature_vec] += self.alpha * td_error
                
                if truncated:
                    break
                
                state = next_state
                action = next_action
            reward_history.append(total_reward)
        return reward_history
