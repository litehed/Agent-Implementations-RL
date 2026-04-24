import numpy as np
import random


class MCBase:
    def __init__(self, actions, gamma=0.99, epsilon=0.5):
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        self.policy = {}

    def convert_state(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

    def get_q(self, state):
        state = self.convert_state(state)
        if state not in self.Q:
            self.Q[state] = np.zeros(self.actions)
        return self.Q[state]

    def get_policy(self, state):
        state = self.convert_state(state)
        if state not in self.policy:
            self.policy[state] = np.ones(self.actions) / self.actions
        return self.policy[state]

    def choose_action(self, state, epsilon=None):
        state = self.convert_state(state)
        if epsilon == 0:
            return int(np.argmax(self.get_q(state)))
        return np.random.choice(self.actions, p=self.get_policy(state))

    def best_path(self, env):
        state, _ = env.reset()
        state = self.convert_state(state)
        path = [state]
        done = False
        best_reward = 0
        max_steps = 3000
        while not done:
            action = self.choose_action(state, epsilon=0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = self.convert_state(next_state)
            done = terminated or truncated
            state = next_state
            best_reward += reward
            path.append(state)
            if len(path) > max_steps:
                break
        return path, best_reward


# Estimates policy rather than action-val
class OnPolicyMC(MCBase):
    def __init__(self, actions, gamma=0.99, epsilon=0.5):
        super().__init__(actions, gamma, epsilon)

    def train(self, env, episodes=1000):
        reward_history = np.zeros(episodes)
        Returns = {}
        for episode in range(episodes):
            state, _ = env.reset()
            state = self.convert_state(state)
            done = False
            episode_data = []
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.convert_state(next_state)
                done = terminated or truncated
                episode_data.append((state, action, reward))
                reward_history[episode] += reward
                state = next_state

            G = 0
            T = len(episode_data)
            # I keep forgetting its [start, end) not [start, end]
            for t in range(T - 1, -1, -1):
                state_t, action_t, reward_t = episode_data[t]
                G = self.gamma * G + reward_t  # reward t rather than t+1 because append

                first_visit = True
                for i in range(t):
                    if episode_data[i][0] == state_t and episode_data[i][1] == action_t:
                        first_visit = False
                        break
                if first_visit:
                    if (state_t, action_t) not in Returns:
                        Returns[(state_t, action_t)] = []
                    Returns[(state_t, action_t)].append(G)
                    self.get_q(state_t)
                    self.Q[state_t][action_t] = np.mean(Returns[(state_t, action_t)])
                    q_vals = self.get_q(state_t)
                    A_optimal = np.argmax(q_vals)
                    for action_choice in range(self.actions):
                        if action_choice == A_optimal:
                            self.policy[state_t][action_choice] = (
                                1 - self.epsilon + (self.epsilon / self.actions)
                            )
                        else:
                            self.policy[state_t][action_choice] = (
                                self.epsilon / self.actions
                            )

        return reward_history


class OffPolicyMC(MCBase):
    def __init__(self, actions, gamma=0.99, epsilon=0.5):
        super().__init__(actions, gamma, epsilon)
        
    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() <= epsilon:
            return random.randint(0, self.actions - 1)

        q_vals = self.get_q(state)
        max_q = np.max(q_vals)
        possible_actions = []

        for action in range(q_vals.shape[0]):
            if q_vals[action] == max_q:
                possible_actions.append(action)

        return random.choice(possible_actions)

    def train(self, env, episodes=1000):
        reward_history = np.zeros(episodes)
        C = {}
        for episode in range(episodes):
            state, _ = env.reset()
            state = self.convert_state(state)
            done = False
            episode_data = []
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.convert_state(next_state)
                done = terminated or truncated
                episode_data.append((state, action, reward))
                reward_history[episode] += reward
                state = next_state

            G = 0
            W = 1
            T = len(episode_data)
            for t in range(T - 1, -1, -1):
                state_t, action_t, reward_t = episode_data[t]
                G = self.gamma * G + reward_t
                if (state_t, action_t) not in C:
                    C[(state_t, action_t)] = 0
                C[(state_t, action_t)] += W
                self.get_q(state_t)
                self.Q[state_t][action_t] += (W / C[(state_t, action_t)]) * (
                    G - self.Q[state_t][action_t]
                )

                best_action = np.argmax(self.get_q(state_t))
                self.policy[state_t] = best_action
                if action_t != best_action:
                    break
                
                b_prob = (1 - self.epsilon) + self.epsilon / self.actions
                
                W *= 1 / b_prob # importance sampling ratio

        return reward_history
