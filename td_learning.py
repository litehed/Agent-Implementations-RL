import numpy as np
import random


class TDLBase:
    # Takes number of actions, alpha - learning rate, gamma - discount factor,
    # epsilon - exploration rate, epsilon_min - minimum exploration rate, epsilon_decay - decay rate for exploration
    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = {}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def convert_state(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

    def get_q(self, state):
        state = self.convert_state(state)
        if state not in self.Q:
            self.Q[state] = np.zeros(self.actions)
        return self.Q[state]

    # Choose action using epsilon-greedy policy
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

    # Finds a path with 0 exploration after training is complete
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
            done = terminated or truncated
            state = next_state
            best_reward += reward
            path.append(state)
            if len(path) > max_steps:
                break
        return path, best_reward


# On-Policy
class SARSA(TDLBase):
    # Update Q-value using SARSA update rule
    def update(self, state, action, reward, next_state, next_action):
        cur_q = self.get_q(state)[action]
        next_q = self.get_q(next_state)[next_action]
        cur_q += self.alpha * (reward + self.gamma * next_q - cur_q)
        self.Q[state][action] = cur_q

    # Train the agent in the environment for a given number of episodes
    def train(self, env, episodes=1000):
        reward_history = np.zeros(episodes)
        for episode in range(episodes):
            state, _ = env.reset()
            state = self.convert_state(state)
            action = self.choose_action(state)
            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.convert_state(next_state)
                done = terminated or truncated
                next_action = self.choose_action(next_state)
                self.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                reward_history[episode] += reward

            self.decay_epsilon()

        return reward_history


class ExpectedSARSA(SARSA):
    def update(self, state, action, reward, next_state, next_action=None):
        cur_q = self.get_q(state)[action]
        next_qs = self.get_q(next_state)
        expected_next_q = 0
        for a in range(self.actions):
            action_prob = self.epsilon / self.actions
            if a == np.argmax(next_qs):
                action_prob += 1 - self.epsilon
            expected_next_q += action_prob * next_qs[a]
        cur_q += self.alpha * (reward + self.gamma * expected_next_q - cur_q)
        self.Q[state][action] = cur_q


class N_Step_SARSA(SARSA):
    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        n=3,
    ):
        super().__init__(actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
        self.n = n

    def update_n_step(self, state, action, G):
        cur_q = self.get_q(state)[action]
        self.Q[state][action] = cur_q + self.alpha * (G - cur_q)

    def train(self, env, episodes=1000):
        reward_history = np.zeros(episodes)
        for episode in range(episodes):
            S = []
            A = []
            R = []

            state, _ = env.reset()
            S.append(self.convert_state(state))
            A.append(self.choose_action(state))
            R.append(0)

            T = float("inf")
            t = 0
            done = False

            while True:
                if t < T:
                    next_state, reward, terminated, truncated, _ = env.step(A[t])
                    next_state = self.convert_state(next_state)
                    done = terminated or truncated
                    R.append(reward)
                    S.append(next_state)

                    if done:
                        T = t + 1
                    else:
                        next_action = self.choose_action(next_state)
                        A.append(next_action)

                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau, min(tau + self.n + 1, T)):
                        G += (self.gamma ** (i - tau - 1)) * R[i]

                    if tau + self.n < T:
                        G += (
                            self.gamma**self.n
                            * self.get_q(S[tau + self.n])[A[tau + self.n]]
                        )

                    self.update_n_step(S[tau], A[tau], G)

                if tau == T - 1:
                    break

                t += 1

            reward_history[episode] = sum(R)
            self.decay_epsilon()
        return reward_history


# TD(lambda) SARSA, mixes monte carlo and TD(0) by using eligibility traces to update
class SARSA_Lambda(SARSA):
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.1, trace_decay=0.9):
        super().__init__(actions, alpha, gamma, epsilon)
        self.trace_decay = trace_decay  # lambda
        self.e = {}  # eligibility traces

    def get_e(self, state):
        state = self.convert_state(state)
        if state not in self.e:
            self.e[state] = np.zeros(self.actions)
        return self.e[state]

    def decay_traces(self):
        for state in self.e:
            self.e[state] *= self.gamma * self.trace_decay

    def update(self, state, action, reward, next_state, next_action):
        td_error = (
            reward
            + self.gamma * self.get_q(next_state)[next_action]
            - self.get_q(state)[action]
        )
        for t_state in self.e:
            self.Q.setdefault(t_state, np.zeros(self.actions))
            self.Q[t_state] += self.alpha * self.e[t_state] * td_error

    def train(self, env, episodes=1000):
        reward_history = np.zeros(episodes)
        for episode in range(episodes):
            self.e = {}
            state, _ = env.reset()
            state = self.convert_state(state)
            action = self.choose_action(state)
            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.convert_state(next_state)
                done = terminated or truncated
                next_action = self.choose_action(next_state)
                self.get_e(state)[action] += 1
                self.update(state, action, reward, next_state, next_action)
                self.decay_traces()

                state = next_state
                action = next_action
                reward_history[episode] += reward

            self.decay_epsilon()

        return reward_history


# Off-Policy
class QLearning(TDLBase):
    # Update Q-value using Q-Learning update rule
    def update(self, state, action, reward, next_state):
        cur_q = self.get_q(state)[action]
        next_q = np.max(self.get_q(next_state))
        cur_q += self.alpha * (reward + self.gamma * next_q - cur_q)
        self.Q[state][action] = cur_q

    # Train the agent in the environment for a given number of episodes
    def train(self, env, episodes=1000):
        reward_history = np.zeros(episodes)
        for episode in range(episodes):
            state, _ = env.reset()
            state = self.convert_state(state)
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.update(state, action, reward, next_state)
                state = next_state
                reward_history[episode] += reward

            self.decay_epsilon()

        return reward_history


class DoubleQLearning(QLearning):
    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        super().__init__(actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
        self.Q2 = {}

    def get_q2(self, state):
        state = self.convert_state(state)
        if state not in self.Q2:
            self.Q2[state] = np.zeros(self.actions)
        return self.Q2[state]

    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() <= epsilon:
            return random.randint(0, self.actions - 1)

        combined_qs = self.get_q(state) + self.get_q2(state)
        max_q = np.max(combined_qs)
        possible_actions = []

        for action in range(combined_qs.shape[0]):
            if combined_qs[action] == max_q:
                possible_actions.append(action)

        return random.choice(possible_actions)

    def update(self, state, action, reward, next_state):
        if random.random() < 0.5:
            cur_q = self.get_q(state)[action]
            next_action = np.argmax(self.get_q(next_state))
            next_q = self.get_q2(next_state)[next_action]
            cur_q += self.alpha * (reward + self.gamma * next_q - cur_q)
            self.Q[state][action] = cur_q
        else:
            cur_q = self.get_q2(state)[action]
            next_action = np.argmax(self.get_q2(next_state))
            next_q = self.get_q(next_state)[next_action]
            cur_q += self.alpha * (reward + self.gamma * next_q - cur_q)
            self.Q2[state][action] = cur_q
