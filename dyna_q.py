import numpy as np
import random
from td_learning import TDLBase


class DynaQ(TDLBase):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1, n_planning_steps=5):
        super().__init__(actions, alpha, gamma, epsilon)
        self.model = {}
        self.n_planning_steps = n_planning_steps

    def update(self, state, action, reward, next_state):
        cur_q = self.get_q(state)[action]
        next_q = np.max(self.get_q(next_state))
        cur_q += self.alpha * (reward + self.gamma * next_q - cur_q)
        self.Q[state][action] = cur_q

    def train(self, env, episodes=1000):
        reward_history = np.zeros(episodes)
        for episode in range(episodes):
            state, _ = env.reset()
            state = self.convert_state(state)
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.convert_state(next_state)
                done = terminated or truncated
                reward_history[episode] += reward
                self.update(state, action, reward, next_state)
                self.model[(state, action)] = (reward, next_state)
                for _ in range(self.n_planning_steps):
                    s, a = random.choice(list(self.model.keys()))
                    R, s_next = self.model[(s, a)]
                    self.update(s, a, R, s_next)
                
                state = next_state

        return reward_history
