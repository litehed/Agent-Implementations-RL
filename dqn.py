import math
import random
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    # Takes number of observations, number of actions, alpha - learning rate,
    # gamma - discount factor, eps start/end/decay - exploration schedule,
    # tau - soft update rate for the target network
    def __init__(
        self,
        n_observations,
        n_actions,
        device=None,
        batch_size=128,
        gamma=0.99,
        eps_start=0.9,
        eps_end=0.01,
        eps_decay=2500,
        tau=0.005,
        alpha=3e-4,
        memory_capacity=10000,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_actions = n_actions
        self.n_observations = n_observations

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=alpha, amsgrad=True
        )
        self.memory = ReplayMemory(memory_capacity)

        self.steps_done = 0

    def convert_state(self, state):
        return torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)

    # Choose action using an epsilon-greedy policy over a decaying epsilon schedule
    def choose_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if sample > eps_threshold:
            return self.best_action(state)
        return torch.tensor(
            [[random.randrange(self.n_actions)]],
            device=self.device,
            dtype=torch.long,
        )

    # Finds the best action with 0 exploration after training is complete
    def best_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Convert batch-array of Transitions to Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # theta' <- tau * theta + (1 - tau) * theta'
    # Updates target network more smoothly than a hard fixed-update
    def soft_update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    # Train the agent in the environment for a given number of episodes
    def train(self, env, episodes=1000, verbose=True):
        reward_history = np.zeros(episodes)
        for episode in range(episodes):
            state, _ = env.reset()
            state = self.convert_state(state)

            for t in count():
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                reward_history[episode] += reward

                next_state = None if terminated else self.convert_state(next_state)
                reward_tensor = torch.tensor([reward], device=self.device)

                self.memory.push(state, action, next_state, reward_tensor)
                state = next_state

                self.optimize_model()
                self.soft_update_target_net()

                if done:
                    if verbose:
                        print(
                            f"Episode {episode + 1}/{episodes} "
                            f"finished after {t + 1} steps"
                        )
                    break

        return reward_history

    # Finds a path with 0 exploration after training is complete
    def best_path(self, env, max_steps=3000):
        state, _ = env.reset()
        state = self.convert_state(state)
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = self.best_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if not done:
                state = self.convert_state(next_state)
            if steps > max_steps:
                break
        return total_reward

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
