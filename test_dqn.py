import gymnasium as gym
import torch

from dqn import DQNAgent

env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1", render_mode="human")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPISODES = 250 if device.type == "cuda" else 50
    EVAL_RUNS = 5

    agent = DQNAgent(
        n_observations=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        device=device,
    )

    reward_history = agent.train(env, episodes=EPISODES)

    print("\nEvaluating trained agent:")
    for i in range(EVAL_RUNS):
        total_reward = agent.best_path(eval_env)
        print(f"Eval episode {i + 1}/{EVAL_RUNS}: total reward = {total_reward}")

    env.close()
    eval_env.close()
