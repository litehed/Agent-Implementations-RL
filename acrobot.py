import gymnasium as gym
from semi_gradient_sarsa import SemiGradientSARSA

env = gym.make('Acrobot-v1')
eval_env = gym.make('Acrobot-v1', render_mode="human")

if __name__ == "__main__":
    state_dims = len(env.observation_space.low)  # 6
    num_tilings = 32  # power of 2 > 4xdims
    tiles_per_dim = 8
    iht_size = 16777216  # power of 2 > num_tilings * tiles_per_dim^state_dims

    agent = SemiGradientSARSA(
        actions=env.action_space.n,
        iht_size=iht_size,
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
        alpha=0.5,
        gamma=1.0,
        epsilon=0.1,
    )
    reward_history = agent.train(env, episodes=1000)
    path, total_reward = agent.best_path(eval_env)
    print(f"Semi-Gradient SARSA total reward: {total_reward}")
    print(f"Semi-Gradient SARSA path: {path}")
