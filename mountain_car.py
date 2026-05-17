import gymnasium as gym
from semi_gradient_sarsa import SemiGradientSARSA

env = gym.make('MountainCar-v0')
eval_env = gym.make('MountainCar-v0', render_mode="human")

if __name__ == "__main__":
    # some of the training vals come from textbook examples
    num_tilings = 8 # must be > 4xdims
    tiles_per_dim = 8 # must be > 4
    iht_size = (num_tilings * tiles_per_dim ** 2) * 8 
    
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
    reward_history = agent.train(env, episodes=800)
    path, total_reward = agent.best_path(eval_env)
    print(f"Semi-Gradient SARSA total reward: {total_reward}")
    print(f"Semi-Gradient SARSA path: {path}")
