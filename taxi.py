import gymnasium as gym
from td_learning import SARSA, N_Step_SARSA, SARSA_Lambda
from visualizer import compare_rewards

env = gym.make("Taxi-v3")
eval_env = gym.make("Taxi-v3", render_mode="human")

if __name__ == "__main__":
    sarsa_agent = SARSA(actions=env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.1)
    reward_history_s = sarsa_agent.train(env, episodes=4000)
    path, total_reward = sarsa_agent.best_path(eval_env)
    
    sarsa_n_agent = N_Step_SARSA(
        actions=env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.1, n=1
    )
    reward_history_n = sarsa_n_agent.train(env, episodes=4000)
    path, total_reward = sarsa_n_agent.best_path(eval_env)
    
    compare_rewards(reward_history_s, reward_history_n)
    
    

