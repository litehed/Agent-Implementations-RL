import gymnasium as gym
from td_learning import DoubleQLearning, QLearning
from visualizer import compare_rewards_all, compare_rewards
import numpy as np
env = gym.make("CliffWalking-v1")
eval_env = gym.make("CliffWalking-v1", render_mode="human")

if __name__ == "__main__":
    # sarsa_agent = SARSA(actions=env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.1)
    # reward_history_s = sarsa_agent.train(env, episodes=2000)
    # path, total_reward = sarsa_agent.best_path(eval_env)
    
    # n_sarsa = N_Step_SARSA(actions=env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.1, n=4)
    # reward_history_n = n_sarsa.train(env, episodes=2000)
    # path, total_reward = n_sarsa.best_path(eval_env)
    
    DoubleQLearning_agent = DoubleQLearning(actions=env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.1)
    reward_history_double_q = DoubleQLearning_agent.train(env, episodes=2000)
    path, total_reward = DoubleQLearning_agent.best_path(eval_env)
    
    QLearning_agent = QLearning(actions=env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.1)
    reward_history_q = QLearning_agent.train(env, episodes=2000)
    path, total_reward = QLearning_agent.best_path(eval_env)
    
    print("Double Q-Learning vs Q-Learning:")
    print(f"Double Q-Learning: {np.mean(reward_history_double_q)}")
    print(f"Q-Learning: {np.mean(reward_history_q)}")
    
    compare_rewards(reward_history_double_q, reward_history_q)
    
    # sarsa_lambda_agent = SARSA_Lambda(actions=env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.1, trace_decay=0.9)
    # reward_history_lambda = sarsa_lambda_agent.train(env, episodes=2000)
    # path, total_reward = sarsa_lambda_agent.best_path(eval_env)
    
    # compare_rewards_all(reward_history_s, reward_history_n, reward_history_lambda)
    
    

