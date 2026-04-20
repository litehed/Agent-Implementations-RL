import gymnasium as gym
from td_learning import SARSA, QLearning, SARSA_Lambda
from visualizer import compare_rewards_all

env = gym.make('Blackjack-v1', natural=False, sab=False)
eval_env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")

if __name__ == "__main__":
    sarsa_agent = SARSA(actions=env.action_space.n, alpha=0.1, gamma=1.0, epsilon=0.2)
    reward_history_s = sarsa_agent.train(env, episodes=1000000)
    path, total_reward = sarsa_agent.best_path(eval_env)
    print(f"SARSA total reward: {total_reward}")
    print(f"SARSA path: {path}")
    
    # q_learning_agent = QLearning(actions=env.action_space.n, alpha=0.1, gamma=1.0, epsilon=0.2)
    # reward_history_q = q_learning_agent.train(env, episodes=1000000)
    # path, total_reward = q_learning_agent.best_path(eval_env)
    
    # sarsa_lambda_agent = SARSA_Lambda(actions=env.action_space.n, alpha=0.1, gamma=1.0, epsilon=0.2, trace_decay=0.9)
    # reward_history_lambda = sarsa_lambda_agent.train(env, episodes=1000000)
    # path, total_reward = sarsa_lambda_agent.best_path(eval_env)
    
    # compare_rewards_all(reward_history_s, reward_history_q, reward_history_lambda)
    
    

