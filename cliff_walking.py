import gymnasium as gym
from agent_helpers import train_off_policy_n_step_sarsa, train_q_learning, train_sarsa_zero
from visualizer import compare_rewards_all, compare_rewards

env = gym.make("CliffWalking-v1")
eval_env = gym.make("CliffWalking-v1", render_mode="human")

if __name__ == "__main__":
    
    reward_history_s, path_s, total_reward_s = train_sarsa_zero(env, eval_env)
    reward_history_n, path_n, total_reward_n = train_off_policy_n_step_sarsa(env, eval_env)
    reward_history_lambda, path_lambda, total_reward_lambda = train_q_learning(env, eval_env)
    
    print(f"SARSA(0) Total Reward: {total_reward_s}")
    print(f"Off-Policy N-Step SARSA Total Reward: {total_reward_n}")
    print(f"Q-Learning Total Reward: {total_reward_lambda}")
    
    compare_rewards_all(reward_history_s, reward_history_n, reward_history_lambda)
    
    

