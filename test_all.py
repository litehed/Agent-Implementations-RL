import gymnasium as gym
from agent_helpers import *

env = gym.make("Taxi-v3")
eval_env = gym.make("Taxi-v3", render_mode="human")

if __name__ == "__main__":
    print("SARSA(0)")
    reward_history_s, path_s, total_reward_s = train_sarsa_zero(env, eval_env)
    print("SARSA Expected")
    reward_history_expected_sarsa, path_expected_sarsa, total_reward_expected_sarsa = train_sarsa_expected(env, eval_env)
    print("SARSA N-Step")
    reward_history_n_step, path_n_step, total_reward_n_step = train_sarsa_n_step(env, eval_env)
    print("Off-Policy N-Step SARSA")
    reward_history_n, path_n, total_reward_n = train_off_policy_n_step_sarsa(env, eval_env)
    print("SARSA(lambda)")
    reward_history_lambda, path_lambda, total_reward_lambda = train_sarsa_lambda(env, eval_env)
    print("Q-Learning")
    reward_history_q, path_q, total_reward_q = train_q_learning(env, eval_env)
    print("Double Q-Learning")
    reward_history_dq, path_dq, total_reward_dq = train_double_q(env, eval_env)
    
    