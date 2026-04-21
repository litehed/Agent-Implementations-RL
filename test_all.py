import gymnasium as gym
from agent_helpers import *
from visualizer import compare_all_agents

EPISODES = 10000

taxi_env = gym.make("Taxi-v3")
taxi_eval_env = gym.make("Taxi-v3", render_mode="human")

cliff_env = gym.make("CliffWalking-v1")
cliff_eval_env = gym.make("CliffWalking-v1", render_mode="human")

frozen_lake_env = gym.make("FrozenLake-v1", is_slippery=True)
frozen_lake_eval_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")

env = taxi_env
eval_env = taxi_eval_env

if __name__ == "__main__":
    print("SARSA(0)")
    reward_history_s, path_s, total_reward_s = train_sarsa_zero(env, eval_env, episodes=EPISODES)
    print("SARSA Expected")
    reward_history_expected_sarsa, path_expected_sarsa, total_reward_expected_sarsa = train_sarsa_expected(env, eval_env, episodes=EPISODES)
    print("SARSA N-Step")
    reward_history_n_step, path_n_step, total_reward_n_step = train_sarsa_n_step(env, eval_env, episodes=EPISODES)
    print("Off-Policy N-Step SARSA")
    reward_history_n, path_n, total_reward_n = train_off_policy_n_step_sarsa(env, eval_env, episodes=EPISODES)
    print("SARSA(lambda)")
    reward_history_lambda, path_lambda, total_reward_lambda = train_sarsa_lambda(env, eval_env, episodes=EPISODES)
    print("Q-Learning")
    reward_history_q, path_q, total_reward_q = train_q_learning(env, eval_env, episodes=EPISODES)
    print("Double Q-Learning")
    reward_history_dq, path_dq, total_reward_dq = train_double_q(env, eval_env, episodes=EPISODES)

    compare_all_agents(
        [
            reward_history_s,
            reward_history_expected_sarsa,
            reward_history_n_step,
            reward_history_n,
            reward_history_lambda,
            reward_history_q,
            reward_history_dq,
        ],
        [
            "SARSA(0)",
            "SARSA Expected",
            "SARSA N-Step",
            "Off-Policy N-Step SARSA",
            "SARSA(lambda)",
            "Q-Learning",
            "Double Q-Learning",
        ],
    )
