import gymnasium as gym
from td_learning import SARSA, SARSA_Lambda, QLearning

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
eval_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

if __name__ == "__main__":
    RUNS = 30

    sarsa_lambda_results = []
    sarsa_results = []
    q_results = []

    for i in range(RUNS):
        print(f"Run {i + 1}/{RUNS}")

        # SARSA(lambda)
        sarsa_lambda_agent = SARSA_Lambda(
            actions=env.action_space.n,
            alpha=0.08,
            gamma=0.99,
            epsilon=0.1,
            trace_decay=0.8,
        )
        sarsa_lambda_agent.train(env, episodes=10000)

        path, total_reward = sarsa_lambda_agent.best_path(eval_env)
        sarsa_lambda_results.append(total_reward)

        # SARSA(0)
        sarsa_agent = SARSA(
            actions=env.action_space.n, alpha=0.08, gamma=0.99, epsilon=0.1
        )
        sarsa_agent.train(env, episodes=10000)

        path, total_reward = sarsa_agent.best_path(eval_env)
        sarsa_results.append(total_reward)

        # Q-Learning
        q_agent = QLearning(
            actions=env.action_space.n, alpha=0.08, gamma=0.99, epsilon=0.1
        )
        q_agent.train(env, episodes=10000)

        path, total_reward = q_agent.best_path(eval_env)
        q_results.append(total_reward)

    # Summary
    lambda_successes = sum(sarsa_lambda_results)
    sarsa_successes = sum(sarsa_results)
    q_successes = sum(q_results)

    print("\nRESULTS OVER 30 RUNS:")
    print(f"SARSA(lambda) successes: {lambda_successes} / {RUNS}")
    print(f"SARSA(0) successes: {sarsa_successes} / {RUNS}")
    print(f"Q-Learning successes: {q_successes} / {RUNS}")

    if lambda_successes > sarsa_successes and lambda_successes > q_successes:
        print("SARSA(lambda) performed better overall")
    elif sarsa_successes > lambda_successes and sarsa_successes > q_successes:
        print("SARSA(0) performed better overall")
    elif q_successes > lambda_successes and q_successes > sarsa_successes:
        print("Q-Learning performed better overall")
    else:
        print("All agents performed equally well")
