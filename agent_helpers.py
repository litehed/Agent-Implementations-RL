from td_learning import (
    SARSA,
    SARSA_Lambda,
    ExpectedSARSA,
    N_Step_SARSA,
    OffPolicy_N_Step_SARSA,
    QLearning,
    DoubleQLearning,
)


EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
TRACE_DECAY = 0.9
N_STEP_SIZE = 3


def train_sarsa_zero(
    train_env,
    eval_env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
):
    sarsa_agent = SARSA(
        actions=train_env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=min_epsilon,
        epsilon_decay=epsilon_decay,
    )
    reward_history_s = sarsa_agent.train(train_env, episodes=episodes)
    path, total_reward = sarsa_agent.best_path(eval_env)
    return reward_history_s, path, total_reward


def train_sarsa_expected(
    train_env,
    eval_env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
):
    expected_sarsa_agent = ExpectedSARSA(
        actions=train_env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=min_epsilon,
        epsilon_decay=epsilon_decay,
    )
    reward_history_expected_sarsa = expected_sarsa_agent.train(
        train_env, episodes=episodes
    )
    path, total_reward = expected_sarsa_agent.best_path(eval_env)
    return reward_history_expected_sarsa, path, total_reward


def train_sarsa_n_step(
    train_env,
    eval_env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
    n_step_size=N_STEP_SIZE,
):
    sarsa_n_step_agent = N_Step_SARSA(
        actions=train_env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=min_epsilon,
        n=n_step_size,
    )
    reward_history_n_step = sarsa_n_step_agent.train(train_env, episodes=episodes)
    path, total_reward = sarsa_n_step_agent.best_path(eval_env)
    return reward_history_n_step, path, total_reward


def train_off_policy_n_step_sarsa(
    train_env,
    eval_env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
    n_step_size=N_STEP_SIZE,
):
    off_policy_n_step_sarsa_agent = OffPolicy_N_Step_SARSA(
        actions=train_env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=min_epsilon,
        n=n_step_size,
    )
    reward_history_off_policy_n_step_sarsa = off_policy_n_step_sarsa_agent.train(
        train_env, episodes=episodes
    )
    path, total_reward = off_policy_n_step_sarsa_agent.best_path(eval_env)
    return reward_history_off_policy_n_step_sarsa, path, total_reward


def train_sarsa_lambda(
    train_env,
    eval_env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
    trace_decay=TRACE_DECAY,
):
    sarsa_lambda_agent = SARSA_Lambda(
        actions=train_env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=min_epsilon,
        trace_decay=trace_decay,
    )
    reward_history_lambda = sarsa_lambda_agent.train(train_env, episodes=episodes)
    path, total_reward = sarsa_lambda_agent.best_path(eval_env)
    return reward_history_lambda, path, total_reward


def train_q_learning(
    train_env,
    eval_env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
):
    q_learning_agent = QLearning(
        actions=train_env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=min_epsilon,
        epsilon_decay=epsilon_decay,
    )
    reward_history_q = q_learning_agent.train(train_env, episodes=episodes)
    path, total_reward = q_learning_agent.best_path(eval_env)
    return reward_history_q, path, total_reward


def train_double_q(
    train_env,
    eval_env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
):
    double_q_agent = DoubleQLearning(
        actions=train_env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=min_epsilon,
        epsilon_decay=epsilon_decay,
    )
    reward_history_double_q = double_q_agent.train(train_env, episodes=episodes)
    path, total_reward = double_q_agent.best_path(eval_env)
    return reward_history_double_q, path, total_reward
