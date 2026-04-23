import gymnasium as gym
from td_learning import SARSA, N_Step_SARSA, OnPolicyMC
from visualizer import visualize_rewards

env = gym.make("Taxi-v3")
eval_env = gym.make("Taxi-v3", render_mode="human")

if __name__ == "__main__":
    onpolicyMC = OnPolicyMC(actions=env.action_space.n)
    reward_history_onpolicyMC = onpolicyMC.train(env, episodes=30000)
    wait = input("Training Finished. Press Enter")
    path, bestreward = onpolicyMC.best_path(eval_env)
    
    visualize_rewards(reward_history_onpolicyMC)
    
    
    

