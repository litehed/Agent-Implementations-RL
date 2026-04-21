import matplotlib.pyplot as plt
import numpy as np


def visualize_path(grid, path, name=""):
    rows = len(grid)
    cols = len(grid[0])

    display = np.zeros((rows, cols, 3))

    WALL = (0, 0, 0)
    EMPTY = (0.9, 0.9, 0.9)
    PATH = (0.5, 1.0, 0.5)
    START = (0.3, 0.7, 0.3)
    GOAL = (0.9, 0.2, 0.2)
    END = (0.9, 0.3, 0.3)

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == "#":
                display[i][j] = WALL
            elif grid[i][j] == "G":
                display[i][j] = GOAL
            else:
                display[i][j] = EMPTY

    for state in path:
        display[state[0]][state[1]] = PATH

    display[path[0][0]][path[0][1]] = START
    display[path[-1][0]][path[-1][1]] = END

    plt.figure(figsize=(8, 8))
    plt.imshow(display)
    plt.title(f"Final Path {name} (Length: {len(path) - 1})")
    plt.axis("off")
    plt.show(block=False)


def visualize_rewards(rewards):
    plt.figure(figsize=(8, 5))
    plt.plot(rewards)
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.show()


def compare_rewards(rewards_sarsa, rewards_q):
    plt.figure(figsize=(8, 5))
    plt.plot(rewards_sarsa, label="SARSA")
    plt.plot(rewards_q, label="Q-Learning")
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.legend()
    plt.show()
    
def compare_rewards_all(rewards_sarsa, rewards_q, rewards_lambda):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_sarsa, label="SARSA(0)")
    plt.plot(rewards_q, label="SARSA Expected")
    plt.plot(rewards_lambda, label="Q-Learning")
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.legend()
    plt.show()
    
def compare_all_agents(reward_histories, labels):
    plt.figure(figsize=(10, 5))
    for rewards, label in zip(reward_histories, labels):
        plt.plot(rewards, label=label)
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.legend()
    plt.show()
