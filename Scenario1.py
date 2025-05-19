import numpy as np
import random
from FourRooms import FourRooms
import sys
import matplotlib.pyplot as plt
import os

# Parameters
EPISODES = 500
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995

# Action space
ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

# Q-table
Q = {}

def get_state(fourRoomsObj):
    x, y = fourRoomsObj.getPosition()
    k = fourRoomsObj.getPackagesRemaining()
    return (x, y, k)

def select_action(state, epsilon):
    if state not in Q:
        Q[state] = np.zeros(len(ACTIONS))
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return np.argmax(Q[state])

def get_reward(cell_type, is_terminal):
    if is_terminal and cell_type > 0:
        return 100  # Package collected
    return -1  # Movement penalty

def main():
    stochastic = '-stochastic' in sys.argv
    fourRoomsObj = FourRooms('simple', stochastic=stochastic)

    # Create log and plot directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    log_file_path = os.path.join("logs", "scenario1_log.txt")
    with open(log_file_path, "w") as log_file:
        episode_rewards = []
        epsilon = EPSILON_START

        for episode in range(EPISODES):
            fourRoomsObj.newEpoch()
            state = get_state(fourRoomsObj)
            total_reward = 0

            while not fourRoomsObj.isTerminal():
                action = select_action(state, epsilon)
                cell_type, new_pos, packages_left, is_terminal = fourRoomsObj.takeAction(action)
                next_state = get_state(fourRoomsObj)
                reward = get_reward(cell_type, is_terminal)

                if next_state not in Q:
                    Q[next_state] = np.zeros(len(ACTIONS))

                best_next_action = np.max(Q[next_state])
                Q[state][action] += ALPHA * (reward + GAMMA * best_next_action - Q[state][action])
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            log_line = f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}\n"
            print(log_line.strip())
            log_file.write(log_line)

    # Save and show path for last run
    fourRoomsObj.showPath(-1, savefig="./plots/scenario1_path.png")

    # Plot rewards
    plt.plot(episode_rewards)
    plt.title("Scenario 1: Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("./plots/scenario1_rewards.png")
    plt.close()

if __name__ == "__main__":
    main()
