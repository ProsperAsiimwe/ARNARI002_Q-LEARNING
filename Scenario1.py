import numpy as np
import random
from FourRooms import FourRooms
import sys
import matplotlib.pyplot as plt
import os
from math import exp

# Parameters
EPISODES = 500
EPSILON_START = 1.0
EPSILON_MIN = 0.1
SEED = 42  # Reproducibility

# Action space
ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

# Hyperparameter combinations to try
PARAM_GRID = [
    {"alpha": 0.1, "gamma": 0.95, "epsilon_decay": 0.99},
    {"alpha": 0.05, "gamma": 0.99, "epsilon_decay": 0.995},
    {"alpha": 0.2, "gamma": 0.90, "epsilon_decay": 0.98},
]

# Q-table
Q = {}

def softmax_selection(state, tau):
    if state not in Q:
        Q[state] = np.zeros(len(ACTIONS))
    preferences = Q[state] / tau
    max_pref = np.max(preferences)
    exp_prefs = np.exp(preferences - max_pref)
    probs = exp_prefs / np.sum(exp_prefs)
    return np.random.choice(ACTIONS, p=probs)

def epsilon_greedy_selection(state, epsilon):
    if state not in Q:
        Q[state] = np.zeros(len(ACTIONS))
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return np.argmax(Q[state])

def get_state(env):
    x, y = env.getPosition()
    k = env.getPackagesRemaining()
    return (x, y, k)

def get_reward(cell_type, is_terminal):
    if is_terminal and cell_type > 0:
        return 100
    return -1

def train_strategy(strategy, alpha, gamma, epsilon_decay, log_file):
    global Q
    Q = {}
    epsilon = EPSILON_START
    tau = 1.0
    rewards = []

    env = FourRooms('simple', stochastic='-stochastic' in sys.argv)

    for episode in range(EPISODES):
        env.newEpoch()
        state = get_state(env)
        total_reward = 0

        while not env.isTerminal():
            action = softmax_selection(state, tau) if strategy == 'softmax' else epsilon_greedy_selection(state, epsilon)
            cell_type, _, _, is_terminal = env.takeAction(action)
            next_state = get_state(env)
            reward = get_reward(cell_type, is_terminal)

            if next_state not in Q:
                Q[next_state] = np.zeros(len(ACTIONS))

            best_next_action = np.max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * best_next_action - Q[state][action])
            state = next_state
            total_reward += reward

        epsilon = max(EPSILON_MIN, epsilon * epsilon_decay)
        tau = max(0.1, tau * 0.99)
        rewards.append(total_reward)

        log_line = f"[{strategy.upper()}] Ep {episode+1}, Reward: {total_reward}, Eps: {epsilon:.3f}, Tau: {tau:.3f}"
        print(log_line)
        log_file.write(log_line + "\n")

    return rewards, env

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("best_plots", exist_ok=True)

    log_file_path = os.path.join("logs", "scenario1_comparison_log.txt")
    with open(log_file_path, "w") as log_file:
        print("Scenario 1: ε-greedy vs Softmax Exploration Comparison")
        log_file.write("Scenario 1: ε-greedy vs Softmax Exploration Comparison\n")
        print(f"Using seed: {SEED}\n")
        log_file.write(f"Using seed: {SEED}\n\n")

        best_avg_reward = float('-inf')
        best_label = None
        best_rewards = []
        best_env = None

        for params in PARAM_GRID:
            alpha, gamma, decay = params['alpha'], params['gamma'], params['epsilon_decay']
            for strategy in ['epsilon_greedy', 'softmax']:
                label = f"{strategy}_a{alpha}_g{gamma}_d{decay}"
                print(f"\nRunning strategy: {label}")
                log_file.write(f"\nRunning strategy: {label}\n")
                rewards, env = train_strategy(strategy, alpha, gamma, decay, log_file)
                plt.plot(rewards, label=label)
                env.showPath(-1, savefig=f"./plots/scenario_1_{label}_path.png")

                avg_last_50 = np.mean(rewards[-50:])
                if avg_last_50 > best_avg_reward:
                    best_avg_reward = avg_last_50
                    best_label = label
                    best_rewards = rewards
                    best_env = env

        # Highlight best run
        plt.plot(best_rewards, label=f"BEST: {best_label}", linewidth=3, linestyle='--')
        best_env.showPath(-1, savefig=f"./best_plots/scenario_1_best_run_path.png")

        plt.title("Scenario 1: Reward Comparison")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("./plots/scenario1_comparison_rewards.png", bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()
