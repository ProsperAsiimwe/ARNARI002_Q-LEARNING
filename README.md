# Reinforcement Learning in the Four-Rooms Domain

This project implements Q-learning agents to solve increasingly complex package collection tasks in a 13x13 Four-Rooms grid-world environment. It is structured into three scenarios, each with specific requirements. All scenarios are executed via Python scripts and support ε-greedy and softmax exploration strategies, with results logged and visualize

## Files

- `FourRooms.py` — The environment (do not modify).
- `Scenario1.py` — Agent must collect one package (`scenario='simple'`).
- `Scenario2.py` — Agent must collect four packages (`scenario='multi'`).
- `Scenario3.py` — Agent must collect three packages in order: Red → Green → Blue (`scenario='rgb'`).
- `requirements.txt` — Lists only the required Python packages.
- `logs/` — Contains training logs for each scenario.
- `plots/` — Contains reward curves and agent paths.

## General Structure

Environment: FourRooms.py (Already provided, I didn't modify this one in any way)

Action Space: UP, DOWN, LEFT, RIGHT

Exploration Strategies:

ε-greedy: Random action with probability ε

Softmax: Probability distribution based on Q-values

Logging: Per scenario in logs/

Plots: Reward curves and agent paths in plots/

Reproducibility: Controlled via SEED = 42

## Scenario 1: Simple Package Collection
The agent must collect 1 package placed randomly in the grid.
Script: python Scenario1.py

The script compares softmax and ε-greedy learning curves and paths, prints and logs performance, and saves visualizations.

## Scenario 2: Multiple Package Collection
The agent must collect 4 packages randomly scattered across the environment.

Script: python Scenario2.py

The script compares exploration methods across parameter configurations and shows learning progression in more complex environment.

## Scenario 3: Ordered RGB Package Collection
The agent must collect three packages in the strict order: Red → Green → Blue. Picking the wrong package ends the run.

Script: python Scenario3.py

This scenario tests the agent’s precision and learning under strict rules. The penalty system encourages careful strategy learning, and both strategies are compared.

## Running in Stochastic Mode (Scenario 4)
Each scenario supports an optional -stochastic flag which this enables 20% chance of unintended transitions to test agent robustness.

#### Visualization and Logging Output
Each scenario produces:
- Learning curve plots: with legends identifying each strategy and parameter combination
- Path visualizations: final agent trajectory per strategy
- Logs: reward per episode, exploration parameters, and episode summaries

## Reproducibility and Experimentation
All scripts are seeded: random.seed(42) and np.random.seed(42)

Hyperparameter combinations are looped using PARAM_GRID

Final output is saved to disk and also printed to terminal for live feedback



## SUMMARY:

python Scenario1.py
python Scenario2.py
python Scenario3.py

## For stochastic movement (20% chance of unintended move):

python Scenario1.py -stochastic
python Scenario2.py -stochastic
python Scenario3.py -stochastic

## Install required packages using (Only numpy and matplotlib are required):

pip install -r requirements.txt
