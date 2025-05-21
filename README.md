# Reinforcement Learning in the Four-Rooms Domain

This project implements Q-learning agents to solve increasingly complex package collection tasks in a 13x13 Four-Rooms grid-world environment. It is structured into three scenarios, each with specific requirements. All scenarios compare ε-greedy and softmax exploration strategies and identify the best-performing configuration based on average reward.

## Files

- `FourRooms.py` — The environment (do not modify).
- `Scenario1.py` — Agent must collect one package (`scenario='simple'`).
- `Scenario2.py` — Agent must collect four packages (`scenario='multi'`).
- `Scenario3.py` — Agent must collect three packages in order: Red → Green → Blue (`scenario='rgb'`).
- `requirements.txt` — Lists only the required Python packages.
- `logs/` — Contains training logs per episode.
- `plots/` — Reward plots and path images for each scenario.
- `best_plots/` — Final path visualization for the best run in each scenario.
- `REPORT_SCENARIO_1.pdf` — Report detailing comparison for scenario 1


## General Structure

- **Environment**: `FourRooms.py` (not modified)
- **Action Space**: UP, DOWN, LEFT, RIGHT
- **Exploration Strategies**:
  - **ε-greedy**: Random action with probability ε
  - **Softmax**: Probability distribution based on Q-values and temperature τ
- **Logging**: Written to `logs/` for each scenario
- **Plots**:
  - Reward curve of the **best run only**
  - Agent path visualizations saved or displayed
- **Reproducibility**: `random.seed(42)` and `np.random.seed(42)`
- **Hyperparameters**: Explored using `PARAM_GRID` in each script


## Scenario 1: Simple Package Collection

- **Objective**: Collect 1 randomly placed package.
- **Run**: `python Scenario1.py`
- **Enhancements**:
  - Compares both ε-greedy and softmax
  - Evaluates multiple hyperparameter settings
  - Automatically identifies the best-performing config
  - Final reward curve shows **only the best run**
  - Final path is saved or displayed


## Scenario 2: Multiple Package Collection

- **Objective**: Collect 4 packages scattered across the grid.
- **Run**: `python Scenario2.py`
- **Behavior**:
  - Compares strategies and hyperparameters
  - Selects the best configuration based on average reward
  - Saves best reward curve and final path



## Scenario 3: Ordered RGB Package Collection

- **Objective**: Collect three packages in the strict order Red → Green → Blue.
- **Run**: `python Scenario3.py`
- **Behavior**:
  - Early termination on wrong order
  - Evaluates agent precision across exploration methods
  - Selects and saves best-performing run

---

## Scenario 4: Stochastic Movement (20% Noise)

Each script supports a `-stochastic` flag, which introduces a 20% chance of unintended movement. This tests robustness to uncertainty.

**Example**:
python Scenario1.py -stochastic
