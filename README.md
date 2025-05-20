# Reinforcement Learning in the Four-Rooms Domain

This project implements Q-learning agents for three scenarios of increasing complexity in a 13x13 Four-Rooms grid environment.

## Files

- `FourRooms.py` — The environment (do not modify).
- `Scenario1.py` — Agent must collect one package (`scenario='simple'`).
- `Scenario2.py` — Agent must collect four packages (`scenario='multi'`).
- `Scenario3.py` — Agent must collect three packages in order: Red → Green → Blue (`scenario='rgb'`).
- `requirements.txt` — Lists only the required Python packages.
- `logs/` — Contains training logs for each scenario.
- `plots/` — Contains reward curves and agent paths.

## Running the Scenarios

python Scenario1.py
python Scenario2.py
python Scenario3.py

## For stochastic movement (20% chance of unintended move):

python Scenario1.py -stochastic
python Scenario2.py -stochastic
python Scenario3.py -stochastic

## Install required packages using (Only numpy and matplotlib are required):

pip install -r requirements.txt
