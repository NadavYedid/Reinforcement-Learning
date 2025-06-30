# Q-Learning Study Agent – README

This project implements a Q-learning agent designed to learn optimal study strategies over time using reinforcement learning.

---

## 📁 File Overview

### 1. `learning_env.py` – The Learning Environment
This file defines the **rules of the simulation**:
- Number of steps per episode
- Valid actions (e.g., study, rest, review)
- How rewards are given for each action
- State transitions
- Functions:
  - `reset()` – initialize a new episode
  - `step(action)` – perform action and get new state + reward

> **Summary:** This is the "world" where the agent lives and learns.

---

### 2. `agent.py` – The Q-Learning Agent
This file defines the **Q-learning algorithm** used by the agent:
- Maintains a Q-table of state-action values
- Selects actions using **epsilon-greedy** policy (balancing exploration and exploitation)
- Updates Q-values based on experience
- Functions:
  - `select_action(state)` – chooses an action
  - `update_q(...)` – updates Q-table using Q-learning update rule
  - `decay_epsilon()` – reduces exploration over time

> **Summary:** This is the learning brain of the agent.

---

### 3. `run_experiments.py` – Main Training & Evaluation Script
This script runs **multiple experiments** using different hyperparameter configurations:
- Trains the agent for 1000 episodes in each setup
- Logs rewards, epsilon values, and Q-table
- Generates and saves 3 plots per experiment:
  1. Total rewards per episode (+ moving average)
  2. Epsilon decay over time
  3. Cumulative reward over episodes
- Extracts and saves the **final optimal policy** from the Q-table

> **Summary:** This file controls the full experiment pipeline, from training to results visualization.

---

## ✅ Output Files
- `results_summary.csv` – Logs of all training runs (per episode)
- `*_rewards.png`, `*_epsilon.png`, `*_cumulative.png` – Visual performance plots
- `*_policy.csv` – Final optimal policy table (best actions per step)

---

## 🔧 Dependencies
Make sure to install required libraries:
```bash
pip install numpy pandas matplotlib
```

---

## 📌 Run the Project
Run the main script to start all experiments:
```bash
python run_experiments.py
```
