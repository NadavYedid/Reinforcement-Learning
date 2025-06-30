import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent import QLearningAgent
from learning_env import LearningEnv


def train_and_log_rewards(episodes, alpha, gamma, decay, exp_name):
    """
    *** Train a Q-learning agent and log total rewards per episode ***
    """
    env = LearningEnv()
    state_dims = (env.max_steps + 1, env.num_topics + 1, 5)
    agent = QLearningAgent(
        state_space_dims=state_dims,
        action_space_size=7,
        alpha=alpha,
        gamma=gamma,
        decay_rate=decay
    )

    logs = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        logs.append({
            "Episode": ep + 1,
            "Total_Reward": total_reward,
            "Epsilon": agent.epsilon,
            "Experiment": exp_name
        })
        agent.decay_epsilon()

    # Save simplified policy summary per experiment
    save_policy_summary(agent, exp_name)

    return logs


def save_policy_summary(agent, exp_name):
    """
    *** Extract simplified optimal policy (actions at fatigue=0, topics=0) ***
    and save to CSV for summary analysis.
    """
    summary = []
    for step in range(agent.state_space_dims[0]):
        state = (step, 0, 0)
        best_action = np.argmax(agent.q_table[state])
        summary.append({
            "Experiment": exp_name,
            "Step": step,
            "Best_Action": best_action
        })
    pd.DataFrame(summary).to_csv(f"{exp_name}_policy.csv", index=False)


def plot_experiment(logs, experiment_name):
    """
    *** Generate reward and epsilon plots for a single experiment ***
    """
    df = pd.DataFrame(logs)

    # Plot 1: Total Reward + Moving Avg
    plt.figure(figsize=(10, 5))
    plt.plot(df["Episode"], df["Total_Reward"], alpha=0.3, label="Total Reward")
    df["Moving_Avg"] = df["Total_Reward"].rolling(window=50).mean()
    plt.plot(df["Episode"], df["Moving_Avg"], color="red", label="Moving Avg (window=50)")
    plt.title(f"Q-Learning Performance: {experiment_name}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{experiment_name}_rewards.png")
    plt.close()

    # Plot 2: Epsilon Decay
    plt.figure(figsize=(10, 4))
    plt.plot(df["Episode"], df["Epsilon"], color="purple", label="Epsilon Decay")
    plt.axhline(y=0.01, color="red", linestyle="--", label="Min Epsilon=0.01")
    plt.title(f"Epsilon Decay: {experiment_name}")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{experiment_name}_epsilon.png")
    plt.close()

    # Plot 3: Cumulative Reward
    plt.figure(figsize=(10, 5))
    df["Cumulative_Reward"] = df["Total_Reward"].cumsum()
    plt.plot(df["Episode"], df["Cumulative_Reward"], color="green", label="Cumulative Reward")
    plt.title(f"Cumulative Reward: {experiment_name}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{experiment_name}_cumulative.png")
    plt.close()


def run_multiple_experiments():
    """
    *** Run multiple experiments and generate visualizations ***
    """
    experiments = [
        {
            "name": "Fast_Learner",  # Learns quickly – high alpha
            "alpha": 0.5,
            "gamma": 0.9,
            "decay": 0.99
        },
        {
            "name": "Slow_Learner",  # Learns cautiously – low alpha
            "alpha": 0.1,
            "gamma": 0.9,
            "decay": 0.99
        },
        {
            "name": "Greedy_ShortSighted",  # Prioritizes short-term rewards
            "alpha": 0.3,
            "gamma": 0.5,
            "decay": 0.99
        },
        {
            "name": "Aggressive_Explorer",  # Explores a lot – slow epsilon decay
            "alpha": 0.3,
            "gamma": 0.9,
            "decay": 0.999
        },
        {
            "name": "Conservative_Stable",  # Conservative but stable learning
            "alpha": 0.2,
            "gamma": 0.95,
            "decay": 0.97
        }
    ]

    all_logs = []
    for exp in experiments:
        logs = train_and_log_rewards(
            episodes=1000,
            alpha=exp["alpha"],
            gamma=exp["gamma"],
            decay=exp["decay"],
            exp_name=exp["name"]
        )
        all_logs.extend(logs)
        plot_experiment(logs, exp["name"])

    df = pd.DataFrame(all_logs)
    df.to_csv("results_summary.csv", index=False)
    print("✅ All experiments completed and saved to 'results_summary.csv'")
    print("📊 Plots saved as PNG files for each experiment.")


if __name__ == "__main__":
    run_multiple_experiments()
