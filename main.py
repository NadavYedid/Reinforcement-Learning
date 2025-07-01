from learning_env import LearningEnv
from agent import train_agent
import numpy as np

def run_random_agent():
    """
    *** Run an episode with a random agent to test the environment ***
    The agent selects random actions (0-6), including resting, reviewing, and studying topics.
    Useful for validating environment transitions and reward structure.
    """
    env = LearningEnv()
    state = env.reset()
    done = False

    while not done:
        action = np.random.choice(7)  # 0-6 possible actions
        next_state, reward, done = env.step(action)
        state = next_state
        env.render()
        print(f"Action: {action}, Reward: {reward}, State: {state}\n")

def run_q_learning():
    """
    *** Train the Q-learning agent and display both detailed and simplified optimal policy ***
    The agent learns over multiple episodes how to act optimally in the environment.
    """
    agent = train_agent(episodes=1000, verbose=True)

    print("\n--- Full Optimal Policy (All states) ---")
    agent.print_best_policy()

    print("\n--- Simplified Optimal Policy (Step-by-step strategy) ---")
    agent.print_simplified_best_policy()

    return agent

if __name__ == "__main__":
    # To test random behavior of the environment:
    # run_random_agent()

    # To train agent and evaluate learned policy:
    run_q_learning()
