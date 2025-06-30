import numpy as np
from learning_env import LearningEnv

class QLearningAgent:
    """
    *** A Q-learning agent that uses epsilon-greedy policy with epsilon decay. ***

    Features:
        - Learns optimal policies through Q-table updates
        - Supports multiple actions, including rest, short rest, review, and study new topics
    """

    def __init__(self, state_space_dims, action_space_size,
                 alpha=0.1, gamma=0.9, epsilon=1.0,
                 min_epsilon=0.01, decay_rate=0.995):
        """*** Initialize the Q-learning agent parameters and Q-table ***"""
        self.state_space_dims = state_space_dims
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.action_space = action_space_size
        self.q_table = np.zeros(state_space_dims + (action_space_size,))

    def select_action(self, state):
        """
        *** Select an action using epsilon-greedy policy ***

        Args:
            state (tuple): Current environment state

        Returns:
            int: Selected action
        """
        state = tuple(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def update_q(self, state, action, reward, next_state, done):
        """
        *** Update Q-table values using the Bellman equation ***

        Args:
            state (tuple): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (tuple): State after taking action
            done (bool): Episode completion status
        """
        state, next_state = tuple(state), tuple(next_state)
        max_next_q = 0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """*** Decay the epsilon value after each episode ***"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def print_best_policy(self):
        """*** Display optimal policy learned by the agent clearly and concisely ***"""
        actions_meaning = {
            0: "Full Rest", 1: "Short Rest", 2: "Review Material",
            3: "Study Topic 1", 4: "Study Topic 2",
            5: "Study Topic 3", 6: "Study Topic 4"
        }

        print("\nOptimal Policy per State (Only learning/review actions shown):")
        for step in range(self.q_table.shape[0]):
            for topics in range(self.q_table.shape[1]):
                for fatigue in range(self.q_table.shape[2]):
                    state = (step, topics, fatigue)
                    best_action = np.argmax(self.q_table[state])
                    if best_action > 1:  # Print only meaningful actions (no rest)
                        action_desc = actions_meaning[best_action]
                        print(f"Step:{step}, Topics:{topics}, Fatigue:{fatigue} → {action_desc}")

    def print_simplified_best_policy(self):
        """
        *** Print a simplified optimal policy summary ***
        Displays only the optimal recommended action for each initial step,
        starting from zero topics learned and no fatigue.
        """
        actions_meaning = {
            0: "Full Rest",
            1: "Short Rest",
            2: "Review Material",
            3: "Study Topic 1",
            4: "Study Topic 2",
            5: "Study Topic 3",
            6: "Study Topic 4"
        }

        print("\nSimplified Optimal Policy Summary:")
        state = (0, 0, 0)  # initial state: (step=0, topics learned=0, fatigue=0)

        for step in range(self.q_table.shape[0]):
            best_action = np.argmax(self.q_table[state])
            action_desc = actions_meaning[best_action]
            print(f"Step {step}: {action_desc}")

            # Simulate state transition according to chosen action
            next_step = state[0] + 1
            next_topics = state[1]
            next_fatigue = state[2]

            if best_action == 0:  # Full Rest
                next_fatigue = max(0, next_fatigue - 2)
            elif best_action == 1:  # Short Rest
                next_fatigue = max(0, next_fatigue - 1)
            elif best_action >= 3:  # Study topic
                next_topics = min(self.q_table.shape[1]-1, next_topics + 1)
                next_fatigue = min(self.q_table.shape[2]-1, next_fatigue + 1)
            elif best_action == 2:  # Review
                next_fatigue = min(self.q_table.shape[2]-1, next_fatigue + 1)

            state = (next_step, next_topics, next_fatigue)


def train_agent(episodes=1000, verbose=True):
    """*** Train the Q-learning agent ***"""
    env = LearningEnv()
    state_dims = (env.max_steps + 1, env.num_topics + 1, env.max_fatigue + 1)
    agent = QLearningAgent(state_space_dims=state_dims, action_space_size=7)

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

        agent.decay_epsilon()

        if verbose and (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

    return agent
