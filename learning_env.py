class LearningEnv:
    """
    *** A reinforcement learning environment simulating a student's daily study schedule ***

    Actions:
        0 - Full Rest (substantial fatigue reduction)
        1 - Short Rest (moderate fatigue reduction)
        2 - Review Material (reinforce already studied topics)
        3-6 - Study New Topics (Topics 1-4)

    Tracks:
        - Topics learned during the day
        - Current fatigue level (0 to max_fatigue)
        - Steps taken (time units)

    Rewards:
        - New topic learned: +1.0
        - Reviewing material: +0.5 (if topics have been learned already)
        - Short Rest: +0.2 (if fatigued)
        - Full Rest: +0.5 (if fatigued)
        - Penalty for studying while severely fatigued: -0.5

    The episode ends after reaching the maximum number of steps.
    """

    def __init__(self):
        """*** Initialize the learning environment parameters ***"""
        self.num_topics = 4
        self.max_steps = 5
        self.max_fatigue = 5
        self.reset()

    def reset(self):
        """*** Reset environment to initial conditions at the start of an episode ***"""
        self.current_step = 0
        self.topics_learned = set()
        self.fatigue = 0
        return self._get_state()

    def step(self, action):
        """
        *** Take an action and return the resulting state, reward, and done status ***

        Args:
            action (int): Chosen action by the agent.

        Returns:
            tuple: (next_state, reward, done)
        """
        reward = 0.0

        if action == 0:  # Full Rest
            if self.fatigue > 0:
                reward = 0.5
            self.fatigue = max(0, self.fatigue - 2)

        elif action == 1:  # Short Rest
            if self.fatigue > 0:
                reward = 0.2
            self.fatigue = max(0, self.fatigue - 1)

        elif action == 2:  # Review Material
            if self.topics_learned:
                reward = 0.5
            if self.fatigue >= 3:
                reward -= 0.5
            self.fatigue = min(self.max_fatigue, self.fatigue + 1)

        elif 3 <= action <= 6:  # Study New Topic
            topic = action - 2
            if topic not in self.topics_learned:
                reward = 1.0
                self.topics_learned.add(topic)
            if self.fatigue >= 3:
                reward -= 0.5  # penalty for severe fatigue
            self.fatigue = min(self.max_fatigue, self.fatigue + 1)

        else:
            raise ValueError("Invalid action provided.")

        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_state(), reward, done

    def render(self):
        """*** Display the current state of the environment ***"""
        action_meaning = {
            0: "Full Rest",
            1: "Short Rest",
            2: "Review Material",
            3: "Study Topic 1",
            4: "Study Topic 2",
            5: "Study Topic 3",
            6: "Study Topic 4"
        }
        print(f"Step: {self.current_step} | Topics Learned: {self.topics_learned} | Fatigue Level: {self.fatigue}")

    def _get_state(self):
        """
        *** Get the current state as a tuple ***

        Returns:
            tuple: (current_step, num_topics_learned, fatigue_level)
        """
        return self.current_step, len(self.topics_learned), self.fatigue
