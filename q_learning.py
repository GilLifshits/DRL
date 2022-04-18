import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearning:

    def __init__(self, environment, epsilon_greedy, decay_rate, learning_rate, discount_factor):
        self.environment = environment
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.epsilon_greedy = epsilon_greedy
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.GOAL_STATE = 15
        self.steps_per_episode = []
        self.reward_per_episode = []

        self.current_state = env.reset()
        self.environment.render()

    def sample_action_by_epsilon_greedy(self):
        if np.random.binomial(1, p=self.epsilon_greedy):
            return self.environment.action_space.sample()
        return self.q[self.current_state, :].argmax()

    def decay_epsilon_greedy(self):
        self.epsilon_greedy *= self.decay_rate

    def step(self, to_render=True):
        action = self.sample_action_by_epsilon_greedy()
        state_tag, reward, done, info = self.environment.step(action)
        if to_render:
            self.environment.render()
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q[state_tag, :])
        self.q[self.current_state, action] = (1 - self.learning_rate) * self.q[self.current_state, action] + self.learning_rate * target
        self.current_state = state_tag
        return done, reward

    def train(self, max_episode=5000, max_steps=100, to_render=True, steps_to_plot_q=None):
        step_per_episode = 0
        for episode in range(max_episode):
            for step in range(max_steps):
                done, reward = self.step(to_render)
                step_per_episode = step + 1
                if done and self.current_state is not self.GOAL_STATE:
                    step_per_episode = max_steps
                if done:
                    self.reward_per_episode.append(reward)
                    break
                if episode * max_steps + step in steps_to_plot_q:
                    self.plot_q()
            self.steps_per_episode.append(step_per_episode)
            self.decay_epsilon_greedy()
            self.current_state = env.reset()
        return self.q

    def plot_mean_steps_to_goal(self):
        mean_steps = []
        for chunk in np.split(np.array(self.steps_per_episode), 50):
            mean_steps.append(np.average(chunk))
        plt.title("Mean steps to goal")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.scatter(np.arange(100, 5100, 100), mean_steps)
        plt.grid(linestyle='dotted')
        plt.show()

    def plot_reward_per_episode(self):
        mean_rewards = []
        for chunk in np.split(np.array(self.reward_per_episode), 50):
            mean_rewards.append(np.average(chunk))
        plt.title("Mean reward per 100 episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.scatter(np.arange(100, 5100, 100), mean_rewards)
        plt.grid(linestyle='dotted')
        plt.show()

    def plot_q(self):
        actions = ["Left", "Down", "Right", "up"]
        states = np.arange(0, 16)

        fig, ax = plt.subplots()
        ax.imshow(self.q.T)

        # Show all ticks and label them with the respective list entries
        ax.set_yticks(np.arange(len(actions)))
        ax.set_yticklabels(actions)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(actions)):
            for j in range(len(states)):
                 ax.text(j, i, np.round(self.q[j, i], 2), ha="center", va="center", color="w")

        ax.set_title("Q-value table")
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    agent = QLearning(env, epsilon_greedy=0.9, decay_rate=0.999, learning_rate=0.05, discount_factor=0.99)
    q = agent.train(to_render=False, steps_to_plot_q=[500, 2000])
    agent.plot_q()
    agent.plot_mean_steps_to_goal()
    agent.plot_reward_per_episode()


