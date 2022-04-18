from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Experiences:

    def __init__(self, experiences_size):
        self.state = None
        self.state_tag = None
        self.action = None
        self.reward = None
        self.done = None
        self.experiences_size = experiences_size

    def add_experience(self, current_state, action, reward, state_tag, done):
        if self.state is None:
            self.state = np.array([current_state])
            self.action = np.array([action])
            self.reward = np.array([reward])
            self.state_tag = np.array([state_tag])
            self.done = np.array([done])
        else:
            self.state = np.vstack((self.state, current_state))
            self.action = np.vstack((self.action, action))
            self.reward = np.vstack((self.reward, reward))
            self.state_tag = np.vstack((self.state_tag, state_tag))
            self.done = np.vstack((self.done, done))
        self.regulate_queue_size()

    def regulate_queue_size(self):
        if self.state.shape[0] > self.experiences_size:
            self.state = self.state[1:, :]
            self.action = self.action[1:]
            self.reward = self.reward[1:]
            self.state_tag = self.state_tag[1:, :]
            self.done = self.done[1:]


class DeepQLearning:

    def __init__(self, environment, epsilon_greedy, decay_rate, learning_rate, discount_factor, batch_size,
                 experiences_size, q_update_freq, tensorboard):
        # Hyper-parameters
        self.epsilon_greedy = epsilon_greedy
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.q_update_freq = q_update_freq

        # Initialize environment and experiences queue
        self.environment = environment
        self.current_state = self.environment.reset()
        self.experiences = Experiences(experiences_size)
        self.rewards_per_episode = []

        # Initialize DNN
        self.q = self.init_q()
        self.q_target = self.init_q()

        self.tensorboard = tensorboard

    def init_q(self):
        q = keras.Sequential([
            keras.layers.InputLayer(input_shape=(4,)),
            keras.layers.Dense(units=32, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=32, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=16, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=8, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=2, activation='linear')
        ])
        q.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), loss="mse")
        return q

    def sample_action(self):
        if np.random.uniform(0, 1) <= self.epsilon_greedy:
            return self.environment.action_space.sample()
        return self.q.predict(np.atleast_2d(self.current_state)).argmax()

    def sample_batch(self):
        batch_size = self.batch_size
        if self.batch_size > self.experiences.state.shape[0]:
            batch_size = self.experiences.state.shape[0]

        idx = np.random.randint(self.experiences.state.shape[0], size=batch_size)
        return self.experiences.state[idx, :], self.experiences.action[idx], self.experiences.reward[
            idx], self.experiences.state_tag[idx, :], self.experiences.done[idx]

    def decay_epsilon_greedy(self):
        self.epsilon_greedy *= self.decay_rate
        if self.epsilon_greedy < 0.05:
            self.epsilon_greedy = 0.05

    def train_on_batch(self):
        v_state, v_action, v_reward, v_state_tag, v_done = self.sample_batch()
        y_j = np.copy(self.q.predict(v_state))
        y_j[np.arange(y_j.shape[0]), v_action.T] = v_reward.T + (v_done.T == 0) * self.discount_factor * np.max(
            self.q_target.predict(v_state_tag), axis=1)
        return self.q.train_on_batch(v_state, y_j)

    def step(self, steps_counter, to_render=True):
        action = self.sample_action()
        state_tag, reward, done, info = self.environment.step(action)
        self.experiences.add_experience(self.current_state, action, reward, state_tag, done)

        if to_render:
            self.environment.render()

        loss = self.train_on_batch()
        with self.tensorboard.as_default():
            tf.summary.scalar('loss', loss, step=steps_counter)

        self.current_state = state_tag
        self.decay_epsilon_greedy()
        if not steps_counter % self.q_update_freq:
            self.q_target.set_weights(self.q.get_weights())
        return done, reward

    def train_agent(self, max_episodes=5000, max_steps=100, to_render=True):
        steps_counter = 0
        for episode in range(max_episodes):
            rewards_per_episode = 0
            for step in range(max_steps):
                steps_counter += 1
                done, reward = self.step(steps_counter, to_render)
                rewards_per_episode += reward
                if done:
                    break
            self.current_state = self.environment.reset()
            self.rewards_per_episode.append(rewards_per_episode)
            with self.tensorboard.as_default():
                tf.summary.scalar('Rewards per episode', rewards_per_episode, step=episode)
                tf.summary.scalar('Mean episode score over 100 consecutive episodes', self.mean_episode_score(), step=episode)
            print("episode: {}, rewards_per_episode: {}, epsilon_greedy: {}".format(episode, rewards_per_episode,
                                                                                    self.epsilon_greedy))
            if self.mean_episode_score() > 480:
                return 1

    def test_agent(self, max_episodes):
        self.current_state = self.environment.reset()
        self.environment.render()

        for episode in range(max_episodes):
            rewards_per_episode = 0
            done = False
            while not done:
                action = self.q_target.predict(np.atleast_2d(self.current_state)).argmax()
                self.current_state, reward, done, info = self.environment.step(action)
                rewards_per_episode += reward
                self.environment.render()
                if done:
                    break
            print("test => episode: {}, rewards_per_episode: {}".format(episode, rewards_per_episode))
            self.current_state = self.environment.reset()

    def mean_episode_score(self):
        self.rewards_per_episode = self.rewards_per_episode[-100:]
        return np.array(self.rewards_per_episode).mean()


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()  # Comment this line in order to save to logs file.
    # Create tensorflow log file. (to use tensorboard, type: tensorboard --logdir=logs/ in pycharm terminal)
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    env = gym.make('CartPole-v1')
    agent = DeepQLearning(env, epsilon_greedy=1, decay_rate=0.9995, learning_rate=0.002,
                          discount_factor=0.95,
                          batch_size=150, experiences_size=10000, q_update_freq=100,
                          tensorboard=train_summary_writer)
    agent.train_agent(max_episodes=500, max_steps=500, to_render=True)
    agent.test_agent(max_episodes=100)
