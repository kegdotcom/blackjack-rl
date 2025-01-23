import tensorflow as tf
# from tf.keras import Sequential
from collections import deque
import random
import numpy as np

from blackjack import Blackjack


class DQNAgent:
    def __init__(self, env: Blackjack):
        self.experience = deque()
        self.env = env
        self.rewards = []
        self.build_model()

        # Hyperparameters
        self.buffer_size = 2048
        self.batch_size = 128
        self.discount_factor = 0.9
        self.learning_rate = 0.1
        self.episodes = int(1e5)
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.hidden_units = [8]

    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(
            shape=(self.env.state_size,), activation="relu"
        ))
        for layer in range(len(self.hidden_units)):
            self.model.add(tf.keras.layers.Dense(
                self.hidden_units[layer], activation="relu"
            ))
        self.model.add(tf.keras.layers.Dense(
            # technically no activation but consistency is cool
            self.env.action_size, activation="linear"
        ))

    def forget(self):
        self.experience.popleft()

    def remember(self, e):
        self.experience.append(e)
        while len(self.experience) > self.buffer_size:
            self.forget()

    def sample_experience(self, k=1):
        return random.sample(self.experience, k=k)

    def predict(self, state, epsilon_greedy=True):
        # random actions only for now
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_size-1)
            self.epsilon *= self.epsilon_decay
        else:
            return random.randint(0, self.env.action_size-1)

    def train(self):
        for ep in range(self.episodes):
            s, done = self.env.get_state()
            i = 0
            ep_reward = 0
            while not done:
                i += 1
                a = self.predict(s)
                s, a, r, s_prime, done = self.env.step(a)
                print(
                    f"{ep}-{i}: (s: {s}, a: {a}, r: {r}, s_prime: {s_prime})")
                sars = (s, a, r, s_prime)
                self.remember(sars)
                s = s_prime
                ep_reward += r
            self.env.reset()
            self.rewards.append(ep_reward)
            if ep % 1e4 == 0:
                print(f"Average reward after episode {
                      ep}: {np.mean(self.rewards)}")
