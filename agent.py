import tensorflow as tf
# from tf.keras import Sequential
from collections import deque
import random
import numpy as np

from blackjack import Blackjack


class DQNAgent:
    def __init__(self, env: Blackjack):
        # Hyperparameters
        self.buffer_size = 2048
        self.batch_size = 128
        self.discount_factor = 0.9
        self.learning_rate = 0.1
        self.episodes = int(1e5)
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.hidden_units = [8]
        self.target_update_delay = 64

        self.buffer = deque(maxlen=self.buffer_size)
        self.env = env
        self.actions = []
        self.avg_reward = 0
        self.n_trains = 0
        self.qnet = self.create_network()
        self.target_net = self.create_network()
        self.target_net.set_weights(self.qnet.get_weights())

    def create_network(self):
        network = tf.keras.Sequential()
        network.add(tf.keras.Input(
            shape=(self.env.state_size,), activation="relu"
        ))
        for units in self.hidden_units:
            network.add(tf.keras.layers.Dense(
                units, activation="relu"
            ))
        # technically no activation but consistency is cool
        network.add(tf.keras.layers.Dense(
            self.env.action_size, activation="linear"
        ))
        network.compile(optimizer="", loss=self.loss_calc,
                        metrics=["accuracy"])
        return network

    def loss_calc(self, y_true, y_pred):
        indices = np.zeros(shape=y_true.shape)
        indices[:, 0] = np.arange(y_true.shape[0])
        indices[:, 1] = self.actions

        ys = tf.gather_nd(y_true, indices=indices)
        qs = tf.gather_nd(y_pred, indices=indices)
        errors = tf.subtract(ys, qs)
        squared_errors = tf.math.pow(errors, 2)
        loss = tf.reduce_mean(squared_errors)
        return loss

    def sample_experience(self):
        return np.array(random.sample(self.buffer, k=self.batch_size))

    def qnet_predict(self, state, epsilon_greedy=True):
        # random actions only for now
        if epsilon_greedy and random.random() < self.epsilon:
            random_action = random.randint(0, self.env.action_size-1)
            self.epsilon *= self.epsilon_decay
            return random_action
        else:
            predicted_action = self.qnet.predict(state)
            return tf.argmax(predicted_action)

    def update_target_net(self):
        self.target_net.set_weights(self.qnet.get_weights())

    def train(self):
        batch = self.sample_experience()  # [ ... (s, a, r, s', done) ... ]
        states = batch[:, 0]
        state_primes = batch[:, 3]
        current_qs = self.qnet.predict(states)
        next_qs = self.target_net.predict(state_primes)

        self.actions = []
        output = np.zeros(shape=(self.batch_size, self.env.action_size))
        for i, (s, a, r, s_prime, done) in enumerate(batch):
            self.actions.append(a)

            y = r
            if not done:
                y += self.discount_factor * np.max(next_qs[i])

            output[i] = current_qs[i]
            output[i, a] = y

        self.qnet.fit(batch, output, batch_size=self.batch_size, epochs=10)
        self.n_trains += 1
        if self.n_trains % self.target_update_delay == 0:
            self.update_target_net()

    def loop(self):
        for ep in range(1, self.episodes+1):
            s, done = self.env.get_state()
            i = 0
            ep_reward = 0
            while not done:
                i += 1
                a = self.qnet_predict(s)
                s, a, r, s_prime, done = self.env.step(a)
                transition = (s, a, r, s_prime, done)
                # print(f"{ep}-{i}: {transition}")
                self.buffer.append(transition)

                if len(self.buffer) > self.batch_size:
                    self.train()

                s = s_prime
                ep_reward += r
            self.env.reset()
            self.avg_reward *= (ep-1)
            self.avg_reward += ep_reward
            self.avg_reward /= ep
            if ep % 1000 == 0:
                print(f"avg rew - {ep}:{self.avg_reward}")
