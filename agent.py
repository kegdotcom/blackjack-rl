import tensorflow as tf
from tensorflow import keras
# from tf.keras import Sequential
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

from blackjack import Blackjack


class DQNAgent:
    def __init__(self, env: Blackjack):
        # Hyperparameters
        self.buffer_size = 2048
        self.batch_size = 128
        self.discount_factor = 0.9
        self.learning_rate = 0.1
        self.epochs = 100
        self.episodes = 25000
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        self.hidden_units = [8]
        self.target_update_delay = 64

        self.checkpoint_path = "checkpoints/checkpoint.weights.h5"
        self.buffer = deque(maxlen=self.buffer_size)
        self.env = env
        self.actions = []
        self.avg_reward = 0
        self.n_trains = 0
        self.reward_tracker = dict()
        self.qnet = self.create_network()
        self.target_net = self.create_network()
        self.target_net.set_weights(self.qnet.get_weights())

        self.save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path, save_weights_only=True, verbose=0)

    def create_network(self):
        network = tf.keras.Sequential()
        network.add(tf.keras.Input(
            shape=(self.env.state_size,)
        ))
        for units in self.hidden_units:
            network.add(tf.keras.layers.Dense(
                units, activation="relu"
            ))
        # technically no activation but consistency is cool
        network.add(tf.keras.layers.Dense(
            self.env.action_size, activation="linear"
        ))
        network.compile(optimizer="rmsprop", loss=self.loss_calc)
        return network

    def loss_calc(self, y_true, y_pred):
        indices = np.zeros(shape=(y_true.shape[0], 2), dtype="int32")
        indices[:, 0] = np.arange(y_true.shape[0], dtype="int32")
        indices[:, 1] = self.actions

        ys = tf.gather_nd(y_true, indices=indices)
        qs = tf.gather_nd(y_pred, indices=indices)
        diffs = tf.subtract(ys, qs)
        sq_diffs = tf.square(diffs)
        return tf.reduce_mean(sq_diffs)

    def sample_experience(self):
        return random.sample(self.buffer, k=self.batch_size)

    def qnet_predict(self, state, epsilon_greedy=True):
        # random actions only for now
        if epsilon_greedy and max(self.epsilon_min, random.random()) < self.epsilon:
            random_action = random.randint(0, self.env.action_size-1)
            self.epsilon *= self.epsilon_decay
            return random_action
        else:
            state = np.array(state).reshape((1, self.env.state_size))
            predicted_action = self.qnet.predict(state, verbose=0)
            return np.argmax(predicted_action, -1)[0]

    def update_target_net(self):
        print("metrics", self.qnet.get_metrics_result())
        self.target_net.set_weights(self.qnet.get_weights())

    def train(self):
        batch = self.sample_experience()  # [ ... (s, a, r, s', done) ... ]
        states = np.zeros((self.batch_size, self.env.state_size))
        state_primes = np.zeros((self.batch_size, self.env.state_size))
        for i, (s, a, r, s_prime, done) in enumerate(batch):
            states[i, :] = s
            state_primes[i, :] = s_prime

        current_qs = self.qnet.predict(states, verbose=0)
        next_qs = self.target_net.predict(state_primes, verbose=0)

        self.actions = []
        output = np.zeros(shape=(self.batch_size, self.env.action_size))
        for i, (s, a, r, s_prime, done) in enumerate(batch):
            self.actions.append(int(a))

            y = r
            if not done:
                y += self.discount_factor * np.max(next_qs[i])

            output[i] = current_qs[i]
            output[i, a] = y

        self.qnet.fit(states, output, batch_size=self.batch_size,
                      callbacks=[self.save_checkpoint], verbose=0)

        self.n_trains += 1
        if self.n_trains % self.target_update_delay == 0:
            self.update_target_net()

    def loop(self, train=False, load_from_checkpoint=True):
        print(tf.config.list_physical_devices('GPU'))
        self.env.reset()

        if load_from_checkpoint:
            self.load_checkpoint()

        if train:
            for epoch in range(1, self.epochs+1):
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
                self.reward_tracker[epoch] = self.avg_reward
                self.avg_reward = 0
                # fig, ax = plt.subplots()
                # ax.plot(self.reward_tracker)
                # plt.savefig("plots/epoch-rewards.png")
        else:
            self.balance = 1000
            self.n_wins = 0
            self.n_losses = 0
            self.n_ties = 0
            while self.balance > 0:
                self.env.reset()
                s, done = self.env.get_state()
                while not done:
                    a = self.qnet_predict(s)
                    s, a, r, s_prime, done = self.env.step(a)
                    if r == self.env.illegal_reward:
                        break
                    elif done:
                        self.balance += r
                        if r < 0:
                            self.n_losses += 1
                        elif r > 0:
                            self.n_wins += 1
                        else:
                            self.n_ties += 1
                        self.env.print_results()
                        print(f"new balance: {self.balance} with {self.n_wins} wins, {
                              self.n_losses} losses, and {self.n_ties} ties")
                        break

    def load_checkpoint(self):
        self.qnet.load_weights(self.checkpoint_path)
        self.target_net.load_weights(self.checkpoint_path)
