import os
import tensorflow as tf
import random
import gc

from log import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque


# Define the Deep Q-Learning algorithm.
class DQN(nn.Module):
    def __init__(self, SETTINGS, env):
        self.env = env  # learning environment
        self.alpha = SETTINGS.alpha  # learning rate
        self.gamma = SETTINGS.gamma  # discount factor
        self.batch_size = SETTINGS.batch_size  # batch size for replay buffer
        self.method = SETTINGS.method
        self.experience_replay = deque(maxlen=self.batch_size)

        self.n_actions = self.env.action_space.shape[0]
        self.q_net = self.build_nn()


    def build_nn(self):
        input = len(np.ravel(self.env.state))

        layer_sizes = [input, 64, self.n_actions]

        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.ReLU()
            layers += (linear, act)
        return nn.Sequential(*layers)

    def save(self, SETTINGS, df, e):

        df.to_csv(SETTINGS.generate_filename(SETTINGS, "results", e) + ".csv", sep=',', index=True)

        self.model.save(SETTINGS.generate_filename(SETTINGS, "models", e))

    def load(self, SETTINGS, e):
        self.model = tf.keras.models.load_model(SETTINGS.generate_filename(SETTINGS, "models", e))

    def select_action(self, state):  # Method to select the action to take
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().cuda())
        Q, A = torch.max(Qp, axis=0)
        A = A if torch.rand(1, ).item() > self.epsilon else torch.randint(0, self.n_actions, (1,))
        return A

    def sample_from_experience(self):
        if len(self.experience_replay) < self.batch_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    # REQUEST-BASED
    def update_request(self):

        # Sample a batch of experiences from the model's memory.
        batch = random.sample(self.memory, self.batch_size)

        states = []
        Q_matrices = []

        for sample in batch:
            # Unpack the experience tuple.
            state, action, reward, next_state, _ = sample

            # Predict the Q-values for the next state.
            Q_next = self.model.predict(np.ravel(next_state).reshape(1, -1), verbose=0)
            # Get the maximum Q-value for the next state.
            max_Q_next = np.max(Q_next, axis=1)
            # Compute the target Q-values using the Bellman equation.
            Q_target = reward + (self.gamma * max_Q_next)

            # Predict the Q-values for the current state.
            Q_matrix = self.model.predict(np.ravel(state).reshape(1, -1), verbose=0)
            # Update the target Q-values for the actions taken.
            Q_matrix[:, action] = Q_target

            # Add the state and Q-matrix to the lists for training the model.
            states.append(np.ravel(state).reshape(1, -1))
            Q_matrices.append(np.ravel(Q_matrix).reshape(1, -1))

        # Train the model on the batch using the target Q-values as the target output.
        self.model.train_on_batch(np.concatenate(states, axis=0), np.concatenate(Q_matrices, axis=0))

    def update_new(self):

    def train(self, SETTINGS, PARAMS):
        # Calculate the total number of days for the simulation.
        n_days = SETTINGS.init_days + SETTINGS.test_days
        # Determine the days at which to save the model.
        model_saving_days = [day for day in range(n_days) if day % 100 == 0] + [n_days - 1]

        # Run the simulation for the given range of episodes.
        for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
            print(f"\nEpisode: {e}")
            # If this isn't the first episode, load the previous episode's saved model.
            if e > 0:
                tf.keras.backend.clear_session()
                del self.model
                gc.collect()
                self.load(SETTINGS, e - 1)

            # Start with an empty memory and initial epsilon.
            self.memory = []
            self.epsilon = SETTINGS.epsilon

            # Reset the environment.
            self.env.reset(SETTINGS, PARAMS, e, max(SETTINGS.n_hospitals, key=lambda i: SETTINGS.n_hospitals[i]))
            # Initialize a dataframe to store the output of the simulation.
            df = initialize_output_dataframe(SETTINGS, PARAMS, self.env.hospital, e)

            # Set the current state and day to the environment's initial state and day.
            state = self.env.state
            day = self.env.day

            # Loop through each day in the simulation.
            while day < n_days:

                # REQUEST-BASED
                done = False
                todays_reward = 0

                # If there are no requests for today, proceed to the next day.
                if sum(self.env.state[:, -1]) == 0:
                    self.env.next_day(PARAMS)
                    done = True

                if self.method == 'day':
                    # DAY-BASED
                    action = self.select_action(state)  # Select an action using the Q-network's epsilon-greedy policy
                    next_state, reward, df = self.env.step(SETTINGS, PARAMS, action, day,
                                                           df)  # Take the action and receive the next state, reward and next day
                    if day >= SETTINGS.init_days:
                        self.memory.append(
                            [state, action, reward, next_state, day])  # Store the experience tuple in memory
                    state = next_state

                else:
                    # REQUEST-BASED
                    # If there are requests for today, loop through each request.
                    while not done:
                        # Select an action using the model's epsilon-greedy policy.
                        action = self.select_action(state)
                        # Calculate the reward and update the dataframe.
                        reward, df = self.env.calculate_reward(SETTINGS, PARAMS, action, day, df)
                        todays_reward += reward
                        # Get the next state and whether the episode is done.
                        next_state, done = self.env.next_request(PARAMS)
                        # Store the experience tuple in memory.
                        if day >= SETTINGS.init_days:
                            self.memory.append([state, action, reward, next_state, day])
                        # Update the current state to the next state.
                        state = next_state

                # If there are enough experiences in memory, update the model.
                if len(self.memory) >= self.batch_size:
                    if self.method == 'day':
                        self.update_day()
                    else:
                        self.update_request()

                # Update the dataframe with the current day's information.
                df.loc[day, "logged"] = True
                print(f"Day {day}, reward {todays_reward}")

                # Update log df
                self.env.log_state(PARAMS, day, df)

                # Update the model's epsilon value.
                df.loc[day, "epsilon current"] = self.epsilon
                self.epsilon = max(self.epsilon * SETTINGS.epsilon_decay, SETTINGS.epsilon_min)

                # Save model and log file on predefined days.
                if day in model_saving_days:
                    self.save(SETTINGS, df, e)

                # Set the current day to the environment's current day.
                day = self.env.day
