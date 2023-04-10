import tensorflow as tf
import gym
import random
import os

from dc import *
from hospital import *
from log import *

# Define the Deep Q-Learning algorithm.
class DQN:
    def __init__(self, SETTINGS, env):
        self.env = env                          # learning environment
        self.alpha = SETTINGS.alpha             # learning rate
        self.gamma = SETTINGS.gamma             # discount factor
        self.batch_size = SETTINGS.batch_size   # batch size for replay buffer
        
        # Initialize the NN for generating Q-matrices.
        self.model = self.build_model()                  


    def save(self, SETTINGS, df, e):

        df.to_csv(SETTINGS.generate_filename(SETTINGS, "results", e)+".csv", sep=',', index=True)

        self.model.save(SETTINGS.generate_filename(SETTINGS, "models", e))


    def load(self, SETTINGS, e):
        self.model = tf.keras.models.load_model(SETTINGS.generate_filename(SETTINGS, "models", e))

    
    # Initialize the NN for generating Q-matrices.
    def build_model(self):

        # Get the size of the input state and output action spaces
        input_size = len(np.ravel(self.env.state))          
        # output_size = self.env.action_space.shape[0]*self.env.action_space.shape[1]   # DAY-BASED
        output_size = self.env.action_space.shape[0]                                      # REQUEST-BASED  

        # Create the input layer and two hidden layers with relu activation and output layer with sigmoid activation
        inputs = tf.keras.layers.Input(shape=(input_size,))
        layer_0 = tf.keras.layers.Dense(self.env.num_bloodgroups, activation='relu')(inputs)
        layer_1 = tf.keras.layers.Dense(64, activation='tanh')(layer_0)
        layer_2 = tf.keras.layers.Dense(32, activation='tanh')(layer_1)
        outputs = tf.keras.layers.Dense(output_size, activation='sigmoid')(layer_2)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))

        return model


    def select_action(self, state):                         # Method to select the action to take
        if np.random.rand() <= self.epsilon:                # Choose a random action with probability epsilon
            Q_values = self.env.action_space.sample()
        else:                                               # Choose the action with the highest predicted Q-value
            # Predict the action given the state.
            Q_values = self.model.predict(np.ravel(state).reshape(1,-1), verbose=0)

        # For each row in the Q_matrix (for each blood group), put a 1 in the cell with the highest Q-value.
        # action = np.argmax(np.reshape(Q_values, self.env.action_space.shape), axis=1)   # DAY-BASED
        action = np.argmax(Q_values)        # REQUEST-BASED
        
        return action


    # REQUEST-BASED
    def update(self):
        
        # Sample a batch of experiences from the model's memory.
        batch = random.sample(self.memory, self.batch_size)

        states = []
        Q_matrices = []

        for sample in batch:

            # Unpack the experience tuple.
            state, action, reward, next_state, _ = sample

            # Predict the Q-values for the next state.
            Q_next = self.model.predict(np.ravel(next_state).reshape(1,-1), verbose=0)
            # Get the maximum Q-value for the next state.
            max_Q_next = np.max(Q_next, axis=1)
            # Compute the target Q-values using the Bellman equation.
            Q_target = reward + (self.gamma * max_Q_next)

            # Predict the Q-values for the current state.
            Q_matrix = self.model.predict(np.ravel(state).reshape(1,-1), verbose=0)
            # Update the target Q-values for the actions taken.
            Q_matrix[:,action] = Q_target

            # Add the state and Q-matrix to the lists for training the model.
            states.append(np.ravel(state).reshape(1,-1))
            Q_matrices.append(np.ravel(Q_matrix).reshape(1,-1))

        # Train the model on the batch using the target Q-values as the target output.
        self.model.train_on_batch(np.concatenate(states, axis=0), np.concatenate(Q_matrices, axis=0))


    # # DAY-DASED
    # def update(self):

    #     batch = random.sample(self.memory, self.batch_size)

    #     states = []
    #     Q_tables = []

    #     for sample in batch:

    #         state, action, reward, next_state, _ = sample

    #         # Compute the target Q-values
    #         Q_next = np.reshape(self.model.predict(np.ravel(next_state).reshape(1,-1), verbose=0), self.env.action_space.shape)             # Predict the Q-values for the next states
    #         max_Q_next = np.max(Q_next, axis=1)
    #         Q_target = reward + (self.gamma * max_Q_next)  # Compute the target Q-values using the Bellman equation

    #         # Compute the current Q-values
    #         Q_table = np.reshape(self.model.predict(np.ravel(state).reshape(1,-1), verbose=0), self.env.action_space.shape)             # Predict the Q-values for the next states
    #         Q_table[:,action] = Q_target # Update the target Q-values for the actions taken

    #         states.append(np.ravel(state).reshape(1,-1))
    #         Q_tables.append(np.ravel(Q_table).reshape(1,-1))

    #     # Train the model on the batch using the target Q-values as the target output
    #     self.model.train_on_batch(np.concatenate(states, axis=0), np.concatenate(Q_tables, axis=0))


    def train(self, SETTINGS, PARAMS):
        # Calculate the total number of days for the simulation.
        n_days = SETTINGS.init_days + SETTINGS.test_days
        # Determine the days at which to save the model.
        model_saving_days = [day for day in range(n_days) if day % 100 == 0] + [n_days-1]

        # Run the simulation for the given range of episodes.
        for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
            print(f"\nEpisode: {e}")
            # If this isn't the first episode, load the previous episode's saved model.
            if e > 0:
                self.load(SETTINGS, e-1)

            # Start with an empty memory and initial epsilon.
            self.memory = []
            self.epsilon = SETTINGS.epsilon

            # Reset the environment.
            self.env.reset(SETTINGS, PARAMS, e, max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i]))
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
                if sum(self.env.state[:,-1]) == 0:
                    self.env.next_day(PARAMS)
                    done = True

                # # DAY-BASED
                # action = self.select_action(state)    # Select an action using the Q-network's epsilon-greedy policy
                # next_state, reward, df = self.env.step(SETTINGS, PARAMS, action, dc, hospital, day, df)   # Take the action and receive the next state, reward and next day
                # self.memory.append([state, action, reward, next_state, day])    # Store the experience tuple in memory
                # state = next_state
                # # Update the Q-network using a batch of experiences from memory
                # if len(self.memory) >= self.batch_size: 
                #     self.update()

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
                    self.memory.append([state, action, reward, next_state, day])
                    # Update the current state to the next state.
                    state = next_state

                # If there are enough experiences in memory, update the model.
                if len(self.memory) >= self.batch_size:
                    self.update()

                # Update the dataframe with the current day's information.
                df.loc[day,"logged"] = True
                print(f"Day {day}, reward {todays_reward}")

                # Update the model's epsilon value.
                df.loc[day,"epsilon current"] = self.epsilon
                self.epsilon = max(self.epsilon * SETTINGS.epsilon_decay, SETTINGS.epsilon_min)

                # Save model and log file on predifined days.
                if day in model_saving_days:
                    self.save(SETTINGS, df, e)

                # Set the current day to the environment's current day.
                day = self.env.day
                
