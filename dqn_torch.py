import os
import random
import gc

from log import *

import torch
import torch.nn as nn
import copy


from collections import deque

# Define the Deep Q-Learning algorithm.
class DQN():
    def __init__(self, SETTINGS, env):
        self.env = env  # learning environment
        self.alpha = SETTINGS.alpha  # learning rate
        self.gamma = SETTINGS.gamma  # discount factor
        self.batch_size = SETTINGS.batch_size  # batch size for replay buffer
        self.method = SETTINGS.method
        self.nn = SETTINGS.nn
        self.experience_replay = deque(maxlen=self.batch_size)
        self.epsilon = SETTINGS.epsilon
   
        self.n_actions = self.env.action_space.shape[0]
        self.q_net = self.build_nn()
        self.q_net.to('cuda')

        self.target = SETTINGS.target

        if self.target:
            self.target_net = copy.deepcopy(self.q_net)
            self.target_net.cuda()
            self.update_counts = 0
            self.target_frequency = SETTINGS.target_frequency

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = torch.nn.MSELoss()
        torch.manual_seed(0)
        random.seed(0)

    def build_nn(self):
        input_size = [len(np.ravel(self.env.state))]
        output = [self.env.action_space.shape[0]]
        layer_sizes = input_size + self.nn + output

        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.ReLU() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)

    def save(self, SETTINGS, df, e):
        df.to_csv(SETTINGS.generate_filename(SETTINGS, "results", e) + ".csv", sep=',', index=True)
        torch.save(self.q_net.state_dict(), SETTINGS.generate_filename(SETTINGS, "models", e))

    def load(self, SETTINGS, e):
        self.q_net.load_state_dict(torch.load(SETTINGS.generate_filename(SETTINGS, "models", e)))

    def select_action(self, state, limit, PARAMS):  # Method to select the action to take
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).flatten().float().cuda())
        Q, A = torch.max(Qp, dim=0)
        A = A if torch.rand(1, ).item() > self.epsilon else torch.randint(0, self.n_actions, (1,))

        # Uncomment below for limited action selection

        # avail = self.available_actions(state, PARAMS)
        # if limit and avail[0].size > 0:
        #     Q, A = torch.max(Qp[avail], dim=0)
        #     A = A if torch.rand(1, ).item() > self.epsilon else np.random.choice(avail[0])
        # else:
        #     Q, A = torch.max(Qp, dim=0)
        #     A = A if torch.rand(1, ).item() > self.epsilon else torch.randint(0, self.n_actions, (1,))
        return A.item()

    def available_actions(self, state, PARAMS):
        I = state[:,:PARAMS.max_age]
        avail = np.where(np.any(I>0, axis=1))
        return avail


    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < self.batch_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.from_numpy(np.vstack([exp[0].flatten() for exp in sample])).float().cuda()
        a = torch.tensor([exp[1] for exp in sample]).float().cuda()
        rn = torch.tensor([exp[2] for exp in sample]).float().cuda()
        sn = torch.from_numpy(np.vstack([exp[3].flatten() for exp in sample])).float().cuda()
        return s, a, rn, sn

    # REQUEST-BASED
    def update_request(self):

        # Sample a batch of experiences from the model's memory.
        s, a, rn, ns = self.sample_from_experience(sample_size=self.batch_size)
        # Predict Q-values of next state
        q_ns = self.q_net(ns.cuda())
        max_q, action = torch.max(q_ns, dim=1)

        q_target = rn.cuda() + self.gamma * max_q

        # Predict q_values of current state
        q_matrix = self.q_net(s.cuda())
        q_m_target = q_matrix.detach().clone()
        q_m_target[:,action] = q_target

        loss = self.loss_fn(q_matrix, q_m_target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q

    def update_target(self):
        # Sample a batch of experiences from the model's memory.
        s, a, rn, ns = self.sample_from_experience(sample_size=self.batch_size)
        # Update target network
        if self.update_counts == self.target_frequency:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.update_counts = 0

        # Predict expected return of current state using main network
        q_matrix = self.q_net(s.cuda())
        pred_return, _ = torch.max(q_matrix, axis=1)

        # Get target return using target network
        q_next = self.get_q_next(ns.cuda())
        target_return = rn.cuda() + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.update_counts += 1
        return loss.item()

    def linear_anneal(self, t, T, start, final, percentage):
        ''' Linear annealing scheduler
        t: current timestep
        T: total timesteps
        start: initial value
        final: value after percentage*T steps
        percentage: percentage of T after which annealing finishes
        '''
        final_from_T = int(percentage * T)
        if t > final_from_T:
            return final
        else:
            return final + (start - final) * (final_from_T - t) / final_from_T

    def train(self, SETTINGS, PARAMS):
        # Calculate the total number of days for the simulation.
        n_days = SETTINGS.init_days + SETTINGS.test_days
        # Determine the days at which to save the model.
        model_saving_days = [day for day in range(n_days) if day % 100 == 0] + [n_days - 1]
        # Set epsilon value


        # Keep track of episode rewards
        e_rewards = []
        # Run the simulation for the given range of episodes.
        for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
            print(f"\nEpisode: {e}")
            # If this isn't the first episode, load the previous episode's saved model.
            # if e > 0:
            #     self.load(SETTINGS, e - 1)

            # Start with an empty memory and initial epsilon.
            self.experience_replay = []
            # self.epsilon = SETTINGS.epsilon

            # Reset the environment.
            self.env.reset(SETTINGS, PARAMS, e, max(SETTINGS.n_hospitals, key=lambda i: SETTINGS.n_hospitals[i]))
            # Initialize a dataframe to store the output of the simulation.
            df = initialize_output_dataframe(SETTINGS, PARAMS, self.env.hospital, e)

            # Set the current state and day to the environment's initial state and day.
            state = self.env.state
            day = self.env.day

            # Limit actions to available actions
            limit = False

            # Episode total reward
            e_reward = 0

            # Loop through each day in the simulation.
            while day < n_days:

                # REQUEST-BASED
                done = False
                todays_reward = 0
                day_loss = 0

                # Update log df with: requests, supplied, inventory
                self.env.log_state(PARAMS, day, df)

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
                        self.experience_replay.append(
                            [state, action, reward, next_state, day])  # Store the experience tuple in memory
                    state = next_state

                else:
                    # REQUEST-BASED
                    # If there are requests for today, loop through each request.
                    while not done:
                        # Select an action using the model's epsilon-greedy policy.
                        action = self.select_action(state, limit, PARAMS)
                        # Create copy of current state
                        current_state = copy.deepcopy(self.env.state)
                        # Calculate the reward and update the dataframe.
                        reward, df, good = self.env.calculate_reward(SETTINGS, PARAMS, action, day, df)
                        todays_reward += reward
                        # Get the next state and whether the episode is done.
                        next_state, done = self.env.next_request(PARAMS, action)
                        # Store the experience tuple in memory.
                        if day >= SETTINGS.init_days:
                            self.experience_replay.append([current_state, action, reward, next_state, day])

                # If there are enough experiences in memory, update the model.
                if len(self.experience_replay) >= self.batch_size:
                    if self.method == 'day':
                        self.update_day()
                    elif self.target:
                        day_loss += self.update_target()
                    else:
                        day_loss += self.update_request()

                # Update the dataframe with the current day's information.
                df.loc[day, "logged"] = True
                print(f"Day {day}, reward {todays_reward}")

                # Sum total episode reward
                e_reward += todays_reward
                # Log daily total loss
                df.loc[day, "day loss"] = day_loss

                # Update the model's epsilon value.
                df.loc[day, "epsilon current"] = self.epsilon
                # self.epsilon = max(self.epsilon * SETTINGS.epsilon_decay, SETTINGS.epsilon_min)

                # Save model and log file on predefined days.
                if day in model_saving_days:
                    self.save(SETTINGS, df, e)

                # Set the current day to the environment's current day.
                day = self.env.day
            e_rewards.append(e_reward)
            self.epsilon = self.linear_anneal(e, int(SETTINGS.episodes[1]), 1, 0.1, 0.8)

        return e_rewards
