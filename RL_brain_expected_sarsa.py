import numpy as np
import pandas as pd
import random


class rlalgorithm:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, buffer_size=1000):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.display_name="Expected Sarsa"
        print(f"Using {self.display_name} ...")

    def store_experience(self, state, action, reward, next_state):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.buffer_idx] = (state, action, reward, next_state)
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        # Add non-existing state to our q_table
        self.check_state_exist(observation)
 
        # Select next action
        if np.random.uniform() >= self.epsilon:
            # Choose argmax action
            state_action_values = self.q_table.loc[observation, :]
            action = np.random.choice(state_action_values[state_action_values == np.max(state_action_values)].index) # handle multiple argmax with random
        else:
            # Choose random action
            action = np.random.choice(self.actions)

        return action
    
    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def learn_from_batch(self, batch):
        for state, action, reward, next_state in batch:
            q_current = self.q_table.loc[state, action]
            if next_state != 'terminal':
                q_next = max(self.q_table.loc[next_state])
                q_target = reward + self.gamma * q_next
            else:
                q_target = reward
            self.q_table.loc[state, action] += self.lr * (q_target - q_current)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.store_experience(s, a, r, s_)

        if len(self.buffer) > 32:
            print("Replaying memory...")
            batch = self.sample_batch(batch_size=32)
            self.learn_from_batch(batch)

        q_current = self.q_table.loc[s, a]

        if s_ != 'terminal':
            # calculate expected value according to epsilon greedy policy
            state_action_values = self.q_table.loc[s_,:]
            value_sum = np.sum(state_action_values)
            max_value = np.max(state_action_values)
            max_count = len(state_action_values[state_action_values == max_value])
            k = len(self.actions) # total number of actions

            expected_value_for_max = max_value * ((1 - self.epsilon) / max_count + self.epsilon / k) * max_count
            expected_value_for_non_max = (value_sum - max_value * max_count) * (self.epsilon / k)

            expected_value = expected_value_for_max + expected_value_for_non_max

            q_target = r + self.gamma * expected_value # max state-action value
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_current)  # update current state-action value

        return s_, self.choose_action(str(s_))


    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat([self.q_table, pd.DataFrame([[0]*len(self.actions)], columns=self.q_table.columns, index=[state])])
