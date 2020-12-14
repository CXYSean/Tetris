
from collections import deque
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as tfk
tfkl = tfk.layers


class DQN():

    def __init__(self, state_size, memory_size=10000):

        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.epsilon_decay = self.epsilon/500
        self.model = self.build_model()

    def build_model(self):
        model = tfk.Sequential([
            tfkl.InputLayer(self.state_size),
            tfkl.Dense(64, activation=tf.nn.relu),
            tfkl.Dense(16, activation=tf.nn.relu),
            tfkl.Dense(1, activation=tf.nn.relu)
        ])

        model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam())
        return model

    def add_memory(self, state, reward, next_state, done):
        self.memory.append([state, reward, next_state, done])


    def predict_reward(self, state):
        tmp_v = self.model.predict(np.array(state))[0]
        return tmp_v[0]

    def predict_move(self, states):
        max_reward = None
        best_state = None
        ind = 0
        if random.random() < self.epsilon:
            return random.randint(0, len(states)-1)
        else:
            res = self.model.predict(states)
            
            ind = np.argmax(res)
            print(ind, res[ind])
            '''
            for i,state in enumerate(states):
                tmp = self.predict_reward(np.array([state]))
                if not max_reward or tmp > max_reward:
                    max_reward = tmp
                    best_state = state
                    ind = i
            '''
        return ind

    def train(self, batch_size = 512, epochs = 3):
        n = len(self.memory)
        if n >= batch_size:
            print("train")
            batch = random.sample(self.memory, batch_size)
            next_states = np.array([x[2] for x in batch])
            next_rewards = [x[0] for x in self.model.predict(next_states)]

            x_train = []
            y_train = []

            for i, (state, reward, _, done) in enumerate(batch):
                x_train.append(state)
                if done:
                    y_train.append(-1)
                else:
                    new_reward = reward + next_rewards[i]
                    y_train.append(new_reward)

            print(sum(y_train))
            self.model.fit(np.array(x_train), np.array(y_train), batch_size = batch_size, epochs = epochs)

        if self.epsilon >= 0:
            self.epsilon -= self.epsilon_decay

    def save_model(self,epochs):
        path = 'saved_model/my_model_'+str(epochs)
        self.model.save(path)



