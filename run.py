from dqn.dqn import DQN
import numpy as np
from tetris import Tetris
import random
import tensorflow as tf
import tensorflow.keras as tfk


if __name__ == "__main__":
    agent = tfk.models.load_model('saved_model/my_model_1800')
    env = Tetris()
    current_state = env.reset()
    done = False
    max_steps = 10000
    for _ in range(max_steps):
        if done:
            print("Score: ", env.score)
            print("Cleared Lines", env.cleared_lines)
            break
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())

        reward_states = agent.predict(next_states)
        ind = np.argmax(reward_states)
        action = next_actions[ind]
        reward, done = env.step(action, render=True)

    print("Done")

