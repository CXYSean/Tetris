from dqn.dqn import DQN
import numpy as np
from tetris import Tetris
import random
import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd


if __name__ == "__main__":
    scores = []
    pieces = []
    lines = []
    for i in range(40):
        num = 50*i
        path = "saved_model/my_model_"+str(num)
        agent = tfk.models.load_model(path)
        env = Tetris()
        current_state = env.reset()
        done = False
        max_steps = 5000
        for _ in range(max_steps):
            if done:
                print("Score: ", env.score)
                print("Cleared Lines", env.cleared_lines)
                scores.append(env.score)
                pieces.append(env.tetrominoes)
                lines.append(env.cleared_lines)
                break
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())

            reward_states = agent.predict(next_states)
            ind = np.argmax(reward_states)
            action = next_actions[ind]
            reward, done = env.step(action, render=False)

    d = {"score": scores, "piece": pieces, "line": lines}
    df = pd.DataFrame.from_dict(d)
    df.to_csv("scores.csv",index=None)

    print("Done")