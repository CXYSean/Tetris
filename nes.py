from nes_py.wrappers import JoypadSpace
#import universe
import gym
import gym_tetris
from gym_tetris.actions import MOVEMENT
from dqn.dqn import DQN
import numpy as np
from tetris import Tetris


def train_tetris():
    num_epochs = 2000
    max_steps = None
    history = []
    agent = DQN(4)
    
    env = Tetris()
    
    done = True
    epoch = 0
    while epoch < num_epochs:
        current_state = env.reset()
        done = False
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        steps = 0

        while not done and (not max_steps or steps < max_steps):
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            best_state_ind = agent.predict_move(next_states)

            action = next_actions[best_state_ind]
            reward, done = env.step(action, render=False)
            agent.add_memory(current_state, reward, next_states[best_state_ind], done)
            current_state = next_states[best_state_ind]
            steps += 1

        if len(agent.memory)<1000:
            continue
        agent.train()
        if epoch % 50 == 0:
            agent.save_model(epoch)
        history.append(current_state)
        print(epoch)
        epoch += 1

    np.savetxt("states.csv",history,delimiter=",",fmt="% s")


'''

    for step in range(5000):
        if done:
            state = env.reset()
        action = np.random.randint(len(MOVEMENT))
        state, reward, done, info = env.step(9)
        print(action, reward, done, info)
        
        env.render()
    print(gym_tetris.actions.MOVEMENT)

    env.close()
    for epoch in range(num_epochs):
        state = env.reset()
        done = False
        steps = 0
        while not done and (not max_steps or steps < max_steps):
            state, reward, done, info = env.step(env.action_space.sample())

    '''

if __name__ == "__main__":
    train_tetris()
