import numpy as np
from maze_env import Maze
from RL_brain_expected_sarsa import rlalgorithm

# Agent & Environment parameters
agentXY = [0, 0]
goalXY = [4, 4]

wall_shape=np.array([[7,4],[7,3],[6,3],[6,2],[5,2],[4,2],[3,2],[3,3],[3,4],[3,5],[3,6],[4,6],[5,6]])
pits=np.array([[1,3],[0,5], [7,7]])

env = Maze(agentXY, goalXY, walls=wall_shape, pits=pits, energy_capacity=50, energy_factor=0.01)
RL = rlalgorithm(actions=list(range(env.n_actions)))
RL.load('./models/model.pkl')
env.run_without_learning(RL, episodes=10, sim_speed=0.001)