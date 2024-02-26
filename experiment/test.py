from pandaenv.envs import *
from multi_agent_experiment import System, Agent
import numpy as np

import unittest

class TestMyEnv(unittest.TestCase):
    def setUp(self):
        agents = [Agent(np.array([0, 0]), 0), Agent(np.array([1, 1]), 1), Agent(np.array([2, 2]), 2)]
        self.env = MyEnv(size=50, agents=agents)

    def test_distance_and_reward(self):
        # Set known agent positions and goal
        self.env.goal = np.array([4, 4])
        self.env.agents['Agent0'].cord = np.array([3, 3])
        self.env.agents['Agent1'].cord = np.array([4, 4])
        self.env.agents['Agent2'].cord = np.array([5, 5])

        observation = self.env._get_obs()
        reward = self.env.compute_reward(observation)

        # Check distances
        expected_dists = [np.linalg.norm(np.array([3, 3]) - self.env.goal),
                          np.linalg.norm(np.array([4, 4]) - self.env.goal),
                          np.linalg.norm(np.array([5, 5]) - self.env.goal)]
        computed_dists = [np.linalg.norm(observation[f'Agent{i}'] - self.env.goal) for i in range(len(self.env.agents))]

        for expected, computed in zip(expected_dists, computed_dists):
            self.assertAlmostEqual(expected, computed, places=5, msg=f"Expected distance {expected}, got {computed}")

        expected_reward = -sum(expected_dists[:2]) + expected_dists[2]
        self.assertAlmostEqual(reward, expected_reward, places=5, msg=f"Expected reward {expected_reward}, got {reward}")

if __name__ == '__main__':
    unittest.main()