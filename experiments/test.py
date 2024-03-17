import unittest
import gym
import numpy as np
from pandaenv.envs.two_dim_env import MyEnv
from  multi_agent_experiment import Agent,System

class TestMyEnv(unittest.TestCase):
    def setUp(self):
        # Initialize agents with starting positions
        agents = [Agent(0), Agent(1), Agent(2)]
        agents[0].set_position([0., 0.])
        agents[1].set_position([1., 1.])
        agents[2].set_position([2., 2.])

        # Initialize the environment with these agents
        self.env = MyEnv(size=50, agents=agents)

    def test_action_observation_space(self):
        """Test if action and observation spaces are correctly set up for each agent."""
        for i in range(len(self.env.agents)):
            agent_key = f'Agent{i}'
            self.assertIsInstance(self.env.action_space[agent_key], gym.spaces.Box, "Action space for each agent should be a Box space.")
            self.assertIsInstance(self.env.observation_space[agent_key], gym.spaces.Box, "Observation space for each agent should be a Box space.")

    def test_distance_and_reward(self):
        """Test correct calculation of distances and reward based on agent positions."""
        # Manually set agent positions and goal
        self.env.goal = np.array([4, 4])
        self.env.agents['Agent0'].set_position([3, 3])
        self.env.agents['Agent1'].set_position([4, 4])
        self.env.agents['Agent2'].set_position([5, 5])

        observation = self.env._get_obs()
        reward = self.env.compute_reward(observation)

        # Check distances to goal
        expected_dists = [
            np.linalg.norm(np.array([3, 3]) - self.env.goal),
            np.linalg.norm(np.array([4, 4]) - self.env.goal),
            np.linalg.norm(np.array([5, 5]) - self.env.goal)
        ]
        computed_dists = [np.linalg.norm(observation[f'Agent{i}'] - self.env.goal) for i in range(len(self.env.agents))]

        for expected, computed in zip(expected_dists, computed_dists):
            self.assertAlmostEqual(expected, computed, places=5, msg=f"Expected distance {expected}, got {computed}")

        # est reward calculation
        expected_reward = -sum(expected_dists[:2]) + expected_dists[2]
        self.assertAlmostEqual(reward, expected_reward, places=5, msg=f"Expected reward {expected_reward}, got {reward}")

    def test_goal_achievement(self):
        """Test if the environment correctly identifies when an agent has reached the goal."""
        #Place one agent near the goal
        self.env.agents['Agent0'].set_position(self.env.goal - np.array([0.05, 0.05]))
        observation = self.env._get_obs()

        self.assertTrue(self.env._is_success(observation, self.env.goal), "Environment should recognize when an agent reaches the goal.")

if __name__ == '__main__':
    unittest.main()
