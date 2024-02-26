import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from gym.spaces import Dict, Box

class MyEnv(gym.Env):
    def __init__(self, size=50, agents=None):
        super(MyEnv, self).__init__()
        self.size = size
        self.window_size = 512

        # Initialize agents from the provided list
        self.agents = {f'Agent{i}': agents[i] for i in range(len(agents))}

        # self.observation_space = spaces.Box(low=-self.size, high=self.size, shape=(2,), dtype=np.float32)

        self.observation_space = Dict({
                f"Agent{i}": Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=float)
                for i in range(len(agents))  # Replace `number_of_agents` with your actual number of agents
            })

        self.action_space = Dict({
                f'Agent{i}': Box(low=-1, high=1, shape=(2,), dtype=float) for i in range(len(agents))
            })
        # self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.goal = np.array([4, 4.])

        self.step_size = 1  # Predefined distance each agent moves in a step

        self.figure = None

    def _get_obs(self):
        
        observations = {}

        for i, (name, agent) in enumerate(self.agents.items()):
            observations[name] = agent.cord


        return observations


    def reset(self, seed=None, **kwargs):
        


        super().reset(seed=seed, **kwargs)


        self.goal = self.goal

        #define initial position of each agent
        for i, (name,agent) in enumerate(self.agents.items()):
            agent.init_dist = np.linalg.norm(agent.cord - self.goal)

        # Reset each agent's position if needed here
        #For example, agent.cord = np.array([starting_position])

        obs = self._get_obs()
        info = {}

        return (obs, info) 

    def step(self, action):

        for i, (name, agent) in enumerate(self.agents.items()):
            agent_action = np.array(action[name], dtype=float)
            agent.cord += agent_action * self.step_size


        observation = self._get_obs()
        reward = self.compute_reward(observation)
        terminated = self._is_success(observation, self.goal)
        

        info = { 
                'is_success': terminated,
                'goal': self.goal,
                'observation': observation,
            }

        return observation, reward, terminated,False, info

    def render(self, mode='human', step=None):

        self.render_mode = mode

        # Create or clear the figure
        if self.figure is None:
            self.figure, self.ax = plt.subplots(figsize=(8, 8))
        else:
            self.ax.clear()  # Clear the plot to redraw updated positions

        # Configure plot limits, title, and grid
        self.ax.set_xlim(-1, self.size + 1)
        self.ax.set_ylim(-1, self.size + 1)
        self.ax.grid(True)
        title = f'Step {step}' if step is not None else 'Environment'
        self.ax.set_title(title)

        # Plot the goal as a red 'x'
        self.ax.plot(self.goal[0], self.goal[1], 'rx', markersize=10, label='Goal')

        # Plot the agents as circles with labels
        for name, agent in self.agents.items():
            self.ax.plot(agent.cord[0], agent.cord[1], 'o', label=name)

        # Add a legend to the plot
        self.ax.legend()

        # Show the plot
        plt.show(block=False)
        plt.pause(1)  # Pause to ensure the plot is updated visually


    def close(self):
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None

    def compute_reward(self, observation):

        distance1 = np.linalg.norm(observation['Agent0'] - self.goal)
        distance2 = np.linalg.norm(observation['Agent1'] - self.goal)
        distance3 = np.linalg.norm(observation['Agent2'] - self.goal)
        
        reward = distance1 + distance2 - distance3
        reward *= -1.0
    
        return reward
    

    def _is_success(self, observation, goal):
        # Check if any agent is close enough to the goal
        for name, pos in observation.items():
            if np.linalg.norm(pos - goal) < 0.1:
                print(f"{name} is at the goal")
                print(f"Distance from target is {np.linalg.norm(pos - goal)}")
                return True
        return False
