import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import clear_output, display

import GPy
import safeopt


class MultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,inter_agent_repulsive_scale=100.0):
        super(MultiAgentEnv, self).__init__()
        #Define action space as a placeholder, actual actions are vectors towards the target.
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.inter_agent_repulsive_scale = inter_agent_repulsive_scale
        #Initialize environment parameters
        self.target_position = np.array((5.5, 10))
        self.environment_bounds = [(0, 20), (0, 20)]
        self.environment_obstacles = [(10, 10, 2), (5, 5, 1), (15, 5, 1), (5, 15, 1), (15, 15, 1)]
        self.agents = []
        
        #Parameters for potential fields
        self.attractive_scale = 1.0
        self.repulsive_scale = 10.0
        self.step_size = 0.1

        self.figure, self.ax = plt.subplots()
        
        self.reset()

    def update_params(self,params):
        self.inter_agent_repulsive_scale = params[0]

    def step(self, action):
        rewards = []
        done = False  
        info = {}

        for index, agent in enumerate(self.agents):
            other_agents_positions = [other_agent.position for idx, other_agent in enumerate(self.agents) if idx != index]
            new_position = self.plan_step(agent, self.target_position, self.environment_obstacles, other_agents_positions)
            agent.position = new_position

            if np.linalg.norm(agent.position - self.target_position) < 0.1:
                done = True

        
        for agent in self.agents:
            distance_to_target = np.linalg.norm(agent.position - self.target_position)
            if distance_to_target < 0.1:
                rewards.append(100)  
            else:
                rewards.append(-distance_to_target)  
        return np.array(self.get_observation()), np.array(rewards), done, info

    def reset(self):

        start_positions = [(1, 1), (1, 19), (19, 1), (19, 19)] 
        self.agents = [Agent(np.array(pos)) for pos in start_positions]

        return np.array(self.get_observation())


    def render(self, mode='human'):
        self.ax.clear()
        self.ax.set_xlim(self.environment_bounds[0])
        self.ax.set_ylim(self.environment_bounds[1])
        self.ax.plot(*self.target_position, 'ro', markersize=10, label='Target')
        for agent in self.agents:
            self.ax.plot(agent.position[0], agent.position[1], 'bo', markersize=10, label=f'Agent {agent.id}')
        for obstacle in self.environment_obstacles:
            circle = Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.5)
            self.ax.add_patch(circle)
        self.ax.legend()
        plt.pause(0.01) 

    def close(self):
        plt.close()

    def get_observation(self):
        positions = [agent.position for agent in self.agents]
        positions.append(self.target_position)
        return positions

    def plan_step(self, agent, target, obstacles, other_agents_positions):
        direction = np.zeros(2)

        direction += self.attractive_scale * (target - agent.position) / np.linalg.norm(target - agent.position)

        for obs in obstacles:
            obs_position = np.array(obs[:2])
            distance = np.linalg.norm(agent.position - obs_position) - obs[2]
            if distance < 1.0: 
                repulsion_force = self.repulsive_scale * (1 / np.maximum(distance, 0.1) - 1 / 2.0)
                direction += repulsion_force * (agent.position - obs_position) / np.linalg.norm(agent.position - obs_position)

        for other_pos in other_agents_positions:
            other_pos = np.array(other_pos)
            distance = np.linalg.norm(agent.position - other_pos)
            if distance < 1.0:  # Ensure agents maintain a buffer from each other
                repulsion_force = self.inter_agent_repulsive_scale * (1 / np.maximum(distance, 0.1) - 1 / 1.0)
                direction += repulsion_force * (agent.position - other_pos) / np.linalg.norm(agent.position - other_pos)

        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction) * self.step_size
        new_position = agent.position + direction
        new_position = np.clip(new_position, *zip(*self.environment_bounds))  # Ensure within bounds
        
        return new_position


class Agent:
    agent_count = 0

    def __init__(self, start_position):
        self.id = Agent.agent_count
        Agent.agent_count += 1
        self.position = start_position


# if __name__ == "__main__":
#     env = MultiAgentEnv()
#     obs = env.reset()
#     done = False
#     while not done:
#         obs, rewards, done, info = env.step(None)  # Actions are determined by the controller
#         env.render()
#     display(env.figure)  # Keep the final plot displayed after the loop
        
def run_experiment(params):
    env = MultiAgentEnv(inter_agent_repulsive_scale=params)
    obs = env.reset()
    done = False
    while not done:
        obs, rewards, done, info = env.step(None)  # Using the modified environment
        env.render()

    #Calculate the reward as the negative distance of all agents from the target
    total_distance = sum(np.linalg.norm(agent.position - env.target_position) for agent in env.agents)
    average_distance = total_distance / len(env.agents)
    reward = -average_distance

    # Check the minimum distance between agents for safety
    min_distance = np.inf
    for i, agent1 in enumerate(env.agents):
        for j, agent2 in enumerate(env.agents):
            if i != j:
                distance = np.linalg.norm(agent1.position - agent2.position)
                if distance < min_distance:
                    min_distance = distance

    # Safety constraint: ensuring agents do not get too close to each other
    is_safe = min_distance > 1.0  

    return reward, is_safe

noise_var = 0.05 ** 2  # Variance of the simulated noise
parameter_bounds = [(10, 100)]  # Bounds for the inter-agent repulsive scale

parameter_set = safeopt.linearly_spaced_combinations(parameter_bounds, num_samples=10)

kernel = GPy.kern.RBF(input_dim=len(parameter_bounds), variance=2.0, lengthscale=1.0)

x0 = np.array([[10]])


def objective_function(x):
    reward, is_safe = run_experiment(x[0, 0])
    print(f"Reward: {reward:.2f}, Safety: {is_safe}")
    return reward if is_safe else None

print("test")
gp = GPy.models.GPRegression(x0, np.array([[objective_function(x0)]]), kernel, noise_var=noise_var)

opt = safeopt.SafeOpt(gp, parameter_set, -4, threshold=1.0,beta=10)



for i in range(20):
    x_next = opt.optimize()
    y = objective_function(x_next)
    opt.add_new_data_point(x_next, y)

    print(f"Iteration {i+1}:")
    print(f"  Selected inter_agent_repulsive_scale: {x_next[0, 0]:.2f}")
    print(f"  Reward: {y:.2f}")
    print(f"  Safety condition satisfied: {'Yes' if y else 'No'}")
    print("----------------------------------------")