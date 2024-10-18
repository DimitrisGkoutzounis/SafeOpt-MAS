from __future__ import print_function, division, absolute_import

import GPy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

from IPython.display import clear_output

import safeopt



class Environment:
    def __init__(self, target, bounds, obstacles=None):

        self.target = target  # The shared target location for the agents

        self.bounds = bounds  # Bounds for the environment [(x_min, x_max), (y_min, y_max)]

        self.obstacles = obstacles if obstacles is not None else []  # List of obstacles (if any)



    def plot_environment(self, agents):

        fig, ax = plt.subplots()

        # Plot the bounds of the environment

        ax.set_xlim(self.bounds[0])

        ax.set_ylim(self.bounds[1])

        ax.set_aspect('equal')

        
        # Plot the target
        ax.plot(*self.target, 'ro', markersize=10, label='Target')
        # Plot the agents
        for agent in agents:

            ax.plot(agent.position[0], agent.position[1], 'bo', markersize=10, label=f'Agent {agent.id}')
        # Plot the obstacles
        for obstacle in self.obstacles:

            circle = Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.5)

            ax.add_patch(circle)
        ax.legend()
        plt.show()


# Define the agents
class Agent:

    agent_count = 0
    def __init__(self, start_position):
        self.id = Agent.agent_count
        Agent.agent_count += 1
        self.position = start_position







# Initialize the environment
target_position = (7.5, 10)
environment_bounds = [(0, 20), (0, 20)]
environment_obstacles = [(10, 10.5, 1), (15, 15, 2)]  # Each obstacle is defined as (x, y, radius)



# Initialize agents with random start positions
np.random.seed(42)  # Seed for reproducibility
agent_positions = [np.random.uniform(low, high, size=2) for (low, high) in environment_bounds]
agents = [Agent(pos) for pos in agent_positions]



# Create the environment
env = Environment(target=target_position, bounds=environment_bounds, obstacles=environment_obstacles)

# Plot the initial environment with agents and the target
env.plot_environment(agents)







# We will use a simple potential field method for trajectory planning in this example.
# Function to compute attractive potential to the target
def attractive_potential(position, target, scale):
    return scale * np.linalg.norm(position - target)

# Function to compute repulsive potential from obstacles
def repulsive_potential(position, obstacles, scale, limit):
    rep_potential = 0
    for obs in obstacles:
        dist_to_obs = np.linalg.norm(position - np.array(obs[:2])) - obs[2]
        if dist_to_obs < limit:
            rep_potential += scale * (1/dist_to_obs - 1/limit)**2
    return rep_potential

# Function to compute repulsive potential from the other agent to avoid collision
def inter_agent_repulsion(agent_position, other_agent_position, scale, limit):
    dist_to_agent = np.linalg.norm(agent_position - other_agent_position)
    if dist_to_agent < limit:
        return scale * (1/dist_to_agent - 1/limit)**2
    return 0

# Function to plan the next step based on the potential field
def plan_step(agent, target, obstacles, other_agent_position, att_scale, rep_scale, agent_rep_scale, step_size):
    att_pot = attractive_potential(agent.position, target, att_scale)
    rep_pot = repulsive_potential(agent.position, obstacles, rep_scale, 2.0)  # assuming a limit of 2 units for repulsion
    agent_rep_pot = inter_agent_repulsion(agent.position, other_agent_position, agent_rep_scale, 1.0)  # limit of 1 unit for inter-agent repulsion
    
    # Compute the gradient of the potential
    gradient = np.zeros(2)
    for dx in [-0.1, 0.1]:
        for dy in [-0.1, 0.1]:
            delta = np.array([dx, dy])
            pos = agent.position + delta
            pot = attractive_potential(pos, target, att_scale) + \
                  repulsive_potential(pos, obstacles, rep_scale, 2.0) + \
                  inter_agent_repulsion(pos, other_agent_position, agent_rep_scale, 1.0)
            gradient += (pot - att_pot - rep_pot - agent_rep_pot) * delta
    
    # Normalize the gradient
    gradient = -gradient / np.linalg.norm(gradient)
    
    # Plan the step
    new_position = agent.position + step_size * gradient
    new_position = np.clip(new_position, *zip(*environment_bounds))  # Ensure the new position is within bounds
    return new_position

# Function to update the plot for the simulation

def update_simulation_plot(ax, agents, env):

    # Clear previous drawings
    ax.clear()
    # Set limits and aspect
    ax.set_xlim(env.bounds[0])

    ax.set_ylim(env.bounds[1])

    ax.set_aspect('equal')
    # Plot the target
    ax.plot(*env.target, 'ro', markersize=10, label='Target')
    # Plot the agents
    for agent in agents:

        ax.plot(agent.position[0], agent.position[1], 'bo', markersize=10, label=f'Agent {agent.id}')

    # Plot the obstacles
    for obstacle in env.obstacles:

        circle = Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.5)

        ax.add_patch(circle)

    # Add legend

    ax.legend()


# Function to perform the simulation with live updates

def run_live_simulation(agents, env, iterations, att_scale, rep_scale, agent_rep_scale, step_size):

    fig, ax = plt.subplots()

    for i in range(iterations):

        # For each agent, plan their next step and update their position

        for agent in agents:

            other_agent_position = [other_agent.position for other_agent in agents if other_agent.id != agent.id][0]

            agent.position = plan_step(agent, env.target, env.obstacles, other_agent_position, att_scale, rep_scale, agent_rep_scale, step_size)

        

        # Update the plot
        update_simulation_plot(ax, agents, env)

        plt.pause(0.1)  # Pause for a short period to create animation effect

        clear_output(wait=True)  # Clear the output to update the plot

# Parameters for the potential field
attractive_scale = 1.0
repulsive_scale = 10.0
inter_agent_repulsive_scale = 50.0
step_size = 0.5

# Number of iterations to run the simulation
simulation_iterations = 80

# Run the simulation
run_live_simulation(agents, env, simulation_iterations, attractive_scale, repulsive_scale, inter_agent_repulsive_scale, step_size)