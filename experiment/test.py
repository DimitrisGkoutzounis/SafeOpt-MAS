import matplotlib.pyplot as plt
import numpy as np

# Define the environment, including the goal and agents
class Environment:
    def __init__(self):
        self.goal = np.array([10, 10])  # Goal position
        self.agents = {
            'Agent1': np.array([0., 0.]),
            'Agent2': np.array([0., 5.]),
            'Agent3': np.array([5., 0.])
        }
        self.step_size = 1  # Predefined distance each agent moves in a step

    def move_agents(self):
        for name, position in self.agents.items():
            direction = self.goal - position
            distance_to_goal = np.linalg.norm(direction)
            step_distance = min(self.step_size, distance_to_goal)
            step_direction = direction / distance_to_goal
            self.agents[name] += step_direction * step_distance

    def plot_environment(self, step):
        plt.figure(figsize=(8, 8))
        for name, position in self.agents.items():
            plt.plot(position[0], position[1], 'o', label=name)
        plt.plot(self.goal[0], self.goal[1], 'rx', label='Goal')
        plt.xlim(-1, 12)
        plt.ylim(-1, 12)
        plt.title(f'Step {step}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Create the environment
env = Environment()

# Simulate and plot each step
for step in range(1, 11):  # Simulate for 10 steps
    env.move_agents()
    env.plot_environment(step)
