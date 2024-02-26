import GPy

import numpy as np
import matplotlib.pyplot as plt

from safeopt import SafeOpt

import gym

from pandaenv.envs import *

from safeopt import linearly_spaced_combinations

from gym.utils.env_checker import check_env


import safeopt





class System(object):

    def __init__(self,agent):

        self.env = gym.make('Env-v0', agents=agent, size=5)

        self.agents = agent
        self.env.reset()

        self.env.render()
    




    def simulate(self, actions):
        # Assuming actions is a list of actions for each agent
        observation, reward, terminated, info = self.env.step(actions)
        # Update each agent's position based on the observation
        for agent in self.agents:

            agent_key = f'Agent{agent.id}'
            agent.cord = observation[agent_key]  # Ensure this matches your observation structure
        self.env.render()  # Visualize after each step
        return observation, reward, terminated, info

    




class Agent(object):

    def __init__(self,cord,agent_id):
        
        #set initial position of each agent
        self.cord = cord
        self.id = agent_id
        self.init_dist = None
        self.dist = np.zeros(0)        
        #set bounds max 2
        self.bounds =[-1,1],[-1,1]

        


    

def plot_gp(opt,x,y):
    opt.plot(1000)

bounds = [(-1., 1.), (-1., 1.)]
kernel = GPy.kern.RBF(input_dim=2, variance=2., lengthscale=1.0, ARD=True)
noise_var = 0.05 ** 2

# Initialize agents
agents = [Agent([2., 2.], 0), Agent([3., 1.], 1), Agent([4., 1.], 2)]


system = System(agents)  # Initialize your system with the agents
observation,reward,terminated,info = system.simulate([np.array([0.5,0.5]),np.array([0.,0.]),np.array([0.,0.])])

print("Info is ",info)





# y0 = np.array([[reward]])

# # Initialize GP models and SafeOpt instances for each agent
# gps = []
# safeopts = []
# for agent in agents:
#     x0 = np.zeros((1, 2))  # Initial safe point for each agent
#     y0 = np.zeros((1, 1))  # Initial dummy reward
#     gp = GPy.models.GPRegression(x0, y0, kernel, noise_var=noise_var)
#     parameter_set = linearly_spaced_combinations(bounds, 100)
#     opt = SafeOpt(gp, parameter_set, -5, threshold=0.2, beta=3.5)
#     gps.append(gp)
#     safeopts.append(opt)
    




# print("Initial reward is ",reward)

# #obtain global reward
# y_new = np.array([[reward]])  

# actions = []


# for r in range(10):
#     for opt in safeopts:
#         x_next = opt.optimize()
#         print("Next action is ",x_next)
#         y_new = np.array([[reward]])  # Assuming the global reward is shared
#         opt.add_new_data_point(x_next, y_new)
#         actions.append(x_next)

#     #simulate the actions in the environment and get the global reward
#     observation, reward, terminated, info = system.simulate(actions)
#     print("Reward is ",reward)


# for step in range(20):
#     actions = []
#     for opt in safeopts:
#         # Obtain next action from SafeOpt for each agent
#         x_next = opt.optimize()
#         actions.append(x_next)

#     # Simulate the actions in the environment and get the global reward
#     observation, reward, terminated, info = system.simulate(actions)
#     print("Reward is ",reward)
    
#     # Update each agent's GP model with the action taken and the observed reward
#     for i, (gp, opt) in enumerate(zip(gps, safeopts)):
#         x_new = actions[i]
#         y_new = np.array([[reward]])  # Assuming the global reward is shared
#         gp.set_XY(np.vstack([gp.X, x_new]), np.vstack([gp.Y, y_new]))
#         opt.add_new_data_point(x_new, y_new)

#     if terminated:
#         print('terminated')
#         break


# Number of steps to simulate

# for i in range(2):
#     # Simulate for 10 steps
#     while True:
#         # Define the actions for each agent
#         actions = [np.random.uniform(-1, 1, 2) for _ in range(len(system.agents))]
#         # Simulate the environment
#         observation, reward, terminated, info = system.simulate(actions)

#         if terminated:
#             print('terminated')
#             break
#         #print the reward
#         print(reward)




        
        


