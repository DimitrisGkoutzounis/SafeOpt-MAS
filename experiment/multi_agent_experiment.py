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
        observation, reward, terminated,truncated, info = self.env.step(actions)
        # Update each agent's position based on the observation
        for agent in self.agents:

            agent_key = f'Agent{agent.id}'
            agent.cord = observation[agent_key]  # Ensure this matches your observation structure
        self.env.render()  # Visualize after each step
        return observation, reward, terminated, info

    

class Agent(object):

    def __init__(self,agent_id):
        
        #set initial position of each agent
        self.cord = np.zeros(2)
        self.id = agent_id
        self.init_dist =0
        self.dist = np.zeros(0)        
        #set bounds max 2
        self.bounds =[-1,1],[-1,1]

    def set_position(self,cord):
        self.cord = np.array(cord,dtype=float)

    def optimize(self):
        """
        Find's the optimal action for the agent, implements SafeOpt
        
        """
        

class MASafeOpt(object):


    pass



# Initialize agents
agents = [Agent(0), Agent(1), Agent(2)]

#set the position of each agent
agents[0].set_position([1.,1.])
agents[1].set_position([4.,2.])
agents[2].set_position([2.,4.])



system = System(agents)  # Initialize your system with the agents

x1_pos = 0.5
y1_pos = 0.5

x2_pos = 0.1
y2_pos = 0.1

x3_pos = 0.3
y3_pos = 0.3



params1 = np.asarray([x1_pos, y1_pos])
params2 = np.asarray([x2_pos, y2_pos])
params3 = np.asarray([x3_pos, y3_pos])

actions = {
    'Agent0': params1,
    'Agent1': params2,
    'Agent2': params3
}
observation,reward,terminated,info = system.simulate(actions)

f1 = np.asarray([[reward]])
f2 = np.asarray([[reward]])
f3 = np.asarray([[reward]])


x1 = params1.reshape(1, -1) # Reshape to 2D array
x2 = params2.reshape(1, -1) # Reshape to 2D array
x3 = params3.reshape(1, -1) # Reshape to 2D array


KERNEL_f_1 = GPy.kern.sde_Matern32(input_dim=x1.shape[1],lengthscale=0.7,ARD=True,variance=1.0)
KERNEL_f_2 = GPy.kern.sde_Matern32(input_dim=x2.shape[1],lengthscale=0.7,ARD=True,variance=1.0)
KERNEL_f_3 = GPy.kern.sde_Matern32(input_dim=x3.shape[1],lengthscale=0.7,ARD=True,variance=1.0)

gp1 = GPy.models.GPRegression(x1[0,:].reshape(1,-1), f1,noise_var=0.05**2, kernel=KERNEL_f_1)
gp2 = GPy.models.GPRegression(x2[0,:].reshape(1,-1), f2,noise_var=0.05**2, kernel=KERNEL_f_2)
gp3 = GPy.models.GPRegression(x3[0,:].reshape(1,-1), f3,noise_var=0.05**2, kernel=KERNEL_f_3)

bounds = [(-1., 1.), (-1., 1.)]
parameter_set = linearly_spaced_combinations(bounds, 100)

opt1 = SafeOpt(gp1,parameter_set,fmin=-5,beta = 3.5)
opt2 = SafeOpt(gp2,parameter_set,fmin=-5,beta = 3.5)
opt3 = SafeOpt(gp3,parameter_set,fmin=-5,beta = 3.5)

opt1.add_new_data_point(params1, f1)
opt2.add_new_data_point(params2, f2)
opt3.add_new_data_point(params3, f3)


x1_next = opt1.optimize()
x2_next = opt2.optimize()
x3_next = opt3.optimize()

print(x1_next)
print(x2_next)
print(x3_next)

actions = {
    'Agent0': x1_next,
    'Agent1': x2_next,
    'Agent2': x3_next
}
observation,reward,terminated,info = system.simulate(actions)



    

