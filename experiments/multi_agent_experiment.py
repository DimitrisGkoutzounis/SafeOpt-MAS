import GPy

import numpy as np
import matplotlib.pyplot as plt
from safeopt import SafeOpt
import gym
from pandaenv.envs import *
from safeopt import linearly_spaced_combinations
from gym.utils.env_checker import check_env
import safeopt

np.random.seed(np.random.randint(0, 1000))
    

class Agent(object):

    def __init__(self,agent_id):
        #set initial position of each agent
        self.cord = np.zeros(2)
        self.id = agent_id
        self.init_dist =0
        self.dist = np.zeros(0)        
        #set bounds max 2
        self.bounds =[-0.1,0.1],[-0.1,0.1]
        self.init_action = np.array([0.0,0.0])

        self.action = {f'Agent{self.id}': self.init_action}

        #random seed each time
        self.optimize_initialized = False

    def set_position(self,cord):
        self.cord = np.array(cord,dtype=float)

    def plot_gp(self,x,y):
        self.opt.plot(100)
        plt.plot(x,y,color="C2",alpha=0.3)
        plt.show()

    def initialize_optimization(self):

        if not self.optimize_initialized:
            X_init = self.cord.reshape(1, -1)
            Y_init = np.asarray([[0]])
            kernel = GPy.kern.sde_Matern32(input_dim=X_init.shape[1],lengthscale=0.7,ARD=True,variance=1.0)
            self.gp = GPy.models.GPRegression(X_init, Y_init, noise_var=0, kernel=kernel)

            parameter_set = linearly_spaced_combinations(self.bounds, 100)
            self.opt = SafeOpt(self.gp, parameter_set, fmin=-5, beta=4)
            self.optimize_initialized = True

            

    def optimize(self,reward):

        if not self.optimize_initialized:
            self.initialize_optimization()
        
        #Update SafeOpt with new data point and find the next action
        # reward *= -1
        self.opt.add_new_data_point(self.cord, np.asarray([[reward]]))
        
        next_action = self.opt.optimize()
        print(f"Next action for agent {self.id} is {next_action}")
        #append the next action to the list of actions
        self.action[f'Agent{self.id}'] = next_action
    




class System(object):

    def __init__(self,agent):

        self.env = gym.make('Env-v0', agents=agent, size=5)

        self.agents = agent
        self.env.reset()
        self.env.render()

        self.total_actions_taken =[]
        self.total_rewards_received = []


    def simulate(self):

        #perform a step in the environment with the initial actions

        system_actions = {f'Agent{agent.id}': agent.action[f'Agent{agent.id}'] for agent in self.agents}

        print("Initial actions: ", system_actions)


        for agent in self.agents:
            observation, reward, terminated,truncated, info = self.env.step(system_actions) 
            agent.optimize(reward)
            system_actions[f'Agent{agent.id}'] = agent.action[f'Agent{agent.id}']
            print("Action added: ", system_actions[f'Agent{agent.id}'])
            print("Observation: ", observation)
        
            

        self.total_rewards_received.append(reward)
        self.total_actions_taken.append(system_actions)
        print("Reward: ", reward)

        for agent in self.agents:
            agent_key = f'Agent{agent.id}'
            agent.cord = observation[agent_key]

        self.env.render()

            







# Initialize agents
agents = [Agent(0), Agent(1), Agent(2)]

#set the position of each agent
agents[0].set_position([2.,2.])
agents[1].set_position([3.,2.])
agents[2].set_position([3.,3.])



system = System(agents)  # Initialize your system with the agents


for i in range(10):
    system.simulate()  # Simulate the system with the initial actions




