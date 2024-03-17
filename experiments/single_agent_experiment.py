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
        self.bounds =[-0.5,0.5],[-0.0,0.0]
        self.init_action = np.array([0.0,0.0])

        self.action = {f'Agent{self.id}': self.init_action} #initialize empty action for each agent
        self.local_reward = np.asarray([[-3.5]]) #initialize empty reward for each agent
        

    def set_position(self,cord):
        self.cord = np.array(cord,dtype=float)

    def plot_gp(self,x,y):
        self.opt.plot(100)
        plt.plot(x,y,color="C2",alpha=0.3)
        plt.show()


    def _add_new_action(self,action):
        self.action[f'Agent{self.id}'] = action


    def initialize_optimization(self):

        X_init = self.cord.reshape(1, -1)
        kernel = GPy.kern.sde_Matern32(input_dim=len(self.bounds),lengthscale=0.7,ARD=True,variance=1.0)
        self.gp = GPy.models.GPRegression(X_init, self.local_reward, noise_var=0, kernel=kernel)
        print("Reward fmin: ", self.local_reward)

        parameter_set = linearly_spaced_combinations(self.bounds, 100)
        self.opt = SafeOpt(self.gp, parameter_set, fmin=-4,threshold=0.4 ,beta=3.5)

        first_action = self.opt.optimize()
        self.opt.add_new_data_point(first_action, self.local_reward)

        self._add_new_action(first_action)

        return self.opt, self.gp,first_action
        

            
    def optimize(self,action,reward):

        self.opt.add_new_data_point(action, reward)
    
        x_next = self.opt.optimize()

        self._add_new_action(x_next)




class System(object):

    def __init__(self,agent):

        self.env = gym.make('Env-v0', agents=agent, size=5)

        self.agents = agent
        self.env.reset()
        self.env.render()

        self.total_actions_taken =[]
        self.total_rewards_received = []

        #containes the actions for each agent
        self.system_actions = {f'Agent{agent.id}': agent.action[f'Agent{agent.id}'] for agent in self.agents} 


        self._init_rewards()

    def _init_rewards(self):

        "Responsible for assigning the initial reward to each agent"

        obs = self.env.get_obs() 

        for agent in self.agents:

            rew = self.env.compute_reward(obs)
            rew = np.asarray([[rew]])
            agent.local_reward = rew

    def simulate(self):


        for agent in self.agents:
            act = agent.action[f'Agent{agent.id}']
            self.system_actions[f'Agent{agent.id}'] = act
                
            obs, reward, terminated,truncated, info = self.env.step(self.system_actions)
            print("Reward: ", reward)
            self.total_rewards_received.append(reward)


            agent.optimize(act,reward)
                
            #update agents coordinates infromation
            agent_key = f'Agent{agent.id}'
            agent.cord = obs[agent_key]


        self.env.render()
        return obs, reward, terminated, info








# Initialize agents
agents = [Agent(0)]


agents[0].set_position([2.5,4.0])
agents[0].initialize_optimization()




system = System(agents)  # Initialize your system with the agents


for i in range(30):

    obs, reward, terminated, info = system.simulate()  # Simulate the system with the initial actions

    if terminated:
        print("Agent reached the goal!")
        system.env.reset()
        break



figure = plt.figure()


plt.plot(system.total_rewards_received)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward per step')
plt.show()
    

