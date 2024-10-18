"""

Local Rewards experiment

Each agent predicts 1 actions

"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
import safeopt
from mpl_toolkits.mplot3d import Axes3D



def global_reward(x1,x2,x3):
    "Each agent contributes to a different part of the global function"
    
    result =  np.sin(x1**3) + np.cos(x2**2) - np.sin(x3)
    
    return result




class Agent:
    def __init__(self,id,bounds,safe_point):
        self.bounds = [bounds]
        self.id = id
        self.safepoint = safe_point

        self.global_x0 = np.asarray([[self.safepoint]])
        self.global_y0 = np.asarray([[1]]) #predermined initial reward for all agents close to the actual

        self.kernel = GPy.kern.RBF(input_dim=1)
        self.gp = GPy.models.GPRegression(self.global_x0,self.global_y0, self.kernel, noise_var=0.05**2)
        self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)

        self.global_opt = safeopt.SafeOpt(self.gp, self.parameter_set, 0.0,beta=3.5,threshold=0.2)

        self.local_x0 = np.asarray([[self.safepoint]])
        self.local_y0 = self.local_reward(self.safepoint)
        self.local_y0 = np.asarray([[self.local_y0]])
        print(f"Local Reward for agent {self.id} is {self.local_y0}")

        self.local_gp = GPy.models.GPRegression(self.local_x0,self.local_y0, self.kernel, noise_var=0.05**2)
        self.local_opt = safeopt.SafeOpt(self.local_gp, self.parameter_set,-0.5,beta=10,threshold=0.8)

    def local_reward(self,x):
        
        if self.id == 1:
            return np.sin(x**3)
        elif self.id == 2:
            return np.cos(x**2)
        elif self.id == 3:
            return -np.sin(x)
        
    def update_local(self,x_next,y_meas):
        self.local_opt.add_new_data_point(x_next, y_meas)


    def predict_local(self):

        print(f"predicting {self.id}")
        x_next_l = self.local_opt.optimize()

        return x_next_l
    
    def update_global(self,x_next,y_meas):
        self.global_opt.add_new_data_point(x_next, y_meas)

    def predict_global(self):
            
        x_next_g = self.global_opt.optimize()
    
        return x_next_g


    def plot_gp_local(self):
        """
        Plot the local reward belief
        """
        if self.id == 1:
            self.local_opt.plot(1000)
            plt.plot(self.parameter_set, self.local_reward(self.parameter_set),color="C2", alpha=0.3)

        if self.id == 2:
            self.local_opt.plot(1000)
            plt.plot(self.parameter_set, self.local_reward(self.parameter_set),color="C2", alpha=0.3)
        if self.id == 3:
            self.local_opt.plot(1000)
            plt.plot(self.parameter_set, self.local_reward(self.parameter_set),color="C2", alpha=0.3)

        plt.title(f"Agent {self.id} Local Reward")
        plt.show()
        
    def plot_gp_global(self):
        """
        Plot the global reward belief
        """
        if self.id == 1:
            self.global_opt.plot(1000)
            plt.plot(self.parameter_set, global_reward(self.parameter_set,self.parameter_set,self.parameter_set),color="C2", alpha=0.3)

        if self.id == 2:
            self.global_opt.plot(1000)
            plt.plot(self.parameter_set, global_reward(self.parameter_set,self.parameter_set,self.parameter_set),color="C2", alpha=0.3)
        if self.id == 3:
            self.global_opt.plot(1000)
            plt.plot(self.parameter_set, global_reward(self.parameter_set,self.parameter_set,self.parameter_set),color="C2", alpha=0.3)

        plt.title(f"Agent {self.id} Global Reward Belief")
        plt.show()




agent1 = Agent(1,(-2,2),0.5)
agent2 = Agent(2,(-2,2),0.1)
agent3 = Agent(3,(-2,2),-0.1)

agents = [agent1,agent2,agent3]


x_next_1 = agent1.predict_global()
