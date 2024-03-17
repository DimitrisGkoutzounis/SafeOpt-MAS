

"""
Multi-Agent experiment

* Each agent predicts 3 actions
* Same starting point for all agents [0]
* Same initial global Reward for all agents [1]
* Each agent applies 
* Communication is enabled:

* Bounds are -2,2

"""





import GPy
import numpy as np
import matplotlib.pyplot as plt
import safeopt
from mpl_toolkits.mplot3d import Axes3D

def global_function(x1,x2,x3):
    "Each agent contributes to a different part of the global function"
    
    result =  np.sin(x1**3) + np.cos(x2**2) - np.sin(x3)
    
    return result


class Agent:
    def __init__(self,id,safepoint,noise_var=0.05**2):

        self.bounds = [(-2, 2), (-2, 2), (-2, 2)]
        self.id = id
        self.kernel =GPy.kern.RBF(input_dim=3)
        self.x0 = np.asarray([[safepoint,safepoint,safepoint]]) 
        self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)

        self.global_reward = global_function(self.x0[0][0],self.x0[0][0],self.x0[0][0])
        self.y0 = np.asarray([[self.global_reward]])    
        self.other_actions = np.asarray([0, 0, 0])

        # self.y0 = np.array([[self.global_reward.item(), self.other_actions[0], self.other_actions[1]]])
        
        self.gp = GPy.models.GPRegression(self.x0,self.y0, self.kernel, noise_var=noise_var) #initiallize gp with initial safe point
        self.opt = safeopt.SafeOpt(self.gp, self.parameter_set, 0.0,beta=3.5,threshold=0.2)

    def predict(self):
        return self.opt.optimize()
    
    def update(self,x_next,y_meas):
        self.opt.add_new_data_point(x_next, y_meas)

    def plot_gp(self):
    
        plt.title(f"Agent {self.id}")  
        #true function plot
        plt.plot(self.parameter_set, global_function(self.opt.inputs,self.opt.inputs,self.opt.inputs), "r--", label="True function")   
        plt.legend()
        plt.show()







def plot_global_function_2d(optimum_xs, text):
    x = np.linspace(-2.5, 2.5, 100)
    y = global_function(x, x, x)

    plt.title(text + " 2D")  # Set the title of the plot
    plt.plot(x, y, label="Function")
    
    # For each agent's prediction, plot it on the function curve
    for opt_x in optimum_xs:
        # Ensure opt_x is a scalar for each agent's prediction
        opt_x_scalar = opt_x.flatten()[0]  # Flatten in case it's a numpy array and take the first element
        plt.scatter(opt_x_scalar, global_function(opt_x_scalar, opt_x_scalar, opt_x_scalar), color='red', label='Optimum Points')

    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.legend()
    plt.show()




def plot_global_function_3d(optimum_xs, text):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = text + " 3D"
    plt.title(title)

    # Create a meshgrid for the x, y dimensions, and compute z for each (x, y) pair
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = global_function(X, Y, X)  # Example of how to compute Z using the global function

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

    # Plot the agents' predictions as points
    for opt_x in optimum_xs:
        # Ensure opt_x is a scalar for each agent's prediction
        opt_x_scalar = opt_x.flatten()[0]  # Flatten in case it's a numpy array and take the first element
        # Here we plot the point. Since global_function expects 3 inputs, we use opt_x_scalar for all to place it in 3D space.
        # This might need adjustment depending on the actual global function's logic.
        z_pred = global_function(opt_x_scalar, opt_x_scalar, opt_x_scalar)
        ax.scatter(opt_x_scalar, opt_x_scalar, z_pred, color='red', s=50)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Function Value')

    plt.show()


agent1 = Agent(0,0)
agent2 = Agent(1,0)
agent3 = Agent(2,0)



agents = [agent1,agent2,agent3]

#experiment 1 - communication Agent 1 receives a21 and a31, all agents apply 1st actions
for _ in range(40):
    agent_pred = [agent.predict() for agent in agents] 

    #each agent applies the first action from his prediction
    y_meas = global_function(agent_pred[0][0], agent_pred[1][0], agent_pred[2][0])


    #Agent 1 updates a12 and a13 with the applied actions of agent2 and agent3
    x_next_1 = np.asarray([agent_pred[0][0], agent_pred[1][0], agent_pred[2][0]]) 
    agent1.update(x_next_1, y_meas)

    #agent 2 updates his own predictions with the applied actions of agent2
    x_next_2 = np.asarray([agent_pred[1][0], agent_pred[2][2], agent_pred[2][0]])  
    agent2.update(x_next_2, y_meas)

    x_next_3 = np.asarray([agent_pred[2][0], agent_pred[0][1], agent_pred[0][2]])  
    agent3.update(x_next_3, y_meas)


print(f"Agent Predictions | Agent 1: {agent_pred[0]} | Agent 2: {agent_pred[1]} | Agent 3: {agent_pred[2]}")
print(f"Applied predictions | Agent 1: {agent_pred[0][0]} | Agent 2: {agent_pred[1][0]} | Agent 3: {agent_pred[2][0]}")


print(f"Global Reward evaluation | {global_function(agent_pred[0][0], agent_pred[1][0], agent_pred[2][0])}")

plot_global_function_2d([agent_pred[0][0], agent_pred[1][0], agent_pred[2][0]], "Agent Predictions Communication")
plot_global_function_3d([agent_pred[0][1], agent_pred[1][0], agent_pred[2][0]], "Agent Predictions Communication")

