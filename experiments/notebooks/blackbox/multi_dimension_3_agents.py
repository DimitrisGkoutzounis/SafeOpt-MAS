
"""

Multi-Agent experiment

* Each agent predicts the 3 next actions
* Same starting point for all agents [0]
* Same initial global Reward for all agents [1]
* Each agent applies his agent.id actions i.e agent 1 applies [1,0,0] agent 2 applies [0,1,0] and agent 3 applies [0,0,1]
* No communication between agents
* Bounds are -2,2

"""



import GPy
import numpy as np
import matplotlib.pyplot as plt
import safeopt

plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['font.size'] = 12

def global_function(x1,x2,x3):
    "Each agent contributes to a different part of the global function"
    
    result =  np.sin(x1**3) + np.cos(x2**2) - np.sin(x3)
    
    #define guassian noise
    # noise = np.random.normal(0,0.08**2)
    # noise = np.random.normal(loc=0.0, scale=0.1)
    return result


class Agent:
    def __init__(self,id,noise_var=0.05**2):

        self.bounds = [(-2, 2), (-2, 2), (-2, 2)]
        self.id = id
        self.kernel =GPy.kern.RBF(input_dim=3)
        self.x0 = np.zeros((1, len(self.bounds))) 
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

def plot_global_function_3d_comparison(optimized_xs, text,tikz_filename):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare the data for 3D plot
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = global_function(X, Y, X)  
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    
    # Add optimized points
    for opt_x in optimized_xs:
        opt_z = global_function(opt_x, opt_x, opt_x)
        ax.scatter(opt_x, opt_x, opt_z, color='red', s=50, label='Optimized' if opt_x == optimized_xs[0] else "")
    
    # Labeling and legends
    ax.set_title(text)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (Function Value)')
    plt.legend()
    plt.show()

    




def plot_global_function_2d(optimum_xs, text,tikz_filename):
    x = np.linspace(-2.5, 2.5, 100)
    y = global_function(x, x, x) 
    
    plt.title(text + " 2D")  # Set the title of the plot
    plt.plot(x, y, label="Function")
    
    # Place red marks for the optimized x values
    for opt_x in optimum_xs: 
        optimum_y = global_function(opt_x, opt_x, opt_x)
        plt.scatter(opt_x, optimum_y, color='red', zorder=5)  
    
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.legend()
    plt.show()









agent1 = Agent(0)
agent2 = Agent(1)
agent3 = Agent(2)

rewards = []


agents = [agent1,agent2,agent3]

#experiment 1 - No communication
for _ in range(40):
    agent_pred = [agent.predict() for agent in agents] 

    y_meas = global_function(agent_pred[0][0], agent_pred[1][1], agent_pred[2][2])

    #agent 1 updates with his own prediction
    x_next_1 = np.asarray([agent_pred[0][0], agent_pred[0][1], agent_pred[0][2]]) 
    agent1.update(x_next_1, y_meas)

    x_next_2 = np.asarray([agent_pred[1][0], agent_pred[1][1], agent_pred[1][2]])  
    agent2.update(x_next_2, y_meas)

    x_next_3 = np.asarray([agent_pred[2][0], agent_pred[2][1], agent_pred[2][2]])  
    agent3.update(x_next_3, y_meas)



print(f"Agent predictions | Agent 1: {agent_pred[0]} | Agent 2: {agent_pred[1]} | Agent 3: {agent_pred[2]}")
print(f"Applied predictions | Agent 1: {agent_pred[0][0]} | Agent 2: {agent_pred[1][1]} | Agent 3: {agent_pred[2][2]}")

print(f"Global reward evaluation | {global_function(agent_pred[0][0], agent_pred[1][1], agent_pred[2][2])}")

plot_global_function_3d_comparison([agent_pred[0][0], agent_pred[1][1], agent_pred[2][2]], "Agent Predictions",'multi_dimension_3_agents_3d_40.tikz')
plot_global_function_2d([agent_pred[0][0], agent_pred[1][1], agent_pred[2][2]], "Agent Predictions",'multi_dimension_3_agents_2d_40.tikz')