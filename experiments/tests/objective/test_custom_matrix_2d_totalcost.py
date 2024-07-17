import numpy as np
import matplotlib.pyplot as plt
import safeopt 
import GPy
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl

file_path = 'logs/experiment_data.csv'
data = pd.read_csv(file_path)
# Define the global reward function
def f(x1, x2):
    y = x1 - x2
    return y

class Agent:
    def __init__(self, id, bounds, safe_point):
        self.bounds = [bounds]
        self.id = id
        self.global_rewards = np.array([])
        self.max_belief = np.array([[]])

        self.z0 = np.asarray([[safe_point]])
        self.y0 = np.asarray([[0.5]])

        self.kernel = GPy.kern.RBF(1)
        self.gp = GPy.models.GPRegression(self.z0, self.y0, self.kernel, noise_var=0.05**2)
        self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)
        self.opt = safeopt.SafeOpt(self.gp, self.parameter_set, -np.inf, beta=2, threshold=0.2)

    def optimize(self):
        z_next = self.opt.optimize()
        return z_next
    
    def update(self, z_next, y_meas):
        self.max_belief = np.append(self.max_belief, self.opt.get_maximum()[1])
        self.opt.add_new_data_point(z_next, y_meas)

    def plot_opt(self):
        self.opt.plot(1000)
        plt.xlabel('Action')
        plt.ylabel('Reward')


        # Define function to process each experiment's data
def extract_data(experiment_data, lambda1, lambda2, lambda3):
    # Extract X and Z values
    X_values = experiment_data[['X1', 'X2']].values
    Z_values = experiment_data[['Z1', 'Z2']].values
    total_cost = experiment_data['total_cost'].values
    R_max = experiment_data['R(Z)_max'].values

    filtered_X = X_values[(experiment_data['Lambda1'] == lambda1) &
                            (experiment_data['Lambda2'] == lambda2) &
                            (experiment_data['Lambda3'] == lambda3)]
    filtered_Z = Z_values[(experiment_data['Lambda1'] == lambda1) &
                            (experiment_data['Lambda2'] == lambda2) &
                            (experiment_data['Lambda3'] == lambda3)]
    filtered_costs = total_cost[(experiment_data['Lambda1'] == lambda1) &
                            (experiment_data['Lambda2'] == lambda2) &
                            (experiment_data['Lambda3'] == lambda3)]
    filtered_max = R_max[(experiment_data['Lambda1'] == lambda1) &
                            (experiment_data['Lambda2'] == lambda2) &
                            (experiment_data['Lambda3'] == lambda3)]
    
    
    return filtered_X, filtered_Z, filtered_costs, filtered_max

def run_normal_experiment(agent1,agent2,N):  

    for i in range(N):

        x1_next = agent1.optimize()
        x2_next = agent2.optimize()
        # x3_next = agent3.optimize()


        y = f(x1_next, x2_next)

        agent1.update(x1_next,y)
        agent2.update(x2_next,y)
        # agent3.update(x3_next,y)


def run_experiments(agent1,agent2,N):

    actions_1 = np.array([])
    actions_2 = np.array([])
    # actions_3 = np.array([])

    for i in range(N):

        z1_next = agent1.optimize()
        z2_next = agent2.optimize()
        # z3_next = agent3.optimize()

        x1_next = model_Z_X_0.predict_noiseless(z1_next.reshape(-1,1))[0]
        x2_next = model_Z_X_1.predict_noiseless(z2_next.reshape(-1,1))[0]
        # x3_next = model_Z_X_2.predict_noiseless(z3_next.reshape(-1,1))[0]  

        x1_next = np.asarray([x1_next]).flatten()
        x2_next = np.asarray([x2_next]).flatten()
        # x3_next = np.asarray([x3_next]).flatten()
        

        actions_1 = np.append(actions_1, x1_next)
        actions_2 = np.append(actions_2, x2_next)
        # actions_3 = np.append(actions_3, x3_next)


        y = f(x1_next, x2_next)

        agent1.update(z1_next,y)
        agent2.update(z2_next,y)
        # agent3.update(z3_next,y)

    return actions_1, actions_2


def plot_data_for_lambda(data, lambda1, lambda2, lambda3):
    # Filter data for specific lambda values
    group_data = data[(data['Lambda1'] == lambda1) & 
                      (data['Lambda2'] == lambda2) & 
                      (data['Lambda3'] == lambda3)]
    return group_data

if __name__ == '__main__':

    grouped = data.groupby(['Lambda1', 'Lambda2', 'Lambda3'])
    lambda3 = 1.0

    data = pd.read_csv(file_path)
    best_msi = np.inf
    worst_msi = -np.inf
    best_params = None
    worst_params = None
    parameter_combinations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    total_msi = np.zeros((len(parameter_combinations), len(parameter_combinations)))
    total_costs_list = np.zeros((len(parameter_combinations), len(parameter_combinations)))
    total_rmax = np.zeros((len(parameter_combinations), len(parameter_combinations)))

    for i, lambda1 in enumerate(parameter_combinations):
        for j, lambda2 in enumerate(parameter_combinations):
                X, Z, total_costs,rmax = extract_data(data, lambda1, lambda2, lambda3)
                X = X[:20]
                Z = Z[:20]
                
                total_costs_list[i,j] = total_costs[0]
                print(total_costs[0])
                total_rmax[i,j] = rmax[0]
                print(rmax[0])

                agent1 = Agent(1, (-1, 1), 0.1)
                agent2 = Agent(2, (-1, 1), 0.2)

                model_Z_X_0 = GPy.models.GPRegression(Z[:, 0].reshape(-1, 1), X[:, 0].reshape(-1, 1), GPy.kern.RBF(1))
                model_Z_X_1 = GPy.models.GPRegression(Z[:, 1].reshape(-1, 1), X[:, 1].reshape(-1, 1), GPy.kern.RBF(1))

                actions_1, actions_2 = run_experiments(agent1, agent2, 20)

                s1 = agent1.opt.x
                s2 = agent2.opt.x
                
                m1 = agent1.gp.predict(s1)
                m2 = agent2.gp.predict(s2)  
                
                #calculate the RMSE
                total_msi[i,j] = np.sqrt(np.mean((s1 - m1)**2) + np.mean((s2 - m2)**2))
                
                print(f"Lambda1={lambda1}, Lambda2={lambda2}, KPI={total_msi[i,j]}")
                
                
    lambda1, lambda2 = np.meshgrid(parameter_combinations, parameter_combinations)
        
    fig = plt.figure(figsize=(21, 7))

    # First subplot for Total Costs
    ax1 = fig.add_subplot(131, projection='3d')  # 121 means 1 row, 2 columns, 1st subplot
    ax1.plot_surface(lambda1, lambda2, total_costs_list, cmap='viridis')
    ax1.set_xlabel('Lambda1')
    ax1.set_ylabel('Lambda2')
    ax1.set_zlabel('Total Cost (J)')
    ax1.set_title('Total Cost')

    # Second subplot for RMSE
    ax2 = fig.add_subplot(132, projection='3d')  # 122 means 1 row, 2 columns, 2nd subplot
    ax2.plot_surface(lambda1, lambda2, total_msi, cmap='viridis')
    ax2.set_xlabel('Lambda1')
    ax2.set_ylabel('Lambda2')
    ax2.set_zlabel('RMSE')
    ax2.set_title('Total RMSE')
    
    total_costs_list,total_msi = np.meshgrid(parameter_combinations, parameter_combinations)
    
    ax3 = fig.add_subplot(133, projection='3d')  # 122 means 1 row, 2 columns, 2nd subplot
    ax3.plot_surface(total_costs_list, total_msi, total_rmax, cmap='viridis')
    ax3.set_xlabel('Total Cost')
    ax3.set_ylabel('RMSE')
    ax3.set_zlabel('R(Z)_max')
    ax3.set_title('R(Z)_max')    

    # Show the plot
    plt.show()
    

    
    # # 3D plot of total costs
    # # lambda1, lambda2 = np.meshgrid(parameter_combinations, parameter_combinations)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(lambda1, lambda2, total_costs_list, cmap='viridis')
    # ax.set_xlabel('Lambda1')
    # ax.set_ylabel('Lambda2')
    # ax.set_zlabel('Total Cost(J)')
    # ax.set_title('Total Cost for varying Lambda1 and Lambda2 with Lambda3=1.0')
    # # plot_path = os.path.join(plot_directory, f'plot_exp_{experiment_number}_lambda1_{lambda1}_lambda2_{lambda2}_lambda3_{lambda3}.png')
    # # plt.savefig(plot_path)
    
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(lambda1, lambda2, total_msi, cmap='viridis')
    # ax.set_xlabel('Lambda1')
    # ax.set_ylabel('Lambda2')
    # ax.set_zlabel('RMSE')
    # ax.set_title('RMSR for varying Lambda1 and Lambda2 with Lambda3=1.0')
    # plt.show()












