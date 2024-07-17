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

file_path = 'experiments/tests/objective/trials/experiment7/logs/experiment_data.csv'
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

    filtered_X = X_values[(experiment_data['Lambda1'] == lambda1) &
                            (experiment_data['Lambda2'] == lambda2) &
                            (experiment_data['Lambda3'] == lambda3)]
    filtered_Z = Z_values[(experiment_data['Lambda1'] == lambda1) &
                            (experiment_data['Lambda2'] == lambda2) &
                            (experiment_data['Lambda3'] == lambda3)]
    
    
    return filtered_X, filtered_Z

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

    lambda1 = 0.2
    lambda2 = 0.2
    lambda3 = 0.2

    lambda11 = 1.0
    lambda22 = 0.2
    lambda33 = 1.0

    X, Z = extract_data(data, lambda1, lambda2, lambda3)
    X = X[:20]
    print(X)
    Z = Z[:20]

    X_init,Z_init = extract_data(data, lambda11, lambda22, lambda33)
    #select the first 20
    X_init = X_init[:20]
    Z_init = Z_init[:20]

    
    agent1 = Agent(1, (-1, 1), 0.1)
    agent2 = Agent(2, (-1, 1), 0.2)
    # agent3 = Agent(3, (-1, 1), 0.3)

    f1 = f(X[:,0], X[:,1])
    f2 = f(Z[:,0], Z[:,1])
    f3 = f(Z_init[:,0], Z_init[:,1])

    print(f1.max())

    plt.figure(figsize=(8, 6))
    plt.plot(f1, label='f(X)')
    plt.plot(f2, label='f(Z_opt)')
    plt.plot(f3, label='f(Z_init)')
    plt.legend()
    plt.show()





    model_Z_X_0 = GPy.models.GPRegression(Z[:,0].reshape(-1,1), X[:,0].reshape(-1,1), GPy.kern.RBF(1))
    model_Z_X_1 = GPy.models.GPRegression(Z[:,1].reshape(-1,1), X[:,1].reshape(-1,1), GPy.kern.RBF(1))
    # model_Z_X_2 = GPy.models.GPRegression(Z[:,2].reshape(-1,1), X[:,2].reshape(-1,1), GPy.kern.RBF(1))


    model_X_Z_0 = GPy.models.GPRegression(X[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), GPy.kern.RBF(1))
    model_X_Z_1 = GPy.models.GPRegression(X[:,1].reshape(-1,1), Z[:,1].reshape(-1,1), GPy.kern.RBF(1))
    # model_X_Z_2 = GPy.models.GPRegression(X[:,2].reshape(-1,1), Z[:,2].reshape(-1,1), GPy.kern.RBF(1))

    actions_1, actions_2 = run_experiments(agent1,agent2,20)


    plt.plot(f1, label='f(X)')
    agent1.plot_opt()
    agent2.plot_opt()

    # plt.plot(f3, label='f(Z_init)')
    plt.show()


    # run_normal_experiment(agent1, agent2, 20)

    # agent1.plot_opt()
    # agent2.plot_opt()

    # plt.show()

    # data = pd.read_csv(file_path)
    # best_msi = -np.inf
    # worst_msi = np.inf
    # best_params = None
    # worst_params = None
    # parameter_combinations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # for lambda1 in parameter_combinations:
    #     for lambda2 in parameter_combinations:
    #         for lambda3 in parameter_combinations:
    #             X, Z = extract_data(data, lambda1, lambda2, lambda3)
    #             X = X[:20]
    #             Z = Z[:20]

    #             agent1 = Agent(1, (-1, 1), 0.1)
    #             agent2 = Agent(2, (-1, 1), 0.2)

    #             model_Z_X_0 = GPy.models.GPRegression(Z[:, 0].reshape(-1, 1), X[:, 0].reshape(-1, 1), GPy.kern.RBF(1))
    #             model_Z_X_1 = GPy.models.GPRegression(Z[:, 1].reshape(-1, 1), X[:, 1].reshape(-1, 1), GPy.kern.RBF(1))

    #             actions_1, actions_2 = run_experiments(agent1, agent2, 20)

    #             msi = agent1.opt.get_maximum()[1]
    #             print(msi)

    #             if msi > best_msi:
    #                 best_msi = msi
    #                 best_params = (lambda1, lambda2, lambda3)

    #             if msi < worst_msi:
    #                 worst_msi = msi
    #                 worst_params = (lambda1, lambda2, lambda3)

    # print(f'Best MSI: {best_msi} with parameters: Lambda1={best_params[0]}, Lambda2={best_params[1]}, Lambda3={best_params[2]}')
    # print(f'Worst MSI: {worst_msi} with parameters: Lambda1={worst_params[0]}, Lambda2={worst_params[1]}, Lambda3={worst_params[2]}')

    # # Plot best parameters
    # X, Z = extract_data(data, best_params[0], best_params[1], best_params[2])
    # X = X[:20]
    # Z = Z[:20]

    # model_best_Z_X_0 = GPy.models.GPRegression(Z[:, 0].reshape(-1, 1), X[:, 0].reshape(-1, 1), GPy.kern.RBF(1))
    # model_best_Z_X_1 = GPy.models.GPRegression(Z[:, 1].reshape(-1, 1), X[:, 1].reshape(-1, 1), GPy.kern.RBF(1))

    # agent1 = Agent(1, (-1, 1), 0.1)
    # agent2 = Agent(2, (-1, 1), 0.2)

    # actions_1, actions_2 = run_experiments(agent1, agent2, 20)

    # agent1.plot_opt()
    # agent2.plot_opt()
    # plt.title('Best Parameters')

    # plt.show()

    
    





    # f_true = f(x1, x2, x3)

    # rewards = f(a1, a2, a3)
    
    # fig = plt.figure(figsize=(12, 8))

    # # First subplot
    # ax1 = fig.add_subplot(121, projection='3d')
    # sc1 = ax1.scatter(a1, a2, a3, c=rewards, cmap=cm.viridis, marker='o')
    # ax1.set_xlabel('X1')
    # ax1.set_ylabel('X2')
    # ax1.set_zlabel('X3')
    # plt.colorbar(sc1, label='Reward')
    # plt.title('SafeOpt')

    # # Second subplot
    # ax2 = fig.add_subplot(122, projection='3d')
    # sc2 = ax2.scatter(x1, x2, x3, c=f_true, cmap=cm.viridis, marker='o')
    # ax2.set_xlabel('X1')
    # ax2.set_ylabel('X2')
    # ax2.set_zlabel('X3')
    # plt.colorbar(sc2, label='Reward')
    # plt.title("True Function")



    # plt.show()


    #     # Assume the data has columns named 'X', 'Y', 'Z', 'C' corresponding to your variables
    # x = a1.flatten()  # Ensuring x is a flat array
    # y = a2.flatten()  # Ensuring y is a flat array
    # z = a3.flatten()  # Ensuring z is a flat array
    # c = rewards.flatten()  # Ensuring c is a flat array

    # # Set variable names for axes labels and titles
    # list_name_variables = ['X', 'Y', 'Z', 'Color Value']
    # index_x, index_y, index_z, index_c = 0, 1, 2, 3



