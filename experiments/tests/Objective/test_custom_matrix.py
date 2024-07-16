import numpy as np
import matplotlib.pyplot as plt
import safeopt 
import GPy
import pandas as pd

file_path = 'experiments/tests/Objective/trials/experiment6/logs/experiment_data.csv'
data = pd.read_csv(file_path)
# Define the global reward function
def f(x1, x2, x3):
    y = np.exp(-x1**2-x2**2)*np.cos(x3)
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
        plt.title(f'Agent {self.id}')


        # Define function to process each experiment's data
def extract_data(experiment_data, lambda1, lambda2, lambda3):
    # Extract X and Z values
    X_values = experiment_data[['X1', 'X2', 'X3']].values
    Z_values = experiment_data[['Z1', 'Z2', 'Z3']].values

    filtered_X = X_values[(experiment_data['Lambda1'] == lambda1) &
                            (experiment_data['Lambda2'] == lambda2) &
                            (experiment_data['Lambda3'] == lambda3)]
    filtered_Z = Z_values[(experiment_data['Lambda1'] == lambda1) &
                            (experiment_data['Lambda2'] == lambda2) &
                            (experiment_data['Lambda3'] == lambda3)]
    
    
    return filtered_X, filtered_Z

def run_normal_experiment(agent1,agent2,agent3,N):  

    for i in range(N):

        x1_next = agent1.optimize()
        x2_next = agent2.optimize()
        x3_next = agent3.optimize()


        y = f(x1_next, x2_next, x3_next)

        agent1.update(x1_next,y)
        agent2.update(x2_next,y)
        agent3.update(x3_next,y)


def run_experiments(agent1,agent2,agent3,N):

    actions_1 = np.array([])
    actions_2 = np.array([])
    actions_3 = np.array([])

    for i in range(N):

        z1_next = agent1.optimize()
        z2_next = agent2.optimize()
        z3_next = agent3.optimize()

        x1_next = model_Z_X_0.predict_noiseless(z1_next.reshape(-1,1))[0]
        x2_next = model_Z_X_1.predict_noiseless(z2_next.reshape(-1,1))[0]
        x3_next = model_Z_X_2.predict_noiseless(z3_next.reshape(-1,1))[0]  

        x1_next = np.asarray([x1_next]).flatten()
        x2_next = np.asarray([x2_next]).flatten()
        x3_next = np.asarray([x3_next]).flatten()
        

        actions_1 = np.append(actions_1, x1_next)
        actions_2 = np.append(actions_2, x2_next)
        actions_3 = np.append(actions_3, x3_next)


        y = f(x1_next, x2_next, x3_next)

        agent1.update(z1_next,y)
        agent2.update(z2_next,y)
        agent3.update(z3_next,y)

    return actions_1, actions_2, actions_3


# Define function to process and plot data for specific lambda parameters
def plot_data_for_lambda(data, lambda1, lambda2, lambda3):
    # Filter data for specific lambda values
    group_data = data[(data['Lambda1'] == lambda1) & 
                      (data['Lambda2'] == lambda2) & 
                      (data['Lambda3'] == lambda3)]
    return group_data

if __name__ == '__main__':

    grouped = data.groupby(['Lambda1', 'Lambda2', 'Lambda3'])

    lambda1 = 0.0
    lambda2 = 0.0
    lambda3 = 0.0

    X, Z = extract_data(data, lambda1, lambda2, lambda3)

    
    agent1 = Agent(1, (-1, 1), 0.1)
    agent2 = Agent(2, (-1, 1), 0.2)
    agent3 = Agent(3, (-1, 1), 0.3)


    model_Z_X_0 = GPy.models.GPRegression(Z[:,0].reshape(-1,1), X[:,0].reshape(-1,1), GPy.kern.RBF(1))
    model_Z_X_1 = GPy.models.GPRegression(Z[:,1].reshape(-1,1), X[:,1].reshape(-1,1), GPy.kern.RBF(1))
    model_Z_X_2 = GPy.models.GPRegression(Z[:,2].reshape(-1,1), X[:,2].reshape(-1,1), GPy.kern.RBF(1))


    model_X_Z_0 = GPy.models.GPRegression(X[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), GPy.kern.RBF(1))
    model_X_Z_1 = GPy.models.GPRegression(X[:,1].reshape(-1,1), Z[:,1].reshape(-1,1), GPy.kern.RBF(1))
    model_X_Z_2 = GPy.models.GPRegression(X[:,2].reshape(-1,1), Z[:,2].reshape(-1,1), GPy.kern.RBF(1))

    actions_1, actions_2, actions_3 = run_experiments(agent1,agent2,agent3,10)


    agent1.plot_opt()
    agent2.plot_opt()
    agent3.plot_opt()


    run_normal_experiment(agent1,agent2,agent3,10)
    plt.show()



 


