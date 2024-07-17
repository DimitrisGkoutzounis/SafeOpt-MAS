import numpy as np
import GPy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimparallel import minimize_parallel
import logging
import os
import pandas as pd

# np.random.seed(2)
J1_history = []
J2_history = []
J3_history = []
Total_cost_history = []


# Define the global reward function
def f(x1, x2):
    y = x1 - x2
    return y


def generate_actions(N):
    x1 = np.random.uniform(-1, 1, N)
    x2 = np.random.uniform(-1, 1, N)
    # x3 = np.random.uniform(-1, 1, N)
    R = f(x1, x2)
    return x1, x2,R

def compute_gradient(model, X):
    dmu_dX, _ = model.predictive_gradients(X)
    return dmu_dX


def compute_trace(model_X,model_Z,X,Z):
    N = X.shape[0]
    D = X.shape[1]

    U_x = np.zeros((N, D))
    U_z = np.zeros((N, D))
    grad_R_X = compute_gradient(model_X, X)
    grad_R_Z = compute_gradient(model_Z, Z)

    for d in range(D):

        grad_R_X = grad_R_X.reshape(N, D)
        grad_R_Z = grad_R_Z.reshape(N, D)

         # create unit vector matrices
        U_z[:, d] = grad_R_Z[:, d] / np.linalg.norm(grad_R_Z[:, d])
        U_x[:, d] = grad_R_X[:, d] / np.linalg.norm(grad_R_X[:, d])

    dot_product_matrix = np.dot(U_z.T, U_x)
    trace= np.trace(dot_product_matrix)

    return U_x, U_z,trace



def column_wise(Z_flat, X, D, N, f,lambda1,lambda2,lambda3):
    Z_local = Z_flat.reshape(N, D)

    rbf_kernel = GPy.kern.RBF(D)

    # Define model_Z with R_z as observations
    model_Z = GPy.models.GPRegression(Z_local, R.reshape(-1,1), rbf_kernel.copy())
    model_all = GPy.models.GPRegression(Z_local, X,  rbf_kernel.copy())
    mu_all, _ = model_all.predict_noiseless(Z_local)

    
    # Initialize matrices for U_z and U_x
    U_z = np.zeros((N, D))
    U_x = np.zeros((N, D))

    grad_R_Z = compute_gradient(model_Z, Z_local).reshape(N, D)
    grad_R_X = compute_gradient(model_X, X).reshape(N, D)

    action_term = 0.0
    cost3 = 0.0

    for d in range(D):
        X_d = np.zeros_like(X)
        X_d[:, d] = X[:, d]
        
        model_d = GPy.models.GPRegression(Z_local, X_d,rbf_kernel.copy())
        mu_d, _ = model_d.predict_noiseless(Z)

        cost1 = np.linalg.norm(X_d - mu_d)**2
        cost2 = np.linalg.norm(mu_d - mu_all[:, [d]])**2
        
        action_term += lambda1 * cost1 + lambda2 * cost2

        # create unit vector matrices
        U_z[:, d] = grad_R_Z[:, d] / np.linalg.norm(grad_R_Z[:, d])
        U_x[:, d] = grad_R_X[:, d] / np.linalg.norm(grad_R_X[:, d])

    #compute the dot product matrix
    dot_product_matrix = np.dot(U_z.T, U_x)
    # gradient_term = np.linalg.norm((1 - np.diag(dot_product_matrix))**2)/D
    cost3 = np.linalg.norm((1 - (np.trace(dot_product_matrix))))/D
    J1_history.append(cost1)
    J2_history.append(cost2)
    J3_history.append(cost3)
    # print("Trace(opt): ", (1-np.trace(dot_product_matrix)/D))
    
    computed_z = action_term + lambda3 * cost3 


    return computed_z



# Ensure directories
log_directory = "logs"
plot_directory = "plots"
os.makedirs(log_directory, exist_ok=True)
os.makedirs(plot_directory, exist_ok=True)

if __name__ == '__main__':
    N = 20
    D = 2
    lambda_values =[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lambda3 = 1.0
    csv_path = os.path.join(log_directory, 'experiment_data.csv')

    # Generate actions
    X1, X2, R_original = generate_actions(N)
    X = np.vstack((X1, X2)).T
    Z = np.random.uniform(-1.5, 1.5, (N, D))

    total_experiments = len(lambda_values)**3
    experiment_number = 0
    
    total_costs = np.zeros((len(lambda_values), len(lambda_values)))

 
    # Iterate over lambda1 and lambda2
    for i, lambda1 in enumerate(lambda_values):
        for j, lambda2 in enumerate(lambda_values):
            
            print(f"Experiment number: ", experiment_number)
            print(f"Running for lambda1={lambda1}, lambda2={lambda2}, lambda3={lambda3}")
            
            R = f(X1, X2)
            R_Z_init = f(Z[:, 0], Z[:, 1])
            
            model_X = GPy.models.GPRegression(X, R_original[:, None], GPy.kern.RBF(input_dim=D))
            model_Z_init = GPy.models.GPRegression(Z, R_Z_init[:, None], GPy.kern.RBF(input_dim=D))
            
            result = minimize(column_wise, Z.flatten(), args=(X, D, N, f, lambda1, lambda2, lambda3), method='L-BFGS-B', options={'ftol': 1e-2, 'gtol': 1e-2, 'xtol': 1e-2})
            total_costs[i, j] = result.fun
            
            
            Z_opt = result.x.reshape(N, D)
            
            R_Z_opt = f(Z_opt[:, 0], Z_opt[:, 1])
            R_Z_opt_max = R_Z_opt.max()
            
            print(f"Lambda1={lambda1}, Lambda2={lambda2}, Cost={result.fun}")
            
            data = {
                'Lambda1': lambda1,
                'Lambda2': lambda2,
                'Lambda3': lambda3,
                'total_cost': result.fun,
                'X1': X1,
                'X2': X2,
                'Z1': Z_opt[:, 0],
                'Z2': Z_opt[:, 1],
                'R_Z_opt': R_Z_opt,
                'R': R,
                'R(Z)_max': R_Z_opt_max
            }
            df = pd.DataFrame(data)
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False, mode='w', header=True)
            else:
                df.to_csv(csv_path, index=False, mode='a', header=False)

            
            

    # 3D plot of total costs
    lambda1_mesh, lambda2_mesh = np.meshgrid(lambda_values, lambda_values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(lambda1_mesh, lambda2_mesh, total_costs, cmap='viridis')
    ax.set_xlabel('Lambda1')
    ax.set_ylabel('Lambda2')
    ax.set_zlabel('Total Cost')
    ax.set_title('Total Cost for varying Lambda1 and Lambda2 with fixed Lambda3')
    plot_path = os.path.join(plot_directory, f'plot_exp_{experiment_number}_lambda1_{lambda1}_lambda2_{lambda2}_lambda3_{lambda3}.png')
    plt.savefig(plot_path)
    # plt.show()
