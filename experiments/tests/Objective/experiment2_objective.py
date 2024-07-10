import numpy as np
import GPy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimparallel import minimize_parallel
import logging
import os
import pandas as pd

np.random.seed(2)

# Define the global reward function
def f(x1, x2, x3):
    y = np.exp(-x1**2-x2**2)*np.cos(x3)
    return y


def generate_actions(N):
    x1 = np.random.uniform(-1, 1, N)
    x2 = np.random.uniform(-1, 1, N)
    x3 = np.random.uniform(-1, 1, N)
    R = f(x1, x2, x3)
    return x1, x2, x3, R

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
    Z = Z_flat.reshape(N, D)

    rbf_kernel = GPy.kern.RBF(D)

    # Define model_Z with R_z as observations
    model_Z = GPy.models.GPRegression(Z, R.reshape(-1,1), rbf_kernel.copy())
    model_all = GPy.models.GPRegression(Z, X,  rbf_kernel.copy())
    mu_all, _ = model_all.predict_noiseless(Z)

    
    # Initialize matrices for U_z and U_x
    U_z = np.zeros((N, D))
    U_x = np.zeros((N, D))

    grad_R_Z = compute_gradient(model_Z, Z).reshape(N, D)
    grad_R_X = compute_gradient(model_X, X).reshape(N, D)

    action_term = 0.0

    for d in range(D):
        X_d = np.zeros_like(X)
        X_d[:, d] = X[:, d]
        
        model_d = GPy.models.GPRegression(Z, X_d,rbf_kernel.copy())
        mu_d, _ = model_d.predict_noiseless(Z)

        diff1 = np.linalg.norm(X_d - mu_d)**2
        diff2 = np.linalg.norm(mu_d - mu_all[:, [d]])**2
        
        action_term += lambda1 * diff1 + lambda2 * diff2

        # create unit vector matrices
        U_z[:, d] = grad_R_Z[:, d] / np.linalg.norm(grad_R_Z[:, d])
        U_x[:, d] = grad_R_X[:, d] / np.linalg.norm(grad_R_X[:, d])

    #compute the dot product matrix
    dot_product_matrix = np.dot(U_z.T, U_x)
    gradient_term = np.linalg.norm((1 - np.diag(dot_product_matrix))**2)/D
    trace_term = np.linalg.norm((1 - (np.trace(dot_product_matrix))))/D
    # print("Trace(opt): ", (1-np.trace(dot_product_matrix)/D))
    
    computed_z = action_term + lambda3 * trace_term 


    return computed_z



# Ensure directories
log_directory = "logs"
plot_directory = "plots"
os.makedirs(log_directory, exist_ok=True)
os.makedirs(plot_directory, exist_ok=True)

if __name__ == '__main__':
    N = 20
    D = 3
    lambda_values =[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    csv_path = os.path.join(log_directory, 'experiment_data.csv')

    # Generate actions
    X1, X2, X3, R_original = generate_actions(N)
    X = np.vstack((X1, X2, X3)).T
    Z = np.random.uniform(-1.5, 1.5, (N, D))

    total_experiments = len(lambda_values)**3
    experiment_number = 0

 
    for lambda1 in lambda_values:
         for lambda2 in lambda_values:
                for lambda3 in lambda_values:
                    experiment_number += 1
                    print("Experiemnt number: ",experiment_number)
                    print(f"Running for lambda1={lambda1}, lambda2={lambda2}, lambda3={lambda3}, Seed={experiment_number}")


                    print(f"X: {X}")
                    print(f"Z: {Z}")

                    # Perform experiments
                    R = f(X1, X2, X3)
                    R_Z_init = f(Z[:,0], Z[:,1], Z[:,2])

                    model_X = GPy.models.GPRegression(X, R_original[:, None], GPy.kern.RBF(input_dim=D))
                    model_Z_init = GPy.models.GPRegression(Z, R_Z_init[:, None], GPy.kern.RBF(input_dim=D))
                    
                    #OPTIMIZE
                    result = minimize(column_wise, Z.flatten(), args=(X, D, N, f, lambda1, lambda2, lambda3), method='L-BFGS-B', options={'ftol': 1e-2, 'gtol': 1e-2, 'xtol': 1e-2})

                    Z_opt = result.x.reshape(N, D)
                    R_Z_opt = f(Z_opt[:, 0], Z_opt[:, 1], Z_opt[:, 2])
                    model_Z_opt = GPy.models.GPRegression(Z_opt, R_Z_opt[:, None], GPy.kern.RBF(input_dim=D))

                    U_x_init, U_z_init,trace_before = compute_trace(model_X, model_Z_init, X, Z)
                    U_X_opt, U_Z_opt,trace_after = compute_trace(model_X, model_Z_opt, X, Z_opt)

                    data = {
                        'Experiment': [experiment_number] * N,
                        'Lambda1': [lambda1] * N,
                        'Lambda2': [lambda2] * N,
                        'Lambda3': [lambda3] * N,
                        'X1': X[:, 0],
                        'X2': X[:, 1],
                        'X3': X[:, 2],
                        'Z1': Z_opt[:, 0],
                        'Z2': Z_opt[:, 1],
                        'Z3': Z_opt[:, 2],
                        'R': R,
                        'R_Z_opt': R_Z_opt,
                        'Trace_before': trace_before,
                        'Trace_after': trace_after
                    }
                    df = pd.DataFrame(data)
                    # Append to CSV, create if does not exist
                    if not os.path.exists(csv_path):
                        df.to_csv(csv_path, index=False, mode='w', header=True)
                    else:
                        df.to_csv(csv_path, index=False, mode='a', header=False)

                    # Plot and save the plot
                    plt.figure(figsize=(15, 6))
                    plt.plot(R, label='R(x)', marker='o', markersize=5)
                    plt.plot(R_Z_opt, label='R(z)', marker='x', markersize=5)
                    plt.title(f'R(x) vs R(z) - Lambda1={lambda1}, Lambda2={lambda2}, Lambda3={lambda3}')
                    plt.xlabel('Sample')
                    plt.ylabel('Reward')
                    plt.legend()
                    plot_path = os.path.join(plot_directory, f'plot_exp_{experiment_number}_lambda1_{lambda1}_lambda2_{lambda2}_lambda3_{lambda3}.png')
                    plt.savefig(plot_path)
                    plt.close()
