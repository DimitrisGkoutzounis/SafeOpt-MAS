import numpy as np
import GPy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Define the global reward function
def f(x1, x2, x3):
    y = np.sin(x1**3) + np.cos(x2**2) - np.sin(x3)
    return y

# Define the function to run experiments
def run_experiments(a1, a2, a3, iterations):
    rewards = np.array([])
    actions_1 = np.array([])
    actions_2 = np.array([])
    actions_3 = np.array([])

    for i in range(iterations):
        x1 = a1.optimize()
        x2 = a2.optimize()
        x3 = a3.optimize()

        actions_1 = np.append(actions_1, x1)
        actions_2 = np.append(actions_2, x2)
        actions_3 = np.append(actions_3, x3)

        y = f(x1, x2, x3)
        rewards = np.append(rewards, y)

        # 50% of the time maximize, 50% of the time minimize (by changing the sign of the reward)
        if i < iterations // 2:
            a1.update(x1, y)
            a2.update(x2, y)
            a3.update(x3, y)
        else:
            a1.update(x1, -y)
            a2.update(x2, -y)
            a3.update(x3, -y)

    return actions_1, actions_2, actions_3, rewards


def generate_actions(N):
    x1 = np.random.uniform(-1.5, 1.5, N)
    x2 = np.random.uniform(-1.5, 1.5, N)
    x3 = np.random.uniform(-1.5, 1.5, N)
    R = f(x1, x2, x3)
    return x1, x2, x3, R

def compute_gradient(model, X):
    dmu_dX, _ = model.predictive_gradients(X)
    return dmu_dX


def create_U_matrice_columnwise(model_X,model_Z,X,Z):
    N = X.shape[0]
    D = X.shape[1]

    U_x = np.zeros((N, D))
    U_z = np.zeros((N, D))

    grad_R_X_norm_column = []
    grad_R_Z_norm_column = []

    for d in range(D):
        grad_R_X = compute_gradient(model_X, X)
        grad_R_Z = compute_gradient(model_Z, Z)

        grad_R_X = grad_R_X.reshape(N, D)
        grad_R_Z = grad_R_Z.reshape(N, D)

        grad_R_Z_norm_column.append(np.linalg.norm(grad_R_Z[:, d]))
        grad_R_X_norm_column.append(np.linalg.norm(grad_R_X[:, d]))

        U_z[:, d] = grad_R_Z[:, d] / grad_R_Z_norm_column[d]
        U_x[:, d] = grad_R_X[:, d] / grad_R_X_norm_column[d]

    dot_product_matrix = np.dot(U_z.T, U_x)
    print('dot_product_matrix\n', dot_product_matrix)
    print("Trace:", np.sum(np.diag(dot_product_matrix))/D)

    return U_x, U_z


def plot_3d_vector_column_wise(U_x, U_z):
    N = U_x.shape[0]
    D = U_x.shape[1]

    print("U_x\n", U_x)
    print("U_z\n", U_z)
    # Create a single figure and axis for all column vectors
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('U_x and U_z vectors')

    colors = ['r', 'g', 'b']
    for i in range(D):
        # Extract columns from U_x and U_z
        v1 = U_x[:, i]
        v2 = U_z[:, i]

        # Origin point for vectors
        origin = [0, 0, 0]
        # Plot each vector
        ax.quiver(*origin, *v1, color=colors[i], length=1.0, label=f'U_x_{i}: {v1}')
        ax.quiver(*origin, *v2, color=colors[i], linestyle='dashed', length=1.0, label=f'U_z_{i}: {v2}')

    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Show legend
    ax.legend()

    # Display the plot
    plt.show()

def column_wise(Z_flat, X, D, N, sigma2, f):
    Z = Z_flat.reshape(N, D)

    # Define model_Z with R_z as observations
    model_Z = GPy.models.GPRegression(Z, R.reshape(-1,1), GPy.kern.RBF(D))
    model_all = GPy.models.GPRegression(Z, X,  GPy.kern.RBF(D))
    mu_all, _ = model_all.predict_noiseless(Z)

    loss = 0.0
    grad_R_Z_norm_column = []
    grad_R_X_norm_column = []

    # Initialize matrices for U_z and U_x
    U_z = np.zeros((N, D))
    U_x = np.zeros((N, D))

    for d in range(D):
        X_d = np.zeros_like(X)
        X_d[:, d] = X[:, d]
        
        model_d = GPy.models.GPRegression(Z, X_d,GPy.kern.RBF(D))
        mu_d, _ = model_d.predict_noiseless(Z)

        diff1 = np.linalg.norm(X_d - mu_d)**2
        diff2 = np.linalg.norm(mu_d - mu_all[:, [d]])**2
        
        loss += diff1 + 0.2 * diff2

        # Gradient-based alignment term
        grad_R_Z = compute_gradient(model_Z, Z).reshape(N, D)
        grad_R_X = compute_gradient(model_X, X).reshape(N, D)

        grad_R_Z_norm_column.append(np.linalg.norm(grad_R_Z[:, d]))
        grad_R_X_norm_column.append(np.linalg.norm(grad_R_X[:, d]))

        U_z[:, d] = grad_R_Z[:, d] / grad_R_Z_norm_column[d]
        U_x[:, d] = grad_R_X[:, d] / grad_R_X_norm_column[d]


    # Compute the dot product matrix after the loop
    dot_product_matrix = np.dot(U_z.T, U_x)
    diag_penalty = np.linalg.norm((1 - np.diag(dot_product_matrix)))/D

    # mu_z_est , _ = model_Z.predict_noiseless(Z)
    # mu_x_est , _ = model_X.predict_noiseless(X)

    # diff3 = np.linalg.norm(mu_x_est - mu_z_est)**2
    
    total_loss = loss + diag_penalty 


    return total_loss



if __name__ == '__main__':
    N=3
    D=3

    X1, X2,X3, R_original = generate_actions(N)
    X = np.vstack((X1, X2,X3)).T

    Z = np.random.uniform(-1.5, 1.5, (N, D))


    #perfrom experiments
    R = f(X1, X2,X3)
    R_Z_init = f(Z[:,0], Z[:,1], Z[:,2])

    model_X = GPy.models.GPRegression(X, R_original[:, None], GPy.kern.RBF(input_dim=D))
    model_Z_init = GPy.models.GPRegression(Z, R_Z_init[:, None], GPy.kern.RBF(input_dim=D))

    U_x, U_z = create_U_matrice_columnwise(model_X,model_Z_init,X,Z)

    result = minimize(column_wise, Z.flatten(), args=(X, D, N, 1e-2, f), method='L-BFGS-B')

    Z_opt = result.x.reshape(N, D)

    R_Z_opt = f(Z_opt[:, 0], Z_opt[:, 1], Z_opt[:, 2])
    
    #model Z_opt-> R
    model_Z_opt = GPy.models.GPRegression(Z_opt, R_Z_opt[:, None], GPy.kern.RBF(input_dim=D))

    U_X_opt, U_Z_opt = create_U_matrice_columnwise(model_X, model_Z_opt, X, Z_opt)



print("U_x\n", U_x)
print("U_z\n", U_z)
print("U_X_opt\n", U_X_opt)
print("U_Z_opt\n", U_Z_opt)

plt.figure(figsize=(15, 6))
plt.plot(R, label='R(x)', marker='o', markersize=5)
plt.plot(R_Z_opt, label='R(z)', marker='x', markersize=5)
plt.title('R(x) vs R(z)')
plt.xlabel('Sample')
plt.ylabel('Reward')
plt.legend()
plt.show()
