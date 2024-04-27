from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fmin_cg  # Non-linear SCG
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data
from sklearn.gaussian_process import kernels
import time
from tqdm import tqdm
# from fake_dataset import generate_observations, plot
from datetime import datetime

name = "experiemnt"
iteration = 0


def kernel(X, Y, alpha, beta, gamma):
    kernel = kernels.RBF(length_scale=(1./gamma**2))
    return np.matrix(alpha*kernel(X, Y) + np.eye(X.shape[0])/(beta**2))

def likelihood(var, *args):
    YYT, N, D, latent_dimension, = args

    X = np.array(var[:-3]).reshape((N, latent_dimension))
    alpha = var[-3]
    beta = var[-2]
    gamma = var[-1]
    K = kernel(X, X, alpha, beta, gamma)


    # return -log likelihood
    trace = np.sum(np.dot(K.I, YYT))
    return D*np.log(np.abs(np.linalg.det(K)))/2 + trace/2

def simple_gplvm(Y, experiment_name="experiment", latent_dimension=1):
    ''' Implementation of GPLVM algorithm, returns data in latent space
    '''
    global name
    name = experiment_name

    X= np.random.normal(0, 1, (Y.shape[0], latent_dimension))  # Initialize latent space

    print("Y shape:", Y.shape)
 
    kernel_params = np.ones(3)  # (alpha, beta, gamma) TODO(oleguer): Should we rewrite those at each iteration? I dont thinkn so

    var = list(X.flatten()) + list(kernel_params)
    YYT = np.dot(Y,Y.T)
    N = Y.shape[0]
    D = Y.shape[1]

    # Optimization
    t1 = time.time()
    var = fmin_cg(likelihood, var, args=tuple((YYT,N,D,latent_dimension,)), epsilon = 0.001, disp=True)
    print("time:", time.time() - t1)

    Z= np.array(var[:-3]).reshape((N, latent_dimension))

    return Z


if __name__ == "__main__":
    N = 2  # Number of observations
    D = 1  # Y dimension (observations)

    # x1= np.random.uniform(-1.5, 1.5, N)
    x1= np.array([0.5928,-1.3193])

    x2 =np.array([0.500,0.511])

    X_data = np.vstack([x1,x2]).T

    print("X_data shape:", X_data.size)

    gp_vals = simple_gplvm(Y=X_data, experiment_name="test")  # Compute values
    # gp_vals = np.array(list(np.load("results/var.npy"))[:-3]).reshape((N, 2))  # Load from memory

    K_II = kernel(X_data, X_data, 1, 1, 1)
    K_II_inv = np.linalg.inv(K_II)
    j_column = np.asarray([gp_vals[:,:]]).T

    K_IJ = kernel(X_data, j_column.reshape(1,N), 1, 1, 1)


    Y_transpose = X_data.T

    mean = np.dot(np.dot(K_IJ,K_II_inv),Y_transpose)

    print(mean)
