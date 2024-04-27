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
    return np.array(alpha*kernel(X, Y) + np.eye(X.shape[0])/(beta**2))

def likelihood(var, *args):
    YYT, N, D, latent_dimension, = args

    X = np.array(var[:-3]).reshape((N, latent_dimension))
    alpha = var[-3]
    beta = var[-2]
    gamma = var[-1]
    K = kernel(X, X, alpha, beta, gamma)

    # return -log likelihood
    trace = np.sum(np.dot(np.linalg.inv(K), YYT))
    return D*np.log(np.abs(np.linalg.det(K)))/2 + trace/2



def simple_gplvm(Y, experiment_name="experiment", latent_dimension=1):
    ''' Implementation of GPLVM algorithm, returns data in latent space
    '''
    # Initialize X through PCA

    latent_params = np.random.normal(0, 1, (Y.shape[0], latent_dimension))  # Initialize latent space
    print("latent_params",latent_params)
    
    kernel_params = np.ones(3)  # (alpha, beta, gamma) TODO(oleguer): Should we rewrite those at each iteration? I dont thinkn so

    var = list(latent_params.flatten()) + list(kernel_params)
    YYT = np.dot(Y,Y.T)
    N = Y.shape[0]
    D = Y.shape[1]

    
    var = fmin_cg(likelihood, var, args=tuple((YYT,N,D,latent_dimension,)), epsilon = 0.001, disp=True)

    var = list(var)

    N = Y.shape[0]
    X = np.array(var[:-3]).reshape((N, latent_dimension))
    alpha = var[-3]
    beta = var[-2]
    gamma = var[-1]

    print("alpha", alpha)
    print("beta", beta)
    print("gamma", gamma)

    return X

# Generate data
if __name__ == "__main__":
    N = 4  # Number of observations

    x1 = np.random.uniform(-1.5, 1.5, N)
    x2 = np.random.uniform(-1.5, 1.5, N)
    x3 = np.random.uniform(-1.5, 1.5, N)

    #make one matrix with n rows and D columns
    observations = np.vstack([x1, x2, x3]).T
    print("rows and columns",observations.shape)

    print("observations",observations)

    gp_vals = simple_gplvm(Y=observations, experiment_name="test")  # Compute values
    # gp_vals = np.array(list(np.load("results/var.npy"))[:-3]).reshape((N, 2))  # Load from memory


    K_zz = kernel(gp_vals,gp_vals,1,1,1)
    print("Kernel\n",K_zz)
