import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from autograd import grad, value_and_grad
import autograd.numpy as np
from autograd.misc.optimizers import adam
import autograd.scipy.stats.multivariate_normal as mvn
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from sklearn.gaussian_process import kernels


def rbf_covariance(X1,X2,variance,lengthscale):
    kernel = kernels.RBF(length_scale=lengthscale)
    return variance * kernel(X1,X2) + np.eye(X1.shape[0]) * 1

def likelihood(params, X, variance, lengthscale, latent_dim):
    N, D = X.shape

    # Z = np.reshape(params, (N, latent_dim)) #latent variables
    Z = np.array([params]).reshape((N, latent_dim))
    K_zz = rbf_covariance(Z, Z, variance, lengthscale)  # latent variable covariance matrix

    k_inv = np.linalg.inv(K_zz)
    XXT = np.dot(X, X.T)

    trace = np.sum(np.dot(k_inv, XXT))
    log_det_K_zz = np.log(np.linalg.det(K_zz)) # log determinant of K_zz

    obj = 0.5 * D * log_det_K_zz + 0.5 * trace
    return obj  # Minimizing negative likelihood (maximizing likelihood)



if __name__ == '__main__':


    latent_dim = 1
    n = 3

    x1 = np.random.uniform(0,1,n)
    x2 = np.array([0.3,0.3])

    X = np.vstack([x1, x2]).T

    params=np.random.normal(size=(n*latent_dim)) #initialize randomly latent variables
    obj = minimize(likelihood,params,args=(X,1,1,latent_dim),method='CG',tol=0.001,options={'maxiter':None,'disp':False})

    Z = np.reshape(obj.x, (n, latent_dim))
    print(Z)
    print("z clumn",Z[:,0])


    K_II = rbf_covariance(X,X,1,1)
    print("Input Space Covariance\n", K_II)

    K_ZZ =rbf_covariance(Z,Z,1,1)
    print("Latent Space Covaraince\n", K_ZZ)
    K_II_inv = np.linalg.inv(K_II)
    j_column = np.asarray([Z[:,0]]).T
    
    K_IJ = rbf_covariance(X,j_column.reshape(1,n),1,1)
    

    Y_transpose = X.T

    mean = np.dot(Y_transpose,np.dot(K_II_inv,K_IJ))

    print("mean",mean)