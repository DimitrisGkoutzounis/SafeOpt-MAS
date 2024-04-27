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

np.set_printoptions(precision=3)
# np.random.seed(0)


def rbf_covariance(X1,X2,variance,lengthscale):
    kernel = kernels.RBF(length_scale=lengthscale)
    noise_term = np.eye(X1.shape[0]) * 1
    return variance * kernel(X1,X2) + noise_term   

def likelihood(params, X, variance, lengthscale, latent_dim,alpha=100):
    N, D = X.shape

    # Z = np.reshape(params, (N, latent_dim)) #latent variables
    Z = np.array([params]).reshape((N, latent_dim))
    K_zz = rbf_covariance(Z, Z, variance, lengthscale)  # latent variable covariance matrix

    k_inv = np.linalg.inv(K_zz)
    XXT = np.dot(X, X.T)

    trace = np.sum(np.dot(k_inv, XXT))
    log_det_K_zz = np.log(np.linalg.det(K_zz)) # log determinant of K_zz

    #regularization term that penaltizes off-diagonal elements of the covariance matrix
    off_diag_K_zz = K_zz - np.diag(np.diag(K_zz))
    penalty = alpha * np.linalg.norm(off_diag_K_zz, ord='fro')**2

    obj = 0.5 * D * log_det_K_zz + 0.5 * trace + penalty
    return obj  # Minimizing negative likelihood (maximizing likelihood)

def predict(X, Z, variance, lengthscale, latent_dim):
    
    #latent space covariance matrix - active latent variables
    K_zz = rbf_covariance(Z, Z, variance, lengthscale)
    K_zz_inv = np.linalg.inv(K_zz)

    jth = np.array(Z[:,0]).reshape(-1,1)
    print("jth\n",jth)

    #covariance between latent space and jth column of latent space
    K_zj = rbf_covariance(Z,jth,1,1)
    print("K_zj\n",K_zj)

    mean = np.dot(np.dot(K_zz_inv,K_zj),X)

    print("Mean\n",mean)

    #variance - k(x_j,x_j) - k_zj.T * K_zz_inv * K_zj
    k_zj_var = rbf_covariance(jth,jth,1,1)
    variance = k_zj_var - np.dot(K_zj.T,np.dot(K_zz_inv,K_zj))
    print("Variance\n",variance)

    #plot the variance
    

    return mean,variance

if __name__ == '__main__':


    latent_dim = 1
    n = 5

    x1 = np.random.uniform(-1.5,1.5,n)
    x2 = np.random.uniform(-1.5,1.5,n)
    x3 = np.random.uniform(-1.5,1.5,n)
    
    print("x1\n",x1)

    X = np.vstack([x1, x2,x3]).T

    print("Input Data\n",X)

    params=np.random.normal(size=(n*latent_dim)) #initialize randomly latent variables
    obj = minimize(likelihood,params,args=(X,1,1,latent_dim),method='CG',tol=0.001,options={'maxiter':None,'disp':False})

    Z = np.reshape(obj.x, (n, latent_dim))
    print("Latent Space:\n",Z)

    Kyy = rbf_covariance(X,X,1,1)
    print("Input Space Covariance Matrix\n",Kyy)
    Kyy_inv = np.linalg.inv(Kyy)

    Kzz = rbf_covariance(Z,Z,1,1)
    print("--Latent Space Covariance Matrix\n",Kzz)




    mean , variance = predict(X, Z, 1, 1, latent_dim)


    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(Kzz, annot=True, fmt=".2f", cmap="viridis")
    plt.title('Latent Space Covariance Matrix $K_{zz}$')

    plt.subplot(2, 2, 2)
    sns.heatmap(Kyy, annot=True, fmt=".2f", cmap="viridis")
    plt.title('Input Space Covariance Matrix $K_{yy}$')

    plt.subplot(2, 2, 3)
    sns.heatmap(variance, annot=True, fmt=".2f", cmap="viridis")
    plt.title('Output Variance $K_{jj} - K_{zj}^T K_{zz}^{-1} K_{zj}$')

    plt.subplot(2, 2, 4)
    sns.heatmap(mean, annot=True, fmt=".2f", cmap="viridis")
    plt.title('Predicted mean $K_{zj}^T K_{zz}^{-1}')


    plt.show()


    # #select the jth column of Z
    # j_column = np.asarray([Z[:,0]]).T
    # print("j_column of Latent Space\n",j_column)

    # K_zj = rbf_covariance(X,j_column,1,1)
    # print("Covariance between latent space and jth column of latent space\n",K_zj)

    # mean = np.dot(X.T,np.dot(Kyy_inv,K_zj))
    # print("Mean\n",mean)

    # variance = Kzz - np.dot(K_zj.T,np.dot(Kyy_inv,K_zj))
    # print("Variance\n",variance)

    # k_zj_var = rbf_covariance(j_column,j_column,1,1)

    # input_jth = X[:,0]
    # input_jth_transpose = np.asarray([input_jth]).T
    # print("Input jth column\n",input_jth_transpose)
    # print("Shape is ",input_jth_transpose.shape)

    # K_Yj= rbf_covariance(X,input_jth,1,1)
    # print("Covariance between input space and jth column of input space\n",K_Yj)
    

    #covariance between 2 points

    # j_Z = Z[:,0]
    # j_Z = np.asarray([j_Z]).T

    # j_X = X[:,1]
    # j_X = np.asarray([j_X]).T
    # # The covariance matrix between the latent variables Z and the original data X
    # K_zx = rbf_covariance(j_X, j_Z, variance=1, lengthscale=1)
    # print("Covariance between latent space and input space\n",K_zx)


    # # The mean of X given Z in the latent space using the Gaussian process
    # # For a GP with a zero mean function, the mean is given by: K_zx * K_zz_inv * X
    # K_zz_inv = np.linalg.inv(Kzz)
    # mean_X_given_Z = np.dot(K_zx.T, np.dot(K_zz_inv, X))

    # print("Mean of Data Space X given Latent Space Z:\n", mean_X_given_Z.T)

    # # The variance of X given Z in the latent space using the Gaussian process
    # # For a GP with a zero mean function, the variance is given by: K_xx - K_zx * K_zz_inv * K_zx.T
    # variance_X_given_Z = Kyy - np.dot(K_zx, np.dot(K_zz_inv, K_zx.T))

    # print("Variance of Data Space X given Latent Space Z:\n", variance_X_given_Z)


























   