import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from autograd import grad, value_and_grad
import autograd.numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF
import safeopt

class GPLVM:
    def __init__(self, latent_dim, variance, lengthscale, alpha):
        self.latent_dim = latent_dim
        self.variance = variance
        self.lengthscale = lengthscale
        self.alpha = alpha  # Regularization parameter

    def rbf_covariance(self, X1, X2):
        kernel = RBF(length_scale=self.lengthscale)
        noise_term = np.eye(X1.shape[0]) * 1
        return self.variance * kernel(X1, X2) + noise_term

    def likelihood(self, params, X):
        N, D = X.shape
        Z = np.array([params]).reshape((N, self.latent_dim))
        K_zz = self.rbf_covariance(Z, Z)

        k_inv = np.linalg.inv(K_zz)
        XXT = np.dot(X, X.T)

        trace = np.sum(np.dot(k_inv, XXT))
        log_det_K_zz = np.log(np.linalg.det(K_zz))

        # Regularization term penalizes off-diagonal elements of the covariance matrix
        off_diag_K_zz = K_zz - np.diag(np.diag(K_zz))
        penalty = self.alpha * np.linalg.norm(off_diag_K_zz, ord='fro')**2

        return 0.5 * D * log_det_K_zz + 0.5 * trace + penalty

    def fit(self, X):
        params_init = np.random.normal(size=(X.shape[0] * self.latent_dim))
        res = minimize(self.likelihood, params_init, args=(X,), method='CG', tol=0.001,
                       options={'maxiter':None, 'disp': False})
        self.Z = np.reshape(res.x, (X.shape[0], self.latent_dim))
        return self.Z

    def predict(self, X):
        K_zz = self.rbf_covariance(self.Z, self.Z)
        K_zz_inv = np.linalg.inv(K_zz)

        jth = np.array(self.Z[:,0]).reshape(-1,1)
        K_zj = self.rbf_covariance(self.Z, jth)
        mean = np.dot(np.dot(K_zz_inv, K_zj), X)

        k_zj_var = self.rbf_covariance(jth, jth)
        variance = k_zj_var - np.dot(K_zj.T, np.dot(K_zz_inv, K_zj))
        return mean, variance

if __name__ == '__main__':
    np.random.seed(0)
    latent_dim = 1
    n = 30

    # Data generation with three distinct groups
    x1 = np.random.normal(-5, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(5, 1, n)
    X = np.vstack([x1, x2, x3]).T

    # Initialize and fit the model
    gplvm = GPLVM(latent_dim=latent_dim, variance=1, lengthscale=1, alpha=100)
    Z = gplvm.fit(X)

    # Make predictions
    mean, variance = gplvm.predict(X)

    # Visualization
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='x1 vs x2')
    plt.scatter(X[:, 0], X[:, 2], c='red', label='x1 vs x3')
    plt.scatter(X[:, 1], X[:, 2], c='green', label='x2 vs x3')
    plt.xlabel('Input Features X')
    plt.ylabel('Input Features X')
    plt.legend()
    plt.title('Input Features X vs Input Features X')

    plt.subplot(1, 2, 2)
    plt.scatter(Z[:, 0], X[:, 0], c='blue', label='Latent vs x1')
    plt.scatter(Z[:, 0], X[:, 1], c='red', label='Latent vs x2')
    plt.scatter(Z[:, 0], X[:, 2], c='green', label='Latent vs x3')
    plt.xlabel('Latent Features Z')
    plt.ylabel('Input Features X')
    plt.legend()

    plt.show()

