import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from autograd import grad, value_and_grad
import autograd.numpy as np
from autograd.misc.optimizers import adam
import autograd.scipy.stats.multivariate_normal as mvn
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF



inv = np.linalg.inv
import matplotlib

np.random.seed(0)

def plot_gp(X, fitted_Z, kernel):
    # Create a grid of values for Z for plotting GP, covering the range of fitted Z
    Z_grid = np.linspace(fitted_Z.min(), fitted_Z.max(), 500)[:, np.newaxis]

    # Calculate covariance matrices
    K = kernel(fitted_Z, fitted_Z) + np.eye(n) * 1e-5  # Add a small noise term for numerical stability
    K_s = kernel(fitted_Z, Z_grid)
    K_ss = kernel(Z_grid, Z_grid) + 1e-8 * np.eye(len(Z_grid))  # GP prior covariance

    # GP posterior mean and covariance
    K_inv = np.linalg.inv(K)
    mu_s = K_s.T @ K_inv @ X[:, -1]  # Posterior mean
    cov_s = K_ss - K_s.T @ K_inv @ K_s  # Posterior covariance

    # Standard deviation for uncertainty plot
    std_s = np.sqrt(np.diag(cov_s))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_Z, X[:, -1], c='k', label='Observed data')
    plt.plot(Z_grid, mu_s, 'b', label='GP mean')
    plt.fill_between(Z_grid[:, 0], mu_s - 1.96 * std_s, mu_s + 1.96 * std_s, color='blue', alpha=0.2, label='95% confidence interval')
    plt.title('Gaussian Process Regression over Latent Variable')
    plt.xlabel('Latent variable Z')
    plt.ylabel('Output')
    plt.legend()
    plt.show()


def make_data(n=100):
	x1 = np.random.uniform(-1.5, 1.5, n)
	x2 = np.random.uniform(-1.5, 1.5, n)
	x3 = np.random.uniform(-1.5, 1.5, n)

	X = np.sin(x1**3) + np.cos(x2**2) -np.sin(x3)
	X = np.vstack([x1, x2, x3]).T


	return X

def rbf_kernel(x1, x2):
	output_scale = 1
	lengthscales = 1
	diffs = np.expand_dims(x1 / lengthscales, 1)\
		  - np.expand_dims(x2 / lengthscales, 0)
	return output_scale * np.exp(-0.5 / 10 * np.sum(diffs**2, axis=2))

def gaussian_log_likelihood(X, mean, covariance, sigma2=1):
	#cr
 
	ll_list =list(mvn.logpdf(X[:, ii], mean, covariance) for ii in range(D))
	print(ll_list[:])

	ll = np.sum([mvn.logpdf(X[:, ii], mean, covariance) for ii in range(D)])
	return ll

def gaussian_prior(Z):
	covariance = np.eye(n)
	mean = np.zeros(n)
	ll = np.sum([mvn.logpdf(Z[:, ii], mean, covariance) for ii in range(d)])
	return ll

def objective(Z):
	# The objective is the negative log likelihood of the data.
	Z = np.reshape(Z, (n, d))
	cov_mat = kernel(Z, Z) + np.eye(n)
	ll = gaussian_log_likelihood(X, zero_mean, cov_mat)
	lprior = gaussian_prior(Z)
	return -(ll + lprior)

#total log likelihood array 
total_ll = []

fig = plt.figure(figsize=(12,8), facecolor='white')
data_ax = fig.add_subplot(121, frameon=False)
latent_ax = fig.add_subplot(122, frameon=False)
plt.show(block=False)

def callback(params):
	print('Log likelihood: {0:1.3e}'.format(-objective(params)))

	Z = np.reshape(params, (n, d))

	data_ax.cla()
	data_ax.scatter(X[:, 0], X[:, 1], c=Z[:, 0])
	data_ax.set_title('Observed Data')
	data_ax.set_xlabel(r"$x1$")
	data_ax.set_ylabel(r"$x2$")

	latent_ax.cla()
	latent_ax.plot(X[:, 0], Z[:, 0], 'kx')
	latent_ax.set_xlim([-2, 2])
	latent_ax.set_ylim([-2, 2])
	latent_ax.set_xlabel(r"$x1$")
	latent_ax.set_ylabel(r"$z$")
	latent_ax.set_title('Latent coordinates')

	plt.pause(1.0/60.0)

n = 3
D = 3 #data dimension
d = 1 #latent dimension
zero_mean = np.zeros(n)


X = make_data(n=n)


kernel = rbf_kernel
init_params = np.random.normal(size=(n * d))
print(init_params.shape)

res = minimize(value_and_grad(objective), init_params, jac=True, method='CG')
fitted_Z = np.reshape(res.x, (n, d))


# plt.show()
# plot_gp(X, fitted_Z, kernel)