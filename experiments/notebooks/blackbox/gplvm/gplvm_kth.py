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
import safeopt
import GPy
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



def simple_gplvm(Y, experiment_name="experiment", latent_dimension=3):
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



class Agent:
    def __init__(self,id,bounds,safe_point):

        self.bounds = [bounds]
        self.id = id
        self.safepoint = safe_point
        self.global_rewards = np.array([])
        self.max_belief = np.array([[]])

        self.x0 = np.asarray([[self.safepoint]])
        self.y0 = np.asarray([[1]])
        print(self.y0)

        self.kernel = GPy.kern.RBF(1)
        self.gp = GPy.models.GPRegression(self.x0,self.y0, self.kernel, noise_var=0.05**2)
        self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)
        self.opt = safeopt.SafeOpt(self.gp, self.parameter_set, -np.inf,beta=4,threshold=0.2)

    def optimize(self):
        x_next = self.opt.optimize()
        return x_next
    
    def update(self,x_next,y_meas):
        self.max_belief = np.append(self.max_belief,self.opt.get_maximum()[1])
    
        self.opt.add_new_data_point(x_next,y_meas)

    def plot_opt(self):
        self.opt.plot(1000)

def run_experiments(a1,a2,a3,iterations):

    for i in range(iterations):
        x1 = a1.optimize()
        x2 = a2.optimize()
        x3 = a3.optimize()

        y = global_reward(x1,x2,x3)

        a1.update(x1,y)
        a2.update(x2,y)
        a3.update(x3,y)

    return  a1.gp.X, a2.gp.X, a3.gp.X


def global_reward(x1,x2,x3):
    y = np.sin(x1**3) + np.cos(x2**2) - x3
    return y


if __name__ == "__main__":
    N = 30  # Number of observations
    D = 1  # Y dimension (observations)

    agent1 = Agent(1, (-1.5, 1.5), 0)
    agent2 = Agent(2, (-1.5, 1.5), 0.1)
    agent3 = Agent(3, (-1.5, 1.5), 0.2)

    x1, x2, x3 = run_experiments(agent1, agent2, agent3, N)


    X_data = np.vstack([x1.T, x2.T, x3.T]).T
    print("X_data shape:", X_data)
    

    print("X_data shape:", X_data.size)

    gp_vals = simple_gplvm(Y=X_data, experiment_name="test")  # Compute values
    # gp_vals = np.array(list(np.load("results/var.npy"))[:-3]).reshape((N, 2))  # Load from memory

    #print latent space
    model1 = GPy.models.GPRegression(gp_vals[:,0].reshape(-1,1),agent1.gp.Y.reshape(-1,1), GPy.kern.RBF(1))
    model1.plot()
    model2 = GPy.models.GPRegression(gp_vals[:,1].reshape(-1,1),agent2.gp.Y.reshape(-1,1), GPy.kern.RBF(1))
    model2.plot()
    model3 = GPy.models.GPRegression(gp_vals[:,2].reshape(-1,1),agent3.gp.Y.reshape(-1,1), GPy.kern.RBF(1))
    model3.plot()
    plt.show()
