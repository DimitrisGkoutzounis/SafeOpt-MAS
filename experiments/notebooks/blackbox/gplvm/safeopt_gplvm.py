import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from autograd import grad, value_and_grad
import autograd.numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF
import safeopt
import GPy




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
    y = np.sin(x1**3) + np.cos(x2**2) - np.sin(x3)
    return y

class GPLVM:
    def __init__(self, latent_dim, variance, lengthscale, alpha):
        self.latent_dim = latent_dim
        self.variance = variance
        self.lengthscale = lengthscale
        self.alpha = alpha  # Regularization parameter

    # def rbf_covariance(self, X1, X2):
    #     kernel = RBF(length_scale=self.lengthscale)
    #     noise_term = np.eye(X1.shape[0]) * 1
    #     return self.variance * kernel(X1, X2) + noise_term

    def rbf_covariance(self, X1, X2):
        kernel = RBF(length_scale=(1./self.lengthscale**2))
        print(kernel(X1,X2).ndim)
        return np.array(self.alpha * kernel(X1, X2) + np.eye(X1.shape[0])/(self.variance**2))

    def likelihood(self, params, X):
        N, D = X.shape
        Z = np.array([params]).reshape((N, self.latent_dim))
        
        K_zz = self.rbf_covariance(Z, Z)
        

        k_inv = np.linalg.inv(K_zz)
        XXT = np.dot(X, X.T)

        trace = np.sum(np.dot(k_inv, XXT))
        log_det_K_zz = np.log(np.linalg.det(K_zz))

        # Scale the penalty by the mean of the diagonal elements
        mean_diag = np.mean(np.diag(K_zz))
        off_diag_K_zz = K_zz - np.diag(np.diag(K_zz))
        penalty = self.alpha * np.sum((off_diag_K_zz / mean_diag) ** 2)


        return 0.5 * D * log_det_K_zz + 0.5 * trace + 0.5 * penalty

    def fit(self, X):
        params_init = np.random.normal(size=(X.shape[0] * self.latent_dim))
        res = minimize(self.likelihood, params_init, args=(X,), method='CG', tol=0.001,
                       options={'maxiter':None, 'disp': False})
        self.Z = np.reshape(res.x, (X.shape[0], self.latent_dim))
        return self.Z

    def predict(self, X):
        K_zz = self.rbf_covariance(self.Z, self.Z)
        K_zz_inv = np.linalg.inv(K_zz)

        print("Z is ", self.Z[:,0].reshape(-1,1))

        K_zj = self.rbf_covariance(self.Z[:,0].reshape(-1,1), self.Z[:,2].reshape(-1,1))
        K_jz = K_zj.T

        mean = np.dot(np.dot(K_zz_inv, K_zj), X)

        variance = self.rbf_covariance(Z,Z) - np.dot(np.dot(K_zj, K_zz_inv), K_jz)
        return mean, variance
    
    
    
    

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




if __name__ == '__main__':
    np.set_printoptions(precision=3)


    np.random.seed(0)
    latent_dim = 3
    n = 30

    Agent1 = Agent(1,(-1.5,1.5),0)
    Agent2 = Agent(2,(-1.5,1.5),0.1)
    Agent3 = Agent(3,(-1.5,1.5),-0.1)

    x1,x2,x3 = run_experiments(Agent1,Agent2,Agent3,n)

    y =Agent1.opt.y

    # # Data generation with three distinct groups
    # x1 = np.random.normal(-5, 1, n)
    # x2 = np.random.normal(0, 1, n)
    # x3 = np.random.normal(5, 1, n)
    

    X = np.vstack([x1.T, x2.T, x3.T]).T

    print("Agent2 actions\n",x2.shape)
    print("Agent3 actions\n",x3.shape)

    # print("Agent 1 beleif\n",Agent1.max_belief)
    # print("Agent 2 beleif\n",Agent2.max_belief)
    # print("Agent 3 beleif\n",Agent3.max_belief)

    print("X Space\n",X.shape)

    #Initialize and fit the model
    gplvm = GPLVM(latent_dim=latent_dim, variance=1, lengthscale=1, alpha=0)
    Z = gplvm.fit(X)

    mean, variance = gplvm.predict(X)

    plt.figure(figsize=(12, 6))
    sns.heatmap(variance, annot=True, fmt=".2f", cmap="viridis")
    plt.title('$\sigma = k(Z,Z) - K(z1,z2)^T K(Z,Z)^{-1}(k(z1,z2)$')
    plt.figure(figsize=(12, 6))
    kern = gplvm.rbf_covariance(Z[:,0].reshape(-1,1),Z[:,1].reshape(-1,1))
    sns.heatmap(kern, annot=True, fmt=".2f", cmap="viridis")
    plt.title('$k(z_1,z_2)$')

    zz = gplvm.rbf_covariance(Z,Z)
    plt.figure(figsize=(12, 6))
    sns.heatmap(zz, annot=True, fmt=".2f", cmap="viridis")  
    plt.title('K(Z,Z)') 
    plt.show()


    #average of the variance matrix 
    print("Average of the variance matrix\n",np.mean(variance - np.diag(np.diag(variance))))


    #plot the latent space
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.scatter(Z[:, 0], X[:,0], c='r', label='Z1 vs X1')
    plt.scatter(Z[:, 1], X[:,1], c='g', label='Z2 vs X2')
    plt.scatter(Z[:, 2], X[:,2], c='b', label='Z3 vs X3')
    plt.legend()
    plt.title('Latent Space')
    plt.xlabel("Z Latent Space")
    plt.ylabel("X Input Space")
    plt.show()


    #model of latent z1 and reward
    # plt.figure(figsize=(12, 6))
    model1 = GPy.models.GPRegression(Z[:,0].reshape(-1,1),y,GPy.kern.RBF(1))
    # model1.optimize()
    model2 = GPy.models.GPRegression(Z[:,1].reshape(-1,1),y,GPy.kern.RBF(1))
    # model2.optimize()
    model3 = GPy.models.GPRegression(Z[:,2].reshape(-1,1),y,GPy.kern.RBF(1))
    # model3.optimize()

    model1.plot()
    plt.xlabel("Z1")
    plt.ylabel("Reward")
    plt.title("Agent 1")
    model2.plot()
    plt.xlabel("Z2")
    plt.ylabel("Reward")
    plt.title("Agent 2")
    model3.plot()
    plt.xlabel("Z3")
    plt.ylabel("Reward")
    plt.title("Agent 3")
    plt.show()




    