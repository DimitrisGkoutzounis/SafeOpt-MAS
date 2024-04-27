import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import safeopt
import GPy
import numpy as np

# Define a function to find continuous positive sections
def find_continuous_positive_sections(x, y,constraint):
    sections = []
    start_idx = None
    for i in range(len(y)):
        if y[i] > constraint and start_idx is None:
            start_idx = i
        elif y[i] <= constraint and start_idx is not None:
            sections.append((x[start_idx:i], y[start_idx:i]))
            start_idx = None
    if start_idx is not None:
        sections.append((x[start_idx:], y[start_idx:]))
    return sections

def find_correlation(sections_y,sections_y2):
    correlations = []
    for x_sec_y, y_sec_y in sections_y:
        x_min, x_max = x_sec_y[0], x_sec_y[-1]
        for x_sec_y2, y_sec_y2 in sections_y2:
            if x_sec_y2[0] <= x_max and x_sec_y2[-1] >= x_min:
                common_x_min = max(x_min, x_sec_y2[0])
                common_x_max = min(x_max, x_sec_y2[-1])
                common_x = np.linspace(common_x_min, common_x_max, num=100)  # 50 points for the correlation
                interp_y = interp1d(x_sec_y, y_sec_y, kind='linear', fill_value="extrapolate")(common_x)
                interp_y2 = interp1d(x_sec_y2, y_sec_y2, kind='linear', fill_value="extrapolate")(common_x)
                corr, _ = pearsonr(interp_y, interp_y2)
                correlations.append(corr)
    return correlations

def generate_correlated_plot(x, y, x2, y2,func_form, sections_y, sections_y2):
    """
    x: x values for the original function
    y: y values for the original function
    x2: x values for the function to be correlated
    y2: y values for the function to be correlated
    func_form: the form of the function to be correlated.Used for the plot legend
    sections_y: the continuous positive sections of the original function
    sections_y2: the continuous positive sections of the function to be correlated
    """
    # Generate the plot for the correlated sections
    plt.figure(figsize=(14, 7))

    # Plot the original functions
    plt.plot(x, y, label='y = sin(x^3) + cos(x^2) - sin(x)')
    plt.plot(x2, y2, label=func_form, linestyle='--')

    corr = find_correlation(sections_y,sections_y2)

    # We interpolate and correlate only if we have matching sections
    for x_sec_y, y_sec_y in sections_y:
        # Find the x range for the current y section
        x_min, x_max = x_sec_y[0], x_sec_y[-1]

        # For each y2 section, check if the x range overlaps with the current y section
        for x_sec_y2, y_sec_y2 in sections_y2:
            if x_sec_y2[0] <= x_max and x_sec_y2[-1] >= x_min:
                # The sections overlap, find the common range
                common_x_min = max(x_min, x_sec_y2[0])
                common_x_max = min(x_max, x_sec_y2[-1])

                # Interpolate both sections to a common x range for comparison
                common_x = np.linspace(common_x_min, common_x_max, num=100)  # 50 points for the correlation
                interp_y = interp1d(x_sec_y, y_sec_y, kind='linear', fill_value="extrapolate")(common_x)
                interp_y2 = interp1d(x_sec_y2, y_sec_y2, kind='linear', fill_value="extrapolate")(common_x)

                # Plot the interpolated, correlated sections
                plt.plot(common_x, interp_y, label='Interpolated y (Correlated Section)', linewidth=2, marker='o')
                plt.plot(common_x, interp_y2, label='Interpolated {func_form} (Correlated Section)'.format(func_form=func_form), linewidth=2, marker='x')

    # Decorate the plot
    plt.xlabel('x')
    plt.ylabel('y and {func_form}'.format(func_form=func_form))
    plt.title('Plot of the Functions and their Correlated Positive Sections')
    if len(corr) > 0:
        plt.annotate('Correlation: {corr}'.format(corr = corr[0]), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    plt.legend()
    plt.grid(True)
    plt.show()

def global_reward(x1,x2,x3):
        "Each agent contributes to a different part of the global function"
    
        result =  np.sin(x1**3) + np.cos(x2**2) - np.sin(x3)

        return result

class Agent:
    def __init__(self,id,bounds,safe_point,criterion=True):
        self.bounds = [bounds]
        self.id = id
        self.safepoint = safe_point
        self.optimization_type = criterion
        self.global_rewards = np.array([])
        self.local_rewards = np.array([])
        self.max_belief_g = np.array([[]])
        self.max_belief_l = np.array([[]])

        self.global_x0 = np.asarray([[self.safepoint]])
        self.global_y0 = np.asarray([[1]]) #predermined initial reward for all agents close to the actual

        self.kernel = GPy.kern.RBF(input_dim=1)
        self.gp = GPy.models.GPRegression(self.global_x0,self.global_y0, self.kernel, noise_var=0.05**2)
        self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)

        self.global_opt = safeopt.SafeOpt(self.gp, self.parameter_set, -np.inf,beta=4,threshold=0.2)

        self.local_x0 = np.asarray([[self.safepoint]])
        self.local_y0 = self.local_reward(self.safepoint)
        self.local_y0 = np.asarray([[self.local_y0]])
        print(f"Local Reward for agent {self.id} is {self.local_y0}")

        print(f"Agent {self.id} is using {self.optimization_type} as optimization criterion.")
        
        if self.optimization_type == False:
            self.local_gp = GPy.models.GPRegression(self.local_x0,self.local_y0*-1, self.kernel, noise_var=0.01**2)
            self.local_opt = safeopt.SafeOpt(self.gp, self.parameter_set, -np.inf,beta=3,threshold=0.2)
        else:
            self.local_gp = GPy.models.GPRegression(self.local_x0,self.local_y0, self.kernel, noise_var=0.01**2)
            self.local_opt = safeopt.SafeOpt(self.local_gp, self.parameter_set,-np.inf,beta=3,threshold=0.2)

    def local_reward(self,x):
        
        if self.id == 1:
            return np.sin(x**3)
        elif self.id == 2:
            return np.cos(x**2)
        elif self.id == 3:
            return np.sin(x)
        
    def predict_local(self):

        x_next_l = self.local_opt.optimize()

        return x_next_l
    
    def predict_global(self):
            
        x_next_g = self.global_opt.optimize()
    
        return x_next_g
    
    def update_local(self,x_next,y_meas):
        self.max_belief_l = np.append(self.max_belief_l,self.local_opt.get_maximum()[1])
        
        self.local_rewards = np.append(self.local_rewards,y_meas)
        self.local_opt.add_new_data_point(x_next, y_meas)

    def update_global(self,x_next,y_meas):
        self.max_belief_g = np.append(self.max_belief_g,self.global_opt.get_maximum()[1])
        
        self.global_rewards = np.append(self.global_rewards,y_meas)
        self.global_opt.add_new_data_point(x_next, y_meas)


    def plot_gp_local(self):
        """
        Plot the local reward belief
        """

        if self.id == 1:
            self.local_opt.plot(1000)
            plt.plot(self.parameter_set, self.local_reward(self.parameter_set),color="C2", alpha=0.3,label='sin(x1^3)')
            plt.legend()
            # plt.annotate(f"Correlation: {corr}", xy=(0.5, 0.5), xycoords='axes fraction')

        if self.id == 2:
            self.local_opt.plot(1000)
            plt.plot(self.parameter_set, self.local_reward(self.parameter_set),color="C2", alpha=0.3, label='sin(x2^2)')
            plt.legend()
            # plt.annotate(f"Correlation: {corr}", xy=(0.5, 0.5), xycoords='axes fraction')
        if self.id == 3:
            self.local_opt.plot(1000)
            plt.plot(self.parameter_set, self.local_reward(self.parameter_set),color="C2", alpha=0.3, label='sin(x3)')
            plt.legend()
            # plt.annotate(f"Correlation: {corr}", xy=(0.5, 0.5), xycoords='axes fraction')

        plt.title(f"Agent {self.id} Local Reward Estimation")
        plt.show()
        
    def plot_gp_global(self):
        """
        Plot the global reward belief
        """
        if self.id == 1:
            self.global_opt.plot(1000)
            plt.plot(self.parameter_set, global_reward(self.parameter_set,self.parameter_set,self.parameter_set),color="r", alpha=0.8, label='sin(x1^3) + sin(x2^2) +sin(x3)')
            plt.legend()
            # plt.annotate(f"Correlation: {corr}", xy=(0.5, 0.5), xycoords='axes fraction')
        if self.id == 2:
            self.global_opt.plot(1000)
            plt.plot(self.parameter_set, global_reward(self.parameter_set,self.parameter_set,self.parameter_set),color="r", alpha=0.8, label='sin(x1^3) + sin(x2^2) +sin(x3)')
            plt.legend()
            # plt.annotate(f"Correlation: {corr}", xy=(0.5, 0.5), xycoords='axes fraction')
        if self.id == 3:
            self.global_opt.plot(1000)
            plt.plot(self.parameter_set, global_reward(self.parameter_set,self.parameter_set,self.parameter_set),color="r", alpha=0.8, label='sin(x1^3) + sin(x2^2) +sin(x3)')
            plt.legend()
            # plt.annotate(f"Correlation: {corr}", xy=(0.5, 0.5), xycoords='axes fraction')
            
        plt.title(f"Agent {self.id} Global Reward Estimation")
        plt.show()


# class Agent_gp:
#     def __init__(self,id,bounds,safe_point):
        