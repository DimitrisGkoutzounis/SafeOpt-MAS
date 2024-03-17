from scipy.optimize import differential_evolution
import numpy as np

def F_vec(x):
    
    return -(np.sin(x[0]**3) + np.cos(x[1]**2) - np.sin(x[2]))  # Negate F for maximization


bounds = [(-2, 2), (-2, 2), (-2, 2)]

result = differential_evolution(F_vec, bounds)

max_value_computed = -result.fun  
max_value_computed

print(result.x, max_value_computed)