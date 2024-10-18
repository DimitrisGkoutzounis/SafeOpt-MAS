import numpy as np
from scipy.optimize import minimize

# Define the individual functions f(x1), f(x2), and f(x3)
def f_x1(x1):
    return -x1**2 + 4*x1  # Example: a parabola shifted to have a maximum at x1=2

def f_x2(x2):
    return x2**2 - 4*x2  # Example: another parabola

def f_x3(x3):
    return x3**2 + 2*x3  # Example: yet another parabola

# Single variable function that combines the effects of f_x1, f_x2, and f_x3
def global_function_single(x):
    return f_x1(x) + f_x2(x) + f_x3(x)

# Multi-variable function
def global_function_multi(x):
    x1, x2, x3 = x
    return f_x1(x1) + f_x2(x2) + f_x3(x3)

# Objective functions for minimization (to maximize the actual functions)
def objective_single(x):
    return -global_function_single(x[0])

def objective_multi(x):
    return -global_function_multi(x)

# Define bounds
single_bounds = [(-10, 10)]
multi_bounds = [(-10, 10), (-10, 10), (-10, 10)]

# Optimization for single variable
single_result = minimize(objective_single, [0], bounds=single_bounds)

# Optimization for multi variables
multi_result = minimize(objective_multi, [0, 0, 0], bounds=multi_bounds)

# Results for single variable
if single_result.success:
    optimized_value_single = single_result.x[0]
    max_value_single = -single_result.fun
    print(f"Global maximum for single input function: {max_value_single} at x={optimized_value_single}")
else:
    print("Single variable optimization failed:", single_result.message)

# Results for multiple variables
if multi_result.success:
    optimized_values_multi = multi_result.x
    max_value_multi = -multi_result.fun
    print(f"Global maximum for multi input function: {max_value_multi} at x1={optimized_values_multi[0]}, x2={optimized_values_multi[1]}, x3={optimized_values_multi[2]}")
else:
    print("Multi-variable optimization failed:", multi_result.message)
