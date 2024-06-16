import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Define the global function with single and multiple inputs
def global_function_single(x):
    return -x**2 + 4*x + x**2 - 4*x + x**2 + 2*x

def global_function_multi(x1, x2, x3):

    return -x1**2 + 4*x1 + x2**2 - 4*x2 + x3**2 + 2*x3
# Create a meshgrid for plotting
x = np.linspace(-10, 10, 100)
x1, x2, x3 = np.meshgrid(x, x, x)

# Compute the function values
y_single = global_function_single(x)
y_multi = global_function_multi(x1, x2, x3)

# Plotting the single input function
fig = plt.figure(figsize=(12, 6))

# Single input plot

ax1 = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(x, x)
Z = global_function_single(X)
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Single Input Function')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Multi input plot
ax2 = fig.add_subplot(122, projection='3d')
Z_multi = global_function_multi(x1, x2, x3)
# Since Z_multi is 3D, take the max across one axis to get a 2D representation
Z_max = np.max(Z_multi, axis=2)
ax2.plot_surface(X, Y, Z_max, cmap='viridis')
ax2.set_title('Multi Input Function')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('Global Max')

plt.tight_layout()
plt.show()

# Find the global maximum for both cases
global_max_single = np.max(y_single)
global_max_multi = np.max(y_multi)
print(f"Global maximum for single input function: {global_max_single}")
print(f"Global maximum for multi input function: {global_max_multi}")





