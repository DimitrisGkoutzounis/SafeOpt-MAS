import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the global function with single and multiple inputs
def global_function_single(x):
    return np.sin(x**3) + np.cos(x**2) - np.sin(x)

def global_function_multi(x1, x2, x3):
    return np.sin(x1**3) + np.cos(x2**2) - np.sin(x3)

# Create a meshgrid for plotting
x = np.linspace(-1.5, 1.5, 100)
x1, x2, x3 = np.meshgrid(x, x, x)

# Compute the function values
y_multi = global_function_multi(x1, x2, x3)

# Since Z_multi is 3D, take the max across one axis to get a 2D representation
Z_max = np.max(y_multi, axis=2)
X, Y = np.meshgrid(x, x)

# Create a new figure specifically for the multi input function plot
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, Z_max, cmap='viridis', alpha=0.8)  # Adjust transparency with alpha
ax2.set_title('Multi Input Function')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('Global Max')

# Find the coordinates of the global maximum
max_idx = np.unravel_index(np.argmax(Z_max), Z_max.shape)
max_x1 = X[max_idx]
max_x2 = Y[max_idx]
max_z = Z_max[max_idx]

# Plot a larger, more visible red dot at the global maximum
ax2.scatter([max_x1], [max_x2], [max_z], color='red', s=100, edgecolor='black', label='Global Max')  # Increased size and added edge

# Adding an annotation to label the global maximum
ax2.text(max_x1, max_x2, max_z, ' Global Max', color='black', style='italic', weight='bold', fontsize=10, verticalalignment='bottom')

# Enhance the overall visibility and aesthetics
ax2.legend()
plt.show()
