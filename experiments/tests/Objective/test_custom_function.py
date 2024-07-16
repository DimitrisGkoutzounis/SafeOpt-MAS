import numpy as np
import matplotlib.pyplot as plt

# Define the reward function
def reward_function(x1, x2, x3):
    return x1 + x2 - x3

# Create a meshgrid for x1 and x2
x1_vals = np.linspace(-1, 1, 50)
x2_vals = np.linspace(-1, 1, 50)
x1, x2 = np.meshgrid(x1_vals, x2_vals)

fig = plt.figure(figsize=(12, 10))

x3_values = [-1, 0, 1] 

for i, x3 in enumerate(x3_values, start=1):
    ax = fig.add_subplot(1, len(x3_values), i, projection='3d')
    
    # Calculate the reward
    R = reward_function(x1, x2, x3)
    
    # Plot the surface
    surf = ax.plot_surface(x1, x2, R, cmap='viridis')
    
    # Title and labels
    ax.set_title(f'$x_3 = {x3}$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Reward')

    # Setting z limits for better comparison
    ax.set_zlim(-3, 3)

# Add a color bar which maps values to colors.
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Reward')

plt.tight_layout()
plt.show()
