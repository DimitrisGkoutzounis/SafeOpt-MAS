import numpy as np
import matplotlib.pyplot as plt

# Define the reward function
def reward_function(x1, x2, x3):
    return np.exp(-x1**2 - x2**2) * np.cos(x3)

x1 = np.linspace(-2, 2, 20)
x2 = np.linspace(-2, 2, 20)
x3 = np.linspace(-2, 2, 20)

x1 = np.random.uniform(-1, 1, 20)
x2 = np.random.uniform(-1, 1, 20)
x3 = np.linspace(-1,1,20)

# Calculate the reward function at each (x1, x2) point
R = reward_function(x1, x2, x3)

# Plotting as a contour plot
plt.figure(figsize=(8, 6))
plt.plot(R)
plt.title("Contour Plot of Reward Function R(x1, x2, x3) with x3 = π/4")
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
