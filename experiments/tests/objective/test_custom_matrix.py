import numpy as np
import matplotlib.pyplot as plt

# Define the global reward function
def f(x1, x2, x3):
    y = np.sin(x1**3) + np.cos(x2**2) - np.sin(x3)
    return y

if __name__ == '__main__':
    N = 3
    D = 3

    X = np.array([
    [-1.42575308,  1.24173385, -0.35161908],
    [ 0.72776006,  1.05583663,  0.54326826],
    [-1.04217986,  1.17542612,  1.41493128],
    [ 1.36172291, -0.26067154, -0.88348173],
    [-0.70160691, -0.39495944,  0.44333208],
    [ 1.42846997, -0.10628884,  0.99366426],
    [-0.76157941,  0.40061836, -0.04915141],
    [-0.01544846, -1.43256817,  1.27191351],
    [-0.57175075, -0.40403998, -0.48343176],
    [-0.59935217,  1.31683308, -1.23235295]])

    # Extract columns as x1, x2, x3
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    Z_opt = np.array([])

    # Compute rewards for Z_initial
    R_Z_initial = f(x1, x2, x3)

    # Compute rewards for Z_opt
    R_Z_opt = f(Z_opt[:, 0], Z_opt[:, 1], Z_opt[:, 2])

    # Plotting the results
    plt.figure(figsize=(15, 6))
    plt.plot(R_Z_initial, label='R(Z_initial)', marker='o', markersize=5)
    plt.plot(R_Z_opt, label='R(Z_opt)', marker='x', markersize=5)
    plt.title('Comparison of R(Z_initial) vs R(Z_opt)')
    plt.xlabel('Sample Index')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
