


# y0 = np.array([[reward]])

# # Initialize GP models and SafeOpt instances for each agent
# gps = []
# safeopts = []
# for agent in agents:
#     x0 = np.zeros((1, 2))  # Initial safe point for each agent
#     y0 = np.zeros((1, 1))  # Initial dummy reward
#     gp = GPy.models.GPRegression(x0, y0, kernel, noise_var=noise_var)
#     parameter_set = linearly_spaced_combinations(bounds, 100)
#     opt = SafeOpt(gp, parameter_set, -5, threshold=0.2, beta=3.5)
#     gps.append(gp)
#     safeopts.append(opt)
    




# print("Initial reward is ",reward)

# #obtain global reward
# y_new = np.array([[reward]])  

# actions = []


# for r in range(10):
#     for opt in safeopts:
#         x_next = opt.optimize()
#         print("Next action is ",x_next)
#         y_new = np.array([[reward]])  # Assuming the global reward is shared
#         opt.add_new_data_point(x_next, y_new)
#         actions.append(x_next)

#     #simulate the actions in the environment and get the global reward
#     observation, reward, terminated, info = system.simulate(actions)
#     print("Reward is ",reward)


# for step in range(20):
#     actions = []
#     for opt in safeopts:
#         # Obtain next action from SafeOpt for each agent
#         x_next = opt.optimize()
#         actions.append(x_next)

#     # Simulate the actions in the environment and get the global reward
#     observation, reward, terminated, info = system.simulate(actions)
#     print("Reward is ",reward)
    
#     # Update each agent's GP model with the action taken and the observed reward
#     for i, (gp, opt) in enumerate(zip(gps, safeopts)):
#         x_new = actions[i]
#         y_new = np.array([[reward]])  # Assuming the global reward is shared
#         gp.set_XY(np.vstack([gp.X, x_new]), np.vstack([gp.Y, y_new]))
#         opt.add_new_data_point(x_new, y_new)

#     if terminated:
#         print('terminated')
#         break


# Number of steps to simulate

# for i in range(2):
#     # Simulate for 10 steps
#     while True:
#         # Define the actions for each agent
#         actions = [np.random.uniform(-1, 1, 2) for _ in range(len(system.agents))]
#         # Simulate the environment
#         observation, reward, terminated, info = system.simulate(actions)

#         if terminated:
#             print('terminated')
#             break
#         #print the reward
#         print(reward)
