 # mean,variance = gplvm.predict(X)



    # mean0 = gplvm.predict(X[:,2].reshape(-1,1))[0]
    # model = GPy.models.GPRegression(mean0,y, GPy.kern.RBF(1))
    # model.plot()
    # plt.xlabel('Iput Space $X1$')
    # plt.ylabel('Output y')
    # plt.show()