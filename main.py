import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from lms.gd import GradientDescent

def plotChart(iterations, cost, accuracy):
    fig, ax = plt.subplots(1,2)
    ax[0].plot(np.arange(iterations), cost, 'r')
    ax[1].plot(np.arange(iterations), accuracy, 'b')
    ax[0].set_xlabel('Iterations')
    ax[1].set_xlabel('Iterations')
    ax[0].set_ylabel('Cost')
    ax[1].set_ylabel('Accuracy')
    ax[0].set_title('Cost vs Iterations')
    ax[1].set_title('Accuracy vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()
    
def plotInputs(x, y, z):
    # Plot X,Y,Z
    # fig, ax = plt.subplots(111, projection='3d')
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_trisurf(z, y, z, 
                    color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(x, y, z, c='red')
    plt.show()

if __name__ == "__main__":
    # Import data
    data = pd.read_csv('./dataset/house_practice.csv')
    # Extract data into X and y
    inputs = data[['Size', 'Bedrooms']].to_numpy()
    actuals = data[['Price']].to_numpy()
    
    num_of_iterations = 1000
    gd = GradientDescent(learning_rate=0.003, num_of_iterations=num_of_iterations,
                         inputs=inputs, actuals=actuals)
    # gd.predict()
    # Run Gradient Descent
    cost, theta, accuracy = gd.gradient_descent()

    # Display cost chart
    plotChart(num_of_iterations, cost, accuracy)
    
    # plotInputs(inputs[:, 0], inputs[:, 1], gd.predict())
    
    
    # # plot to verify cost function decreases
    # h = plt.figure('Verification')
    # plt.plot(color='b')
    # h.show()