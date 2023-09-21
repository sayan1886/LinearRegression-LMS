import numpy as np

class GradientDescent:
    def __init__(self, learning_rate,inputs, actuals, 
                 num_of_iterations=100):
        # Normalize our features
        inputs = (inputs - inputs.mean()) / inputs.std()
        # Add a 1 column to compensate x_0 = 1 to the actual input  
        self.inputs = np.c_[np.ones(inputs.shape[0]), inputs]
        self.iterations = num_of_iterations
        self.actuals = actuals
        self.learning_rate = learning_rate
        self.bias = np.zeros(self.inputs.shape[1])
   
    def compute_cost(self):
        x = self.inputs
        y = self.actuals
        theta = self.bias
        m = y.size
        error = np.dot(x, theta.T) - y
        cost = (1 / ( 2 * m)) * np.dot(error.T, error)
        return cost, error  
    
    def gradient_descent(self):
        cost_array = np.array([])
        alpha = self.learning_rate
        x = self.inputs
        m = self.actuals.size
        for i in range(self.iterations):
            theta = self.bias
            cost, error = self.compute_cost()
            theta = theta - (alpha * (1 / m) * np.dot(x.T, error))
            # cost_array[i] = cost
            cost_array = np.append(cost_array, cost)
            self.bias = theta
        return cost_array, theta
    
    def get_current_accuracy(self, predicted):
        p, a = predicted, self.actuals
        n = self.actuals.size
        return 1 - sum(
            [
                np.abs(p[i] - a[i]) / a[i]
                for i in range(n)
                if a[i] != 0 ]
        ) / n

    # def mse_gradient(x,y,b):
    #     res = b[0] + b[1] * x - y		
    #     return res.mean(), (res * x).mean() 

    # def gradient_descent(gradient, x, y, start, learn_rate=0.1, n_iter=50, 
    #                     tolerance=1e-06):

    #     vector = start

    #     for _ in range(n_iter):
    #         diff = -learn_rate * np.array(gradient(x, y, vector))

    #         if np.all(np.abs(diff) <= tolerance):
    #             break
    #         vector += diff

    #     return vector

