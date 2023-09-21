import numpy as np

class GradientDescent:
    def __init__(self, learning_rate,inputs, actuals, 
                 num_of_iterations=100):
        # Normalize our features
        inputs = (inputs - inputs.mean()) / inputs.std()
        # Add a 1 column to compensate x_0 = 1 to the actual input  
        self.inputs = np.c_[np.ones(inputs.shape[0]), inputs]
        self.iterations = num_of_iterations
        self.actuals = actuals.T
        self.learning_rate = learning_rate
        self.bias = np.array([np.zeros(self.inputs.shape[1])])
   
    def compute_cost(self):
        x = self.inputs
        y = self.actuals
        theta = self.bias
        m = y.size
        error = np.dot(theta, x.T) - y
        cost = (1 /  2) * m * np.dot(error, error.T)
        return cost, error  
    
    def gradient_descent(self):
        cost_array = np.array([])
        accuracy = np.array([])
        alpha = self.learning_rate
        x = self.inputs
        m = self.actuals.size
        for i in range(self.iterations):
            theta = self.bias
            cost, error = self.compute_cost()
            theta = theta - (alpha * (1 / m) * np.dot(error, x))
            cost_array = np.append(cost_array, cost)
            current_accuracy = self.get_current_accuracy()
            accuracy = np.append(accuracy, current_accuracy)
            self.bias = theta
        return cost_array, theta, accuracy
    
    def get_current_accuracy(self):
        predicted = np.dot(self.inputs, self.bias.T)
        p, a = predicted.T, self.actuals
        n = self.actuals.size
        # return 1 - (np.sum(np.abs(p - a) / a) / n)
        errors = []
        for i in range(n):
            if a[0][i] != 0:
                error = np.abs(p[0][i] - a[0][i]) / a[0][i]
                errors.append(error)
        return 1 - np.sum(errors) / n
        # return 1 - np.sum(
        #     [
        #         np.abs(p[i] - a[i]) / a[i]
        #         for i in range(n)
        #         if a[i] != 0 ]
        # ) / n
