import numpy as np

def relu(x, derivative=False):
    if derivative:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    return np.maximum(0, x)

def sigmoid(x, derivative=False):
    if derivative:
        sig = sigmoid(x)
        return  sig * (1 - sig)

    return 1 / (1+np.exp(-x))

def sum_of_squares(x, y, derivative=False):
    if derivative:
        return 2 * np.sum(x - y)
    
    return np.sum((x - y)**2)

def bin_cross_entropy(x, y, derivative=False):
    if derivative:
        return - (np.divide(y, x) - np.divide(1 - y, 1 - x))

    m = x.shape[1]
    cost = -1 / m * (np.dot(y, np.log(x).T) + np.dot(1 - y, np.log(1 - x).T))
    return np.squeeze(cost)

class NeuralNetwork:
    def __init__(self, dimensions, activation_function=relu, readout_function=sigmoid, cost_function=sum_of_squares, seed=99):
        self.dimensions = dimensions
        self.activation_function = activation_function
        self.readout_function = readout_function
        self.cost_function = cost_function

        np.random.seed(seed)
        self.weights = [np.random.randn(y, x) * 0.1 for x, y in zip(dimensions[:-1], dimensions[1:])]
        self.baises = [np.random.randn(y, 1) * 0.1 for y in dimensions[1:]]

    def forward_prop(self, x):
        a_mem = [x]
        z_mem = []

        for l in range(len(self.weights)-1):
            w = self.weights[l]
            b = self.baises[l]
            x = np.dot(w, x) + b
            z_mem.append(x)
            x = self.activation_function(x)
            a_mem.append(x)
        
        w = self.weights[-1]
        b = self.baises[-1]
        x = np.dot(w, x) + b
        z_mem.append(x)
        x = self.readout_function(x)
        a_mem.append(x)

        return a_mem, z_mem


    def backward_prop(self, y, a_mem, z_mem):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.baises]
      
        da_curr = self.cost_function(a_mem[-1], y, True)
        a_prev = a_mem[-2]
        z_curr = z_mem[-1]
        w_curr = self.weights[-1]

        m = a_prev.shape[1]
        dz_curr = self.readout_function(z_curr, True) * da_curr

        nabla_w[-1] = np.dot(dz_curr, a_prev.T) / m
        nabla_b[-1] = np.sum(dz_curr, axis=1, keepdims=True) / m
        da_curr = np.dot(w_curr.T, dz_curr)

        for l in range(2, len(self.dimensions)):
            a_prev = a_mem[-l-1]
            z_curr = z_mem[-l]
            w_curr = self.weights[-l]

            m = a_prev.shape[1]
            dz_curr = self.activation_function(z_curr, True) * da_curr

            nabla_w[-l]  = np.dot(dz_curr, a_prev.T) / m
            nabla_b[-l] = np.sum(dz_curr, axis=1, keepdims=True) / m
            da_curr = np.dot(w_curr.T, dz_curr)
        
        return nabla_w, nabla_b

    def update(self, nabla_w, nabla_b, lr):
        for l in range(len(self.weights)):
            self.weights[l] -= lr * nabla_w[l]
            self.baises[l] -= lr * nabla_b[l]


    def train(self, x, y, epochs, learning_rate=0.1, verbose=False, callback=None):
        for e in range(epochs):
            a_mem, z_mem = self.forward_prop(x)
            cost = self.cost_function(a_mem[-1], y)
            nabla_w, nabla_b = self.backward_prop(y, a_mem, z_mem)
            self.update(nabla_w, nabla_b, learning_rate)

            if e%50 == 0:
                if verbose:
                    print("Iteration: {:05} - cost: {:.5f}".format(e, cost))
                if callback:
                    callback(e)
