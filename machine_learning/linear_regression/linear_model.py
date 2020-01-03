import numpy as np

class LinearModel:
    def __init__(self): 
        self.b = 0
        self.m = 0

    def fit(self, data_x, data_y, eval_func):
        eval_func = globals()[eval_func]
        eval_func(self, data_x, data_y)

    def predict(self, data_x):
        pred_y = []
        
        for i in range(len(data_x)):
            pred_y.append(self.b + self.m * data_x[i])
        
        return pred_y

    def coefficients(self):
        return [self.b, self.m]

    def reset(self):
        self.b = 0
        self.m = 0
   
def least_squares(model, data_x, data_y):
    mean_x = np.mean(data_x)
    mean_y = np.mean(data_y)
        
    sum_n = 0
    sum_d = 0

    for x, y in zip(data_x, data_y):
        sum_n += (x - mean_x)*(y - mean_y)
        sum_d += (x - mean_x)**2
    
    model.m = sum_n/sum_d
    model.b = mean_y - model.m*mean_x

def gradient_descent(model, data_x, data_y):
    iterations = 1000
    learning_rate = 0.001

    for _ in range(iterations):
        pred_y = data_x * model.m + model.b
        dm = -2 * sum(data_x * (data_y - pred_y))
        db = -2 * sum(data_y - pred_y)

        model.m -= dm * learning_rate
        model.b -= db * learning_rate
