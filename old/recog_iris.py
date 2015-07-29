#-*- encoding: utf-8 -*-
import numpy as np
import iris_data
import target_data

class MLP(object):
    def __init__(self, n_input_units, n_hidden_units, n_output_units):
        self.nin = n_input_units
        self.nhid = n_hidden_units
        self.nout = n_output_units

        self.v = np.random.uniform(-1.0, 1.0, (self.nhid, self.nin+1))
        self.w = np.random.uniform(-1.0, 1.0, (self.nout, self.nhid+1))

    def fit(self, inputs, targets, learning_rate=0.5, epochs=100000):
        inputs = np.array(inputs)
        inputs = self.__add_bias(inputs, axis=1)
        targets = np.array(targets)

        for loop_cnt in xrange(epochs):
            p = np.random.randint(inputs.shape[0])
            xp = inputs[p]
            bkp = targets[p]

            gjp = self.__sigmoid(np.dot(self.v, xp))
            gjp = self.__add_bias(gjp)
            gkp = self.__sigmoid(np.dot(self.w, gjp))

            eps2 = self.__sigmoid_deriv(gkp) * (gkp - bkp)
            eps = self.__sigmoid_deriv(gjp) * np.dot(self.w.T, eps2)
            # output layer training
            gjp = np.atleast_2d(gjp)
            eps2 = np.atleast_2d(eps2)
            self.w = self.w - learning_rate * np.dot(eps2.T, gjp)
            # hidden layer training
            xp = np.atleast_2d(xp)
            eps = np.atleast_2d(eps)
            self.v = self.v - learning_rate * np.dot(eps.T, xp)[1:, :]

    def __add_bias(self, x, axis=None):
        return np.insert(x, 0, 1, axis=axis)

    def __sigmoid(self, u):
        return (1.0 / (1.0 + np.exp(-u)))

    def __sigmoid_deriv(self, u):
        return (u * (1 - u))

    def predict(self, inputs):
        inputs = self.__add_bias(inputs)
        xp = np.array(inputs)
        gjp = self.__sigmoid(np.dot(self.v, xp))
        gjp = self.__add_bias(gjp)
        gkp = self.__sigmoid(np.dot(self.w, gjp))
        return gkp
        
if __name__ == '__main__':
    mlp = MLP(n_input_units=4, n_hidden_units=4, n_output_units=3)

    inputs = iris_data.inputs
    targets = target_data.targets

    # training
    mlp.fit(inputs, targets)

    # predict
    print '--- predict ---'
    for i in [[4.9  ,2.9, 1.5, 0.4], [6.1, 3.4, 4.3, 1.6], [6.0, 2.4, 5.5, 1.6]]:
        print i, mlp.predict(i)
