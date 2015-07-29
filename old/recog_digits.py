#-*- encoding: utf-8 -*-
import numpy as np
from sklearn import datasets

class MLP(object):
    def __init__(self, n_input_units, n_hidden_units, n_output_units):
        self.nin = n_input_units
        self.nhid = n_hidden_units
        self.nout = n_output_units

        self.v = np.random.uniform(-1.0, 1.0, (self.nhid, self.nin+1))
        self.w = np.random.uniform(-1.0, 1.0, (self.nout, self.nhid+1))

    def fit(self, inputs, targets, learning_rate=1, epochs=10000):
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
    mlp = MLP(n_input_units=64, n_hidden_units=12, n_output_units=10)

    digits = datasets.load_digits()

    inputs = digits.data
    inputs /= inputs.max()
    answers = digits.target
    targets = []
    for answer in answers:
        target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        target[answer] = 1.0
        targets.append(target)

    # training
    mlp.fit(inputs, targets)

    # predict
    j = [8, 10, 20, 34]
    print '--- predict ---'
    for i in j:
        print "=========="
        print "targets"
        print targets[i]
        print "predict"
        print mlp.predict(inputs[i])


    tegaki = [
        0,0,0,1.0,1.0,0,0,0,
        0,0,0,1.0,1.0,0,0,0,
        0,0,0,1.0,1.0,0,0,0,
        0,0,0,1.0,1.0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,1.0,1.0,0,0,0,
        0,0,0,1.0,1.0,0,0,0,
        ]
    print mlp.predict(tegaki)
