#-*- encoding: utf-8 -*-
import numpy as np
import shelve
import datetime
from sklearn import cross_validation
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
import pylab as pl

class MLP(object):
    def __init__(self, n_input_units, n_hidden_units, n_output_units):
        self.nin = n_input_units
        self.nhid = n_hidden_units
        self.nout = n_output_units

        self.v = np.random.uniform(-1.0, 1.0, (self.nhid, self.nin+1))
        self.w = np.random.uniform(-1.0, 1.0, (self.nout, self.nhid+1))

    def train(self, inputs, targets, learning_rate=0.5, epochs=100000):
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
            eps  = self.__sigmoid_deriv(gjp) * np.dot(self.w.T, eps2)

            # output layer training
            gjp  = np.atleast_2d(gjp)
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

    def distinct(self, inputs):
        inputs = self.__add_bias(inputs)
        xp = np.array(inputs)
        gjp = self.__sigmoid(np.dot(self.v, xp))
        gjp = self.__add_bias(gjp)
        gkp = self.__sigmoid(np.dot(self.w, gjp))
        return gkp

    def save(self):
        # shelve the mlp object
        d = datetime.datetime.today()
        datestr = d.strftime("%Y-%m-%d-%H:%M:%S")
        neurons = [self.v, self.w]
        dic = shelve.open('./save/mlp'+datestr)
        dic['data'] = neurons
        dic.close()

    def load(self, filename):
        dic = shelve.open(filename)
        self.v = dic['data'][0]
        self.w = dic['data'][1]

    def visualize_data(self, data, width):
        new_data = data.reshape((width, -1))
        pl.gray()
        pl.matshow(new_data)
        pl.show()

    def show_weight(self, layer, num, width):
        if layer == 0 :
            data = self.v[num]
        elif layer == 1 :
            data = self.w[num]

        data = data[1::1]
        self.visualize_data(data, width)

        #new_data = data.reshape((width, -1))

        #pl.gray()
        #pl.matshow(new_data)
        #pl.show()

    def binalize(self, data):
        for i in range(0, data.shape[0]):
            if data[i] > 0.5:
                data[i] = 1
            else:
                data[i] = 0

        return data

    def evaluate(self, X_test, y_test):
        ok = 0
        ng = 0
        for i in range(0, y_test.shape[0]):
            y = self.binalize(mlp.distinct(X_test[i]))
            if np.array_equal(y, y_test[i]):
                ok = ok + 1
            else:
                ng = ng + 1
                # self.visualize_data(X_test[i], 8)

        ok_rate = float(ok) / float(ok + ng)
        return ok_rate

if __name__ == '__main__':
    mlp = MLP(n_input_units=784, n_hidden_units=300, n_output_units=10)

    # create input and target
    #digits = datasets.load_digits()
    mnist = datasets.fetch_mldata('MNIST original', data_home=".")
    inputs = mnist.data
    inputs /= inputs.max()

    answers = mnist.target

    import time
    from pprint import pprint
    # data broken?
    for i in range(0, inputs.shape[0]):
        print i
        print answers[i]
        pprint(inputs[i])
        #mlp.visualize_data(inputs[i], 28)
        time.sleep(0.3)
    
    exit()

    # substituted by LabelBinalizer
    #targets = []
    #for answer in answers:
    #    target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #    target[int(answer)] = 1.0
    #    targets.append(target)
  
    #targets = np.array(targets) #need to change list into nparray for cross_validation

    # split data into train and test
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        inputs, answers, test_size=0.2, random_state=0
    )
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    # training
    mlp.train(X_train, labels_train)

    # visualize weight
    #for i in range(0, 11):
    #    mlp.show_weight(0, i, 8)

    # save mlp weights
    mlp.save()

    # readout mlp weights
    #mlp.load('save/mlp2015-04-25-23:40:44')

    # evaluate mlp
    print mlp.evaluate(X_test, labels_test)

    exit()

    # distinct each case
    for i in range(0, y_test.shape[0]):
        print y_test[i]
        print mlp.distinct(X_test[i])

