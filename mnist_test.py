#coding: utf-8
import numpy as np
import pylab
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home=".")

p = np.random.random_integers(0, len(mnist.data), 25)
print p 
for index, (data, label) in enumerate(np.array(zip(mnist.data, mnist.target))[p]):
    print label
    pylab.subplot(5, 5, index + 1)
    pylab.axis('off')
    pylab.imshow(data.reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
    pylab.title('%i' % label)
pylab.show()
