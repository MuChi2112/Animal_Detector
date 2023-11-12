# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch03/neuralnet_mnist.py

import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize = False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)



