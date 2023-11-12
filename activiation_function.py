import matplotlib.pylab as plt
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

# x= np.array([-5, 5, 0.1])

x= np.arange(-5, 5, 0.1)

y=sigmoid(x)

# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

def relu(x):
    return np.maximum(0,x)