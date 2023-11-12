import numpy as np
import activiation_function as af


def idientify_functioon(x):
    return x

def init_network():
    network={}
    network['W1']= np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['B1']= np.array([0.1, 0.2, 0.3])
    network['W2']= np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['B2']= np.array([0.1, 0.2])
    network['W3']= np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B3']= np.array([[0.1, 0.2]])

    return network

def forward(network, x):
    w1, w2, w3= network['W1'], network['W2'], network['W3']
    B1, B2, B3= network['B1'], network['B2'], network['B3']

    A1= np.dot(x, w1)+ B1
    Z1= af.sigmoid(A1)
    A2= np.dot(Z1, w2)+ B2
    Z2= af.sigmoid(A2)
    A3= np.dot(Z2, w3)+ B3

    Y=idientify_functioon(A3)
    return Y

    