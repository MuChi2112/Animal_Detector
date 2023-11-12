import numpy as np
import activiation_function as af

X= np.array([1.0, 0.5])
W1= np.array([[0.1, 0.3, 0.5], [0.2,0.4, 0.6]])
B1= np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1= np.dot(X, W1)+ B1

Z1= af.sigmoid(A1)

print(Z1)
W2= np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2= np.dot(Z1, W2)+B2
Z2=af.sigmoid(A2)

print(Z2)

def idientify_functioon(x):
    return x