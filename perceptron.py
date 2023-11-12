import numpy as np

# AND
# NAND
# OR
# XOR

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = -0.7
    tmp = np.sum(w*x) + theta
    if tmp <=0:
        return 0
    if tmp > 0:
        return 1


def NAND(x1, x2):
    # x = np.array([x1, x2])
    # w = np.array([-0.5, -0.5])
    # theta = 0.7
    # tmp = np.sum(w*x) + theta
    # if tmp <=0:
    #     return 0
    # if tmp > 0:
    #     return 1

    return not AND(x1, x2)

def OR (x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = -0.2
    tmp = np.sum(w*x) + theta
    if tmp <=0:
        return 0
    if tmp > 0:
        return 1
    
def XOR(x1, x2):
    s1= AND(x1, x2)
    s2= OR(x1,x2)
    y = AND(s1, s2)
    return y
    