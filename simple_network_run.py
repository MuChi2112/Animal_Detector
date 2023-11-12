import simple_network_data as snd
import numpy as np

network = snd.init_network()
x= np.array([1.0, 0.5])
y= snd.forward(network, x)

print(y)