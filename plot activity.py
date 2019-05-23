#%% functions 

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy import outer
import pandas as pd


def compute_z(w, r):
    """Returns the output of the network. 
    
    ---
    Arguments:
    - w is the weight vector
    - r is the neurons activity vector
    """
    
    return dot(w, r)


def error(w, r, f):
    """Returns the error of the system, defined as the distance between the
    output and the desired function f(t)
    
    ---
    Arguments:
    - w weight vector
    - r neurons activity vector
    - f the desired output value of f(t)
    """
    
    return compute_z(w, r) - f



def verif(x):
    print(np.min(x), np.max(x))
    
    
#%% setting initial conditions
np.random.seed(22)
n_neuron = 1000
TOTAL_TIME = 600
alpha = 1
g = 1.5         #controls the chaos of the system
p = np.identity(n_neuron) / alpha
tau = 0.01
dt = 0.001
pz = 0.1

#the two ws
J = np.random.normal(0, 1/np.sqrt(0.1*n_neuron),size=(n_neuron, n_neuron))
#w_out = np.zeros(n_neuron)
w_out= np.random.normal(0, 1/np.sqrt(1*n_neuron),size=n_neuron)


x = np.zeros((n_neuron, TOTAL_TIME))
x[:,0] = np.random.normal(size=n_neuron)

time = np.arange(TOTAL_TIME)

z = []
learning_end = 0

z.append(compute_z(w_out, np.tanh(x[:,0])))

#running

for i in time[1:]:
    
    x_old = x[:, i-1]
    r_old = np.tanh(x_old)
    x[:,i] = x_old  + dt*(1/tau)*(-x_old + g*dot(J, r_old))
    
    r = np.tanh(x[:,i])
    z.append(compute_z(w_out, r))
    
    

for neuron in range(10):
    plt.plot(time, x[neuron,:])
    plt.show()
 
plt.plot(time, z, color="red")
plt.ylabel("output of the network")
plt.xlabel("time in ms")
plt.show()

