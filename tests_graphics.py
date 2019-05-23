# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:27:48 2019

@author: Antoine
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy import outer
#%% test triangle
def triangle(t):
    a = 20
    return (np.abs((t/a) - int((t/a)+0.5)))

def tri(t):
    a = 20
    b = np.ceil((t/a)+0.5)
    return (2/a) * (t - a*b) * (-1)**np.ceil((t/a)+0.5)


def sinus(t):
    A = 10
    return A*np.sin(t*(np.pi*2)/40)


def update_activity(x, y, z, sigma, rho, beta):
    """update the activity of one neuron with chaotic activity in function of
    the 3 parameters sigma, p and beta"""
    
    dt = 0.01
    
    def d_x(x, y):
        return x + dt * (sigma * y - sigma * x)
    
    def d_y(x, y, z):
        return y + dt* (rho * x - x * z - y)
    
    def d_z(x, y, z):
        return z + dt* (x * y - beta * z)
    
    return d_x(x,y), d_y(x, y, z), d_z(x, y, z)

sinus = np.vectorize(sinus)
triangle = np.vectorize(triangle)
tri = np.vectorize(tri)
x = np.arange(0,100,0.2)
y = triangle(x)
y = sinus(x)

plt.plot(x, y)
plt.show()

#%% test chaotic model
x = [0.5]
y = [0]
z_c = [2]
sigma = 10
rho = 28
beta = 8/3

for t in range(2000):
    temp =update_activity(x[t], y[t], z_c[t], sigma, rho, beta)
    x.append(temp[0])
    y.append(temp[1])
    z_c.append(temp[2])
    
plt.plot(range(2001), x)
plt.show()

#%% plot of all activities together


#triangular signal
def triangle(t):
    a = 10
    return 2 * np.abs((t/a) - int((t/a)+0.5))

triangle = np.vectorize(triangle)


#all neuronal activities
increase = 0 #parameter used to plot the graphs without mess
for neuron in range(n_neuron):
    y = scaled(activity[neuron, 0, :])
    if neuron != 0:
        y += increase
        plt.plot(range(2000),y , color="blue")
    
    else:
        plt.plot(range(2000),y , color="blue")
    increase += max(scaled(activity[neuron-1, 0, :])) + 4
    
x = np.arange(TOTAL_TIME)
triangle_signal = triangle(x) + increase

plt.plot(range(2000), triangle_signal , color="red")
plt.xlabel("time (ms)")
plt.show()

#%% sinus

def sinus(x):
    return 5 *np.sin(2*np.pi*3*x)
sinus = np.vectorize(sinus)
x = np.arange(0, 10, 0.01)
y = sinus(x)

plt.plot(x, y)
plt.show()

