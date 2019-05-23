# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:45:22 2019

@author: Antoine
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:33:05 2019

@author: Antoine
"""

#%% functions 

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy import outer

def scaled(x):
    """returns the rescale activity such that it is centered and weighted by 
    the standard deviation"""
    return (x - np.mean(x)) / np.std(x)


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


def update_p(p, r):
    """Returns the updated matrix p(t).
    
    ---
    Arguments:
    - p is a diagonal matrix
    - r is the neurons activity vector
    ---
    Return: Matrix p of dimension NxN
    """
    Pr = dot(p,r)
    rpr = dot(r,Pr)
    c = 1/(1+rpr)
    return p - outer(Pr,Pr*c)


def triangle(t):
    """returns the desired value of the output, which is a triangle signal
    
    ---
    Arguments:
    - t is the time step
    """
    a = 1000
    return (np.abs((t/a) - int((t/a)+0.5)))


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

def sinus(t):
    A = 10
    return A*np.sin(t*(np.pi*2)/400)



def verif(x):  
    print(np.min(x), np.max(x))
    

#%% setting initial conditions
np.random.seed(22)
n_neuron = 1000
TOTAL_TIME = 8000
alpha = 1
g = 1.5         #controls the chaos of the system
gz = 1
p = np.identity(n_neuron) / alpha
tau = 0.01
dt = 0.001
time = np.arange(TOTAL_TIME)

#the two ws
Jg = np.random.normal(0, 1/np.sqrt(n_neuron),size=(n_neuron, n_neuron))
Jz = np.random.uniform(-1, 1, size=n_neuron)
w = np.zeros(n_neuron)
#w= np.random.normal(0, 1/np.sqrt(1*n_neuron),size=n_neuron)


target = sinus(time)
#target = np.array([np.sin(t/10) for t in time])
#target = np.array([triangle(t) for t in time])  #desired output we want the model to mimic
target = scaled(target)


#%% running 
x = np.random.normal(size=n_neuron)

save_x1 = []
save_x2 = []

all_w = []

errors=[]
z = []
z.append(compute_z(w, np.tanh(x)))

learn_every = 5
for i in time:
    save_x1.append(x[0])
    save_x2.append(x[1])
    
    x_old = x
    r_old = np.tanh(x_old)
    
    #update of network internal activity
    x = x_old  + dt*(1/tau)*(-x_old + g*dot(Jg, r_old) + gz*Jz*z[i])
    
    f = target[i]
    r = np.tanh(x)
    z.append(compute_z(w, r)) 
    
    if TOTAL_TIME/1.5>i and i%learn_every==0:
        e = error(w, r, f)
        errors.append(e)
        old_w = w
        
        #update
        p = update_p(p, r)
        w = w - e*dot(r,p)
        
        all_w.append(np.mean(np.abs(w-old_w)))
        new_e = error(w,r,f)
        if new_e / e ==1:
            learning_end = 1
    
    print(f"{round(100*i/len(time-1))} %")
    


plt.plot(time, z[:-1], color="red")
plt.plot(time, target,':', color="blue")
plt.title("training")
plt.xlabel("time steps")
plt.ylabel("activity")
plt.show()

plt.plot(time[:(len(all_w))], all_w)
plt.xlabel("time")
plt.ylabel("update amplitude of w")
plt.show()

  

