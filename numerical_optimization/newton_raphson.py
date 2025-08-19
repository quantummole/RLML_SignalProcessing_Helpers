# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import numpy as np
from matplotlib import pyplot as plt
class NewtonRaphson :
    def __init__(self,function,derivative,epsilon=1e-3) :
        self.func = function
        self.derivative = derivative
        self.eps = epsilon
    def search(self,x_0,max_epochs) :
        epochs = max_epochs
        f_0 = self.func(x_0)
        xs = [x_0]
        fs = [f_0]
        while epochs > 0 and np.abs(f_0)>self.eps :
            f1_0 = self.derivative(x_0)
            x_0 = x_0 - f_0/f1_0
            f_0 = self.func(x_0)
            epochs = epochs - 1
            xs.append(x_0)
            fs.append(f_0)
        return x_0,f_0,xs,fs
    
if __name__ == '__main__' :
    func = lambda x :np.cos(3*x) - x

    xs = np.linspace(-2,2,100)
    ys = [func(x) for x in xs]
    plt.plot(xs,ys)
    plt.show()
    
    derivative = lambda x : -3*np.sin(3*x) -1
    solver = NewtonRaphson(func,derivative)
    x,f,_,_ = solver.search(-0.75,100)
    print("Actual derivative ",x,f)

    delta = 1e-4
    derivative = lambda x : (func(x+delta)-func(x))/delta
    solver = NewtonRaphson(func,derivative)
    x,f,x0s,fs = solver.search(-0.75,100)
    print("Derivative using finite difference ",x,f)


    xs = np.linspace(x-1,x+1,100)
    ys = [func(x) for x in xs]
    plt.plot(xs,ys)
    plt.plot(x0s,fs)
    plt.show()
    
    func = lambda x :np.cos(3*x) - x
    derivative = lambda x : -3*np.sin(3*x) -1
    solver = NewtonRaphson(func,derivative)
    x,f,_,_ = solver.search(0.75,100)
    print("Actual derivative ",x,f)

    delta = 1e-4
    derivative = lambda x : (func(x+delta)-func(x))/delta
    solver = NewtonRaphson(func,derivative)
    x,f,x0s,fs = solver.search(0.75,100)
    print("Derivative using finite difference ",x,f)


    xs = np.linspace(x-1,x+1,100)
    ys = [func(x) for x in xs]
    plt.plot(xs,ys)
    plt.plot(x0s,fs)
    plt.show()
    