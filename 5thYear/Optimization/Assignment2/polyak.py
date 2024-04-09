import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sympy import *

def gradientDescent(fn, x0, alpha, num_iters=50):
    x = x0
    X = np.array([x])
    F = np.array(fn.f(x))
    for _ in range(num_iters):
        step = alpha.stepsize(x) * fn.dd(x)
        x = x - step
        X = np.append(X, [x], axis = 0)
        F = np.append(F, fn.f(x))
    return(X, F)

def gradientDescentWithTwoVariables(fn, x0, alpha, num_iters = 50):
    x = x0
    X1 = np.array([x[0]])
    X2 = np.array([x[1]])
    F = np.array(fn.f(x))
    for _ in range(num_iters):
        step = alpha.stepsize(x)
        x = x - step
        X1 = np.append(X1, [x[0]], axis = 0)
        X2 = np.append(X2, [x[1]], axis = 0)
        F = np.append(F, fn.f(x))
    return(X1, X2, F)

class Polyak():
    def __init__(self, fn, minf, epsilon = 0):
        self.fn = fn
        self.fmin = minf
        self.epsilon = epsilon

    def stepsize(self, x):
        numerator = self.fn.f(x) - self.fmin
        denominator = self.fn.dd(x) * self.fn.dd(x) + self.epsilon
        #print(numerator)
        #print(denominator)
        alpha = numerator / denominator
        #print(alpha)
        # print(numerator)
        # print(denominator)
        # print(alpha)
        return alpha
    
class function():
    def f(self, x):
        return x**2
    
    def dd(self, x):
        return 2*x

class function_one():
    def f(self, x):
        return 5 * (x[0] - 8)**4 + 6 * (x[1] - 3)**2
    
    def dd(self, x):
        return np.array([self.dfdx0(x[0]), self.dfdx1(x[1])])

    def dfdx0(self, x0):
        return 20 * (x0 - 8)**3
    
    def dfdx1(self, x1):
        return 12 * x1 - 36
    
def singleVariablePolyak():
    fn = function()
    x = np.array([1])
    fmin = np.array([0])
    alpha = Polyak(fn, fmin)
    (X, F) = gradientDescent(fn, x, alpha)
    iterations = range(51)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(iterations, X)
    ax2.plot(iterations, F)

    ax2.set_xlabel('iterations')
    ax1.set_ylabel('x')
    ax2.set_ylabel('f(x)')
    fig.suptitle("Effect of each iteration of gradient descent on x and f(x)")
    plt.show() 

fn = function_one()
x = np.array([-5, -5])
fstar = np.array([7.5, 4])
alpha = Polyak(fn, fstar)

# step = alpha.stepsize(x) * fn.dd(x)
# print(step)
# x = x - step
# print(x)


# (X, F) = gradientDescent(fn, x, alpha, num_iters=1)
# X1 = X[:, 0]
# X2 = X[:, 1]
# iterations = range(2)
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax1.plot(iterations, X2)
# ax2.plot(iterations, F)

# ax2.set_xlabel('iterations')
# ax1.set_ylabel('x2')
# ax2.set_ylabel('f(x)')
# fig.suptitle("Effect of each iteration of gradient descent on x and f(x)")
# plt.show() 

#fig = plt.figure()
#ax = fig.gca(projection="3d")

x = np.arange(7, 9, 0.1)
y = np.arange(2, 4, 0.1)
X, Y = np.meshgrid(x, y)
# f = lambda a, b : np.sin(np.sqrt(a ** 2 + b ** 2))
# Z = f(X, Y)

objective = lambda a, b : 5 * (a - 8)**4 + 6 * (b - 3)**4

Z = objective(X, Y)

figure = plt.figure()
axis = figure.gca(projection="3d")
axis.plot_surface(X, Y, Z, cmap='jet')

# ax.contour3D(X, Y, Z, 50)
axis.set_xlabel('x')
axis.set_ylabel('y')
axis.set_zlabel('z')

plt.show()

#(X, F) = gradientDescent(fn, x, alpha)
# iterations = range(51)
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax1.plot(iterations, X)
# ax2.plot(iterations, F)

# ax2.set_xlabel('iterations')
# ax1.set_ylabel('x')
# ax2.set_ylabel('f(x)')
# fig.suptitle("Effect of each iteration of gradient descent on x and f(x)")
# plt.show() 