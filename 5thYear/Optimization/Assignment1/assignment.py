from sympy import *
import numpy
import matplotlib.pyplot as plt
from statistics import mean

plt.rcParams.update({'font.size': 22})

# Part A
def partA():
    # Part(i)
    x = symbols('x')
    expr = x**4
    dydx = diff(expr, x)
    print(dydx)

    # Part(ii)
    deriv = lambdify(x, dydx, "numpy")
    a = numpy.arange(-0.5, 0.5, 0.01)
    derivative = deriv(a)

    # finite difference with def = 0.01
    # f'(x) = f(x + del) - f(x) / del
    delta = 0.01
    finiteDerive = lambda x, delta : ( ((x + delta)**4 - x**4) / delta)
    finiteDifference = finiteDerive(a, delta)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a, derivative)
    ax.plot(a, finiteDifference)
    ax.set_xlabel("x")
    ax.set_ylabel("dydx")
    ax.legend(["Derivative", "Finite Difference"])
    fig.suptitle("Difference in derivative achieved through differentiation and finite difference")
    plt.show()

    #Part(iii)
    deltaValues = [0.001, 0.05, 0.1, 0.5, 1]
    differences = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a, derivative)
    for delta in deltaValues:
        finiteDifference = finiteDerive(a, delta)
        differences.append(mean(abs(derivative - finiteDifference)))
        ax.plot(a, finiteDifference)
    ax.legend(["Derivative", "Delta = 0.001", "Delta = 0.05", "Delta = 0.1", "Delta = 0.5", "Delta = 1"])
    fig.suptitle("Effects of different delta values on finite difference")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(deltaValues, differences)
    ax.set_xlabel("Delta Value")
    ax.set_ylabel("Mean difference")
    fig.suptitle("Mean difference between finite difference and differentiation")
    plt.show()    

def gradientDescent(fn, x0, alpha, num_iters=50):
    x = x0
    X = numpy.array([x])
    F = numpy.array(fn.f(x))
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        X = numpy.append(X, [x], axis = 0)
        F = numpy.append(F, fn.f(x))
    return(X, F)

def bareBonesGradientDescent(fn, x0, alpha, num_iters=50):
    x = x0
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step

class quanticFunction():
    def f(self, x):
        return x**4
    def df(self, x):
        return 4*x**3

# Part B
def partB():
    # Part(i) is in gradientDescent function

    # Part(ii)
    fn = quanticFunction()
    (X, F) = gradientDescent(fn, 1, 0.1)

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

    # Part(iii)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    startingPoints = range(-1, 3)
    for x in startingPoints:
        (X, F) = gradientDescent(fn, x, 0.1, num_iters=50)
        ax1.plot(iterations, X)
        ax2.plot(iterations, F)
    ax2.set_xlabel('iterations')
    ax1.set_ylabel('x')
    ax2.set_ylabel('f(x)')
    fig.legend(startingPoints)
    fig.suptitle("Effect of changing starting value of x with alpha = 0.1")
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    stepSizes = [0.0001, 0.001, 0.1, 0.5]
    for alpha in stepSizes:
        (X, F) = gradientDescent(fn, 1, alpha, num_iters=50)
        ax1.plot(iterations, X)
        ax2.plot(iterations, F)
    ax2.set_xlabel('iterations')
    ax1.set_ylabel('x')
    ax2.set_ylabel('f(x)')
    fig.legend(stepSizes)
    fig.suptitle("Effect of changing step size value with starting x = 1")
    plt.show()

class quadraticFunction():
    def __init__(self, gamma):
        self.gamma = gamma

    def f(self, x):
        return self.gamma*x**2
    
    def df(self, x):
        return 2*self.gamma*x

class absoluteFunction():
    def __init__(self, gamma):
        self.gamma = gamma

    def f(self, x):
        return self.gamma * abs(x)
    
    def df(self, x):
        return (self.gamma * x) / abs(x)

# Part C
def partC():
    # Part(i)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    smallGamma = numpy.arange(0, 1, 0.5)
    gammas = numpy.arange(1, 4)
    gammas = numpy.concatenate((smallGamma, gammas))
    iterations = range(51)
    for gamma in gammas:
        fn = quadraticFunction(gamma)
        (X, F) = gradientDescent(fn, 1, 0.1)
        ax1.plot(iterations, X)
        ax2.plot(iterations, F)
    ax2.set_xlabel('iterations')
    ax1.set_ylabel('x')
    ax2.set_ylabel('f(x)')
    fig.legend(gammas)
    fig.suptitle("Effect of changing gamma values on quadratic function")
    plt.show()

    specialGammas = [-1, 10, 11]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    smallIterations = range(11)
    for gamma in specialGammas:
        fn = quadraticFunction(gamma)
        (X, F) = gradientDescent(fn, 1, 0.1, num_iters=10)
        ax1.plot(smallIterations, X)
        ax2.plot(smallIterations, F)
    ax2.set_xlabel('iterations')
    ax1.set_ylabel('x')
    ax2.set_ylabel('f(x)')
    fig.legend(specialGammas)
    fig.suptitle("Special case gamma values on the quadratic function")
    plt.show()

    # Part(ii)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    gammas = numpy.arange(-1, 3)
    for gamma in gammas:
        fn = absoluteFunction(gamma)
        (X, F) = gradientDescent(fn, 1, 0.1)
        ax1.plot(iterations, X)
        ax2.plot(iterations, F)
    fig.legend(gammas)
    ax2.set_xlabel('iterations')
    ax1.set_ylabel('x')
    ax2.set_ylabel('f(x)')
    fig.suptitle("Effect of gamma values on absolute function")
    plt.show()

partA()
partB()
partC()