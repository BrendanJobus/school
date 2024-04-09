import numpy as np
import matplotlib.pyplot as plt
from sympy import *

# function: 5*(x-8)^4+6*(y-3)^2
# function: Max(x-8,0)+6*|y-3|

# x1, x2 = symbols("x1 x2")
# f1 = 5 * (x1 - 8)**4 + 6 * (x2 - 3)**2
# f2 = Max(x1 - 8, 0) + 6 * Abs(x2 - 3)
# f1 = function()
# f2 = function(f2)

def vars():
    x, y = symbols("x y", real = True)
    f1 = 5 * (x - 8)**4 + 6 * (y - 3)**2
    f2 = Max(x - 8, 0) + 6 * Abs(y - 3)

    df1dx = diff(f1, x)
    df1dy = diff(f1, y)
    df2dx = diff(f2, x)
    df2dy = diff(f2, y)

    print("df1/dx:", df1dx)
    print("df1/dy:", df1dy)
    print("df2/dx:", df2dx)
    print("df1/dy:", df2dy)

class function():
    def f(self, x):
        return x**2
    
    def df(self, x):
        return 2*x

class function_one():
    def __init__(self):
        self.name = "5(x - 8)^4 + 6(y - 4)^2"

    def f(self, x):
        return 5 * (x[0] - 8)**4 + 6 * (x[1] - 3)**2
    
    def fx(self, x, y):
        return 5 * (x - 8)**4 + 6 * (y - 3)**2

    def df(self, x):
        return np.array([self.dfdx0(x[0]), self.dfdx1(x[1])])

    def dfdx0(self, x0):
        return 20 * (x0 - 8)**3
    
    def dfdx1(self, x1):
        return 12 * x1 - 36
    
class function_two():
    def __init__(self):
        self.name = "Max(x - 8, 0) + 6|y - 3|"

    def f(self, x):
        return np.maximum(x[0] - 8, 0) + 6 * np.abs(x[1] - 3)
    
    def fx(self, x, y):
        return np.maximum(x - 8, 0) + 6 * np.abs(y - 3)

    def df(self, x):
        return np.array([self.dfdx0(x[0]), self.dfdx1(x[1])])

    def dfdx0(self, x0):
        return np.heaviside(x0 - 8, 0.5)
    
    def dfdx1(self, x1):
        return 6 * np.sign(x1 - 3)

class ReLu_function():
    def __init__(self):
        self.name = "ReLu"

    def f(self, x):
        return np.maximum(0, x[0])

    def df(self, x):
        return np.heaviside(x, 0.5)

class Polyak():
    def __init__(self, fn, minf, epsilon = 0.0001):
        self.fn = fn
        self.fmin = minf
        self.epsilon = epsilon

    def step(self, x):
        numerator = self.fn.f(x) - self.fmin
        denominator = np.sum(self.fn.df(x) * self.fn.df(x)) + self.epsilon
        alpha = numerator / denominator
        return alpha * self.fn.df(x)
    
class RMSProp():
    def __init__(self, fn, base_alpha, beta, base_sum = np.array([0, 0]), epsilon = 0.001):
        self.fn = fn
        self.base_alpha = base_alpha
        self.beta = beta
        self.sum = base_sum
        self.epsilon = epsilon
    
    def step(self, x):
        if self.sum.any() == 0:
            alpha = self.base_alpha
        else:
            alpha = self.base_alpha / (np.sqrt(self.sum) + self.epsilon)
        self.sum = (self.beta * self.sum) + ( (1 - self.beta) * (self.fn.df(x)**2) )
        return alpha * self.fn.df(x)

class HeavyBall():
    def __init__(self, fn, base_alpha, beta):
        self.fn = fn
        self.alpha = base_alpha
        self.beta = beta
        self.sum = 0

    def step(self, x):
        self.sum = (self.beta * self.sum) + (self.alpha * self.fn.df(x))
        return self.sum

class Adam():
    def __init__(self, fn, alpha, b1, b2, epsilon = 0.00001):
        self.fn = fn
        self.alpha = alpha
        self.beta_one = b1
        self.beta_two = b2
        self.m_hat_beta = 1
        self.v_hat_beta = 1
        self.m = 0
        self.v = 0
        self.epsilon = epsilon

    def step(self, x):
        self.m = (self.beta_one * self.m) + (1 - self.beta_one) * self.fn.df(x)
        self.v = (self.beta_two * self.v) + (1 - self.beta_two) * (self.fn.df(x) * self.fn.df(x))

        self.m_hat_beta = self.m_hat_beta * self.beta_one
        self.v_hat_beta = self.v_hat_beta * self.beta_two

        m_hat = self.m / (1 - self.m_hat_beta)
        v_hat = self.v / (1 - self.v_hat_beta)

        return self.alpha * (m_hat / (np.sqrt(v_hat) + self.epsilon))

def gradientDescent(fn, x0, alpha, single_variable = False, num_iters=50):
    xt = x0
    if not single_variable:
        X1 = np.array([xt[0]])
        X2 = np.array([xt[1]])
        F = np.array(fn.f(xt))
        A1 = np.array([])
        A2 = np.array([])
    else:
        X = np.array(xt)
        F = np.array(fn.f(xt))
        A = np.array([])

    for _ in range(num_iters):
        step = alpha.step(xt)
        xt1 = xt - step
        xt = xt1
        if not single_variable:
            X1 = np.append(X1, [xt[0]], axis = 0)
            X2 = np.append(X2, [xt[1]], axis = 0)
            F = np.append(F, fn.f(xt))
            A1 = np.append(A1, step[0])
            A2 = np.append(A2, step[1])
        else:
            X = np.append(X, [xt[0]], axis = 0)
            F = np.append(F, fn.f(xt))
            A = np.append(A, step)

    if not single_variable:
        return(X1, X2, F, A1, A2)
    else:
        return (X, F, A)

def basic_polyak(fn, x, fmin):
    alpha = Polyak(fn, fmin)
    (X, F, A) = gradientDescent(fn, x, alpha, single_variable = True)
    return (X, F, A)

def basic_RMSProp(fn, x):
    b = 0.9
    a0 = 0.12

    alpha = RMSProp(fn, a0, b, base_sum = np.array([0]))
    (X, F, A) = gradientDescent(fn, x, alpha, single_variable = True)

    return (X, F, A)

def basic_HB(fn, x):
    a0 = 0.05
    b = 0.25
    alpha = HeavyBall(fn, a0, b)
    (X, F, A) = gradientDescent(fn, x, alpha, single_variable = True)
    return(X, F, A)

def basic_Adam(fn, x, alpha = 2, beta_one = 0.9, beta_two = 0.999):
    alpha = Adam(fn, alpha, beta_one, beta_two)
    return gradientDescent(fn, x, alpha, single_variable = True)

def plot_basic_implementations(fn, x, fmin):
    (polyX, polyF, polyA) = basic_polyak(fn, x, fmin)
    (rmsX, rmsF, rmsA) = basic_RMSProp(fn, x)
    (hbX, hbF, hbA) = basic_HB(fn, x)
    (adamX, adamF, adamA) = basic_Adam(fn, x)

    fig = plt.figure()
    iterations = range(51)
    stepsize_iters = range(50)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.plot(iterations, polyX, alpha=0.8)
    ax2.plot(iterations, polyF, alpha=0.8)
    ax3.plot(stepsize_iters, polyA, alpha=0.8)

    ax1.plot(iterations, rmsX, alpha=0.8)
    ax2.plot(iterations, rmsF, alpha=0.8)
    ax3.plot(stepsize_iters, rmsA, alpha=0.8)

    ax1.plot(iterations, hbX, alpha=0.8)
    ax2.plot(iterations, hbF, alpha=0.8)
    ax3.plot(stepsize_iters, hbA, alpha=0.8)

    ax1.plot(iterations, adamX, alpha=0.8)
    ax2.plot(iterations, adamF, alpha=0.8)
    ax3.plot(stepsize_iters, adamA, alpha=0.8)

    ax3.set_xlabel('iterations')
    ax1.set_ylabel('x')
    ax2.set_ylabel('f(x)')
    ax3.set_ylabel('step size')
    fig.legend(["polyak", "RMSProp", "HeavyBall", "Adam"])
    #fig.suptitle("Effect of each iteration of gradient descent on x and f(x)")
    fig.suptitle("Optimization functions applied to {} function with starting point {}".format(fn.name, x[0]))
    plt.show() 

    if x[0] == 100:
        # try Adam with larger alpha on x = 100
        X, F, A = basic_Adam(fn, x, alpha = 10)
        fig = plt.figure()
        iterations = range(51)
        stepsize_iters = range(50)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        ax1.plot(iterations, X, alpha=0.8)
        ax2.plot(iterations, F, alpha=0.8)
        ax3.plot(stepsize_iters, A, alpha=0.8)

        ax3.set_xlabel('iterations')
        ax1.set_ylabel('x')
        ax2.set_ylabel('f(x)')
        ax3.set_ylabel('step size')
        fig.suptitle("Adam with alpha = 10 applied to {} function with starting point {}".format(fn.name, x[0]))
        plt.show() 

def multivariate_polyak(fn, x, fmin = 0, name = "Polyak"):
    alpha = Polyak(fn, fmin)
    (X1, X2, F, A1, A2) = gradientDescent(fn, x, alpha)
    plot_multivariate_optimization(X1, X2, F, A1, A2, fn.name, name)

def multivariate_RMSProp(fn, x, a = 0.05, b = 0.9, name = "RMSProp"):
    alpha = RMSProp(fn, a, b)
    (X1, X2, F, A1, A2) = gradientDescent(fn, x, alpha)
    plot_multivariate_optimization(X1, X2, F, A1, A2, fn.name, name)

def multivariate_HB(fn, x, a0 = 0.01, b = 0.9, name = "Heavy Ball"):
    alpha = HeavyBall(fn, a0, b)
    (X1, X2, F, A1, A2) = gradientDescent(fn, x, alpha)
    plot_multivariate_optimization(X1, X2, F, A1, A2, fn.name, name)

def multivariate_Adam(fn, x, alpha = 0.1, beta_one = 0.9, beta_two = 0.999, name = "Adam"):
    alpha = Adam(fn, alpha, beta_one, beta_two)
    (X1, X2, F, A1, A2) = gradientDescent(fn, x, alpha)
    print(F.shape)
    plot_multivariate_optimization(X1, X2, F, A1, A2, fn.name, name)

def plot_multivariate_optimization(X1, X2, F, A1, A2, function_name, optimization_method):
    fig = plt.figure(layout="constrained")
    iterations = range(51)
    stepsize_iters = range(50)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(iterations, X1)
    ax2.plot(iterations, X2)
    ax3.plot(iterations, F)
    ax4.plot(stepsize_iters, A1)
    ax4.plot(stepsize_iters, A2)

    ax3.set_xlabel('iterations')
    ax4.set_xlabel('iterations')
    ax1.set_ylabel('x1')
    ax2.set_ylabel('x2')
    ax3.set_ylabel('f(x)')
    ax4.set_ylabel('step size')
    ax4.legend(["X1", "X2"])
    fig.suptitle("Effect of each iteration of {} on {}".format(optimization_method, function_name))
    plt.show()

def plot_optimization_comparison(fn, algorithm_name, F, A1, A2, theta1s, theta2s, alpha, beta, beta_two = 0):
    # Plot f(x) vs iteration
    iterations = range(51)
    fig = plt.figure()
    plt.plot(iterations, F.T, alpha=0.8)
    fig.supxlabel("iterations")
    fig.supylabel("f(x)")
    plt.legend(alpha)
    if algorithm_name == "Adam":
        fig.suptitle("Effect of different alpha values on function value with beta one = {} and beta two {}".format(beta, beta_two))
    else:
        fig.suptitle("Effect of different alpha values on function value with beta = {}".format(beta))
    plt.show()

    # Plot step size vs iteration
    # Step size 1
    iterations = range(50)
    fig = plt.figure()
    plt.plot(iterations, A1.T, alpha=0.8)
    fig.supxlabel("iterations")
    fig.supylabel("step size 1")
    plt.legend(alpha)
    if algorithm_name == "Adam":
        fig.suptitle("Effect of different alpha values on step size with beta one = {} and beta two {}".format(beta, beta_two))
    else:
        fig.suptitle("Effect of different alpha values on step size with beta = {}".format(beta))
    plt.show()

    # Step size 2
    fig = plt.figure()
    plt.plot(iterations, A2.T, alpha=0.8)
    fig.supxlabel("iterations")
    fig.supylabel("step size 2")
    plt.legend(alpha)
    if algorithm_name == "Adam":
        fig.suptitle("Effect of different alpha values on x2 step size with beta one = {} and beta two = {}".format(beta, beta_two))
    else:
        fig.suptitle("Effect of different alpha values on x2 step size with beta = {}".format(beta))
    plt.show()

    # Plot contour plot
    buffer = 1
    xMin = np.amin(theta1s) - buffer
    xMax = np.amax(theta1s) + buffer
    yMin = np.amin(theta2s) - buffer
    yMax = np.amax(theta2s) + buffer
    x = np.arange(xMin, xMax, 0.1)
    y = np.arange(yMin, yMax, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = fn.fx(X, Y)

    plt.contour(X, Y, Z)
    plt.xlabel("theta 1"); plt.ylabel("theta 2")
    plt.plot(theta1s.T, theta2s.T, alpha=0.8)
    plt.legend(alpha)
    if algorithm_name == "Adam":
        plt.suptitle("Contour plot with beta one = {} and beta two = {}".format(beta, beta_two))
    else:
        plt.suptitle("Contour plot with beta = {}".format(beta))
    plt.show()

def optimization_hyperparameter_comparison(fn, theta0, algorithm_name, alphas, beta, beta_two = [], default_alpha = 0.05, default_beta_one = 0.9, default_beta_two = 0.25, plot_contour = False):
    if algorithm_name == "RMSProp":
        alphaAl = lambda f, x, y : RMSProp(f, x, y)
    elif algorithm_name == "HeavyBall":
        alphaAl = lambda f, x, y : HeavyBall(f, x, y)
    else:
        alphaAl = lambda f, x, y : Adam(f, x, y, default_beta_two)

    for b in beta:
        theta1s = np.array([])
        theta2s = np.array([])
        function_values = np.array([])
        theta1_stepsizes = np.array([])
        theta2_stepsizes = np.array([])
        for a in alphas:
            alphaAlgorithm = alphaAl(fn, a, b)
            (THETA1, THETA2, F, A1, A2) = gradientDescent(fn, theta0, alphaAlgorithm)
            if function_values.size == 0:
                theta1s = np.array([THETA1])
                theta2s = np.array([THETA2])
                function_values = np.array([F])
                theta1_stepsizes = np.array([A1])
                theta2_stepsizes = np.array([A2])
            else:
                theta1s = np.vstack([theta1s, THETA1])
                theta2s = np.vstack([theta2s, THETA2])
                function_values = np.vstack([function_values, F])
                theta1_stepsizes = np.vstack([theta1_stepsizes, A1])
                theta2_stepsizes = np.vstack([theta2_stepsizes, A2])
        if algorithm_name == "Adam":
            plot_optimization_comparison(fn, algorithm_name, function_values, theta1_stepsizes, theta2_stepsizes, theta1s, theta2s, alphas, b, default_beta_two)
        else:
            plot_optimization_comparison(fn, algorithm_name, function_values, theta1_stepsizes, theta2_stepsizes, theta1s, theta2s, alphas, b)

    if algorithm_name == "Adam":
        alphaAl = lambda f, x, y : Adam(f, x, default_beta_one, y)
        for b in beta_two:
            del function_values
            function_values = np.array([])
            theta1_stepsizes = np.array([])
            theta2_stepsizes = np.array([])
            for a in alphas:
                alphaAlgorithm = alphaAl(fn, a, b)
                (_, _, F, A1, A2) = gradientDescent(fn, theta0, alphaAlgorithm)
                if function_values.size == 0:
                    function_values = np.array([F])
                    theta1_stepsizes = np.array([A1])
                    theta2_stepsizes = np.array([A2])
                else:
                    function_values = np.vstack([function_values, F])
                    theta1_stepsizes = np.vstack([theta1_stepsizes, A1])
                    theta2_stepsizes = np.vstack([theta2_stepsizes, A2])
            
            plot_optimization_comparison(fn, algorithm_name, function_values, theta1_stepsizes, theta2_stepsizes, theta1s, theta2s, alphas, default_beta_one, b)

def Part_B():
    # Function One
    fn = function_one()
    x = [9, 4]
    # multivariate_polyak(fn, x)
    # multivariate_RMSProp(fn, x)
    # multivariate_HB(fn, x)
    #multivariate_Adam(fn, x)

    # Start with values that will converge
    #optimization_hyperparameter_comparison(fn, x, "RMSProp", alphas = [0.01, 0.05, 0.1], beta = [0.25, 0.9])
    #optimization_hyperparameter_comparison(fn, x, "HeavyBall", alphas=[0.01, 0.025, 0.05], beta=[0.25, 0.9])
    #optimization_hyperparameter_comparison(fn, x, "Adam", alphas=[0.01, 0.1, 1, 2], beta=[0.25, 0.9], beta_two=[0.25, 0.999])

    # Function Two
    fn = function_two()
    x = [10, 5]
    # multivariate_polyak(fn, x)
    # multivariate_RMSProp(fn, x)
    # multivariate_HB(fn, x)
    # multivariate_Adam(fn, x)

    #optimization_hyperparameter_comparison(fn, x, "RMSProp", alphas = [0.01, 0.05, 0.1, 0.12], beta = [0.25, 0.9], plot_contour=True)
    #optimization_hyperparameter_comparison(fn, x, "HeavyBall", alphas=[0.01, 0.025, 0.05], beta=[0.25, 0.9], plot_contour=True)
    optimization_hyperparameter_comparison(fn, x, "Adam", alphas=[0.01, 0.1, 1, 2], beta=[0.25, 0.9], beta_two=[0.25, 0.999], plot_contour=True)

def Part_C():
    fn = ReLu_function()

    x = np.array([-1])
    plot_basic_implementations(fn, x, 0)
    
    x = np.array([1])
    plot_basic_implementations(fn, x, 0)

    x = np.array([100])
    plot_basic_implementations(fn, x, 0)

#vars()
fn = function()
x = np.array([1])
plot_basic_implementations(fn, x, 0)
Part_B()
Part_C()