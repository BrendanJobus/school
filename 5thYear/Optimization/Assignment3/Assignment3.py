import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sympy import *

def generate_trainingdata(m = 25):
    # creates a randomised m x 2 set of training data
    return np.array([0,0]) + 0.25 * np.random.randn(m, 2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y = 0; count = 0
    for w in minibatch:
        z = x - w - 1
        y = y + Min(13 * (z[0]**2 + z[1]**2), (z[0] + 6)**2 + (z[1] + 10)**2)   
        count = count + 1
    return y / count

def fn(x, w):
    z = x - w - 1
    return Min(13 * (z[0]**2 + z[1]**2), (z[0] + 6)**2 + (z[1] + 10)**2)  

def dfdx0(x, w):
    return (-26*w[0] + 26*x[0] - 26)*Heaviside(-13*(-w[0] + x[0] - 1)**2 + (-w[0] + x[0] + 5)**2 - 13*(-w[1] + x[1] - 1)**2 + (-w[1] + x[1] + 9)**2) + (-2*w[0] + 2*x[0] + 10)*Heaviside(13*(-w[0] + x[0] - 1)**2 - (-w[0] + x[0] + 5)**2 + 13*(-w[1] + x[1] - 1)**2 - (-w[1] + x[1] + 9)**2)

def dfdx1(x, w):
    return (-26*w[1] + 26*x[1] - 26)*Heaviside(-13*(-w[0] + x[0] - 1)**2 + (-w[0] + x[0] + 5)**2 - 13*(-w[1] + x[1] - 1)**2 + (-w[1] + x[1] + 9)**2) + (-2*w[1] + 2*x[1] + 18)*Heaviside(13*(-w[0] + x[0] - 1)**2 - (-w[0] + x[0] + 5)**2 + 13*(-w[1] + x[1] - 1)**2 - (-w[1] + x[1] + 9)**2)

def df(x, w):
    return [dfdx0(x, w), dfdx1(x, w)]

class constant_step_size():
    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, derivative, _):
        return self.alpha * derivative
    
class Polyak():
    def __init__(self, fn, minf, epsilon = 0.0001):
        self.fn = fn
        self.fmin = minf
        self.epsilon = epsilon

    def step(self, derivative, x):
        numerator = self.fn.f(x) - self.fmin
        denominator = np.sum(derivative * derivative) + self.epsilon
        alpha = numerator / denominator
        return alpha * derivative
    
class RMSProp():
    def __init__(self, base_alpha, beta, base_sum = np.array([0, 0]), epsilon = 0.001):
        self.base_alpha = base_alpha
        self.beta = beta
        self.sum = base_sum
        self.epsilon = epsilon
    
    def step(self, derivative, _):
        if self.sum.any() == 0:
            alpha = self.base_alpha
        else:
            alpha = self.base_alpha / (np.sqrt(self.sum) + self.epsilon)
        self.sum = (self.beta * self.sum) + ( (1 - self.beta) * (derivative) )
        return alpha * derivative
    
class HeavyBall():
    def __init__(self, base_alpha, beta):
        self.alpha = base_alpha
        self.beta = beta
        self.sum = 0

    def step(self, derivative, _):
        self.sum = (self.beta * self.sum) + (self.alpha * derivative)
        return self.sum

class Adam():
    def __init__(self, alpha, b1, b2, epsilon = 0.00001):
        self.alpha = alpha
        self.beta_one = b1
        self.beta_two = b2
        self.m_hat_beta = 1
        self.v_hat_beta = 1
        self.m = 0
        self.v = 0
        self.epsilon = epsilon

    def step(self, derivative, _):
        self.m = (self.beta_one * self.m) + (1 - self.beta_one) * derivative
        self.v = (self.beta_two * self.v) + (1 - self.beta_two) * (derivative * derivative)

        self.m_hat_beta = self.m_hat_beta * self.beta_one
        self.v_hat_beta = self.v_hat_beta * self.beta_two

        m_hat = self.m / (1 - self.m_hat_beta)
        v_hat = self.v / (1 - self.v_hat_beta)

        return self.alpha * (m_hat / (np.sqrt(v_hat) + self.epsilon))
    
def eval_func():
    w, n = symbols('w n')
    l, x = symbols('l x', cls=Function)

    x0, x1, w0, w1 = symbols("x0 x1 w0 w1", real = True)
    J = Min( 13 * ( (x0 - w0 - 1)**2 + (x1 - w1 - 1)**2 ), ( (x0 - w0 - 1) + 6)**2 + ( (x1 - w1 - 1) + 10)**2 )
    dfdx0 = diff(J, x0)
    dfdx1 = diff(J, x1)

    print(dfdx0)
    print(dfdx1)

def gradientDescent(fn, x0, training_data, alpha, batch_size = 5, single_variable = False, num_iters=50):
    xt = x0
    if not single_variable:
        X1 = np.array([xt[0]])
        X2 = np.array([xt[1]])
        F = np.array(np.mean([fn(xt, w) for w in training_data], axis=0))
        A1 = np.array([])
        A2 = np.array([])
    else:
        X = np.array(xt)
        F = np.array(fn(xt))
        A = np.array([])

    for _ in range(num_iters):
        derivative = np.mean([df(xt, w) for w in training_data], axis=0)
        step = alpha.step(derivative, xt)
        xt1 = xt - step
        xt = xt1
        if not single_variable:
            X1 = np.append(X1, [xt[0]], axis = 0)
            X2 = np.append(X2, [xt[1]], axis = 0)
            F = np.append(F, np.mean([fn(xt, w) for w in training_data], axis=0))
            A1 = np.append(A1, step[0])
            A2 = np.append(A2, step[1])
        else:
            X = np.append(X, [xt[0]], axis = 0)
            F = np.append(F, fn(xt))
            A = np.append(A, step)

    if not single_variable:
        return (X1, X2, F, A1, A2)
    else:
        return (X, F, A)
    
# fn is function that we are minimising, x0 is first estimates of minimum x, alpha is a function that generates the step size
# it will take a two parameters: D, the derivative to use for the step calculation, and x, the current estimate of the minimum,
# batch_size is the mini_batch size and num iters is the number of epochs that will take place
def SGD(fn, x0, training_data, alpha, batch_size = 5, num_iters = 50):
    m = len(training_data)
    x = x0

    for _ in range(num_iters):
        np.random.shuffle(training_data)
        for i in np.arange(0, m, batch_size):
            batch_indeces = np.arange(i, i + batch_size)
            batch_array = [training_data[i] for i in batch_indeces]

            # Calculating the derivative of loss(x, w)
            mini_batch_derivatives = [df(x, w) for w in batch_array]

            # Calculating 1/b sum(loss(x, w))
            approximate_derivative = np.mean(mini_batch_derivatives, axis=0)

            x = x - alpha.step(approximate_derivative, x)

    return x

m = 25
training_data = generate_trainingdata(m)
batch_size = 5

#eval_func()
alpha = constant_step_size(0.05)

# (b) (i)
# (X1, X2, F, A1, A2) = gradientDescent(fn, [3, 3], training_data, alpha)
# print(X1[-1], X2[-1])

# (b) (ii)
# x = SGD(fn, [3, 3], training_data, alpha)
# print(x)

# (b) (iii)
batch_sizes = [10]
for s in batch_sizes:
    x = SGD(fn, [3, 3], training_data, alpha, batch_size = s)
    print(x)
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Grab some test data.
X1 = np.arange(0, 10, 0.01)
X2 = np.arange(0, 10, 0.01)
X = [[X1[i], X2[i]] for i in range(len(X1))]
Y = generate_trainingdata()

#print(X)
#print(Y)
#Z = f(X, Y)

#X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

#plt.show()