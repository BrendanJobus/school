
import numpy as np
import pandas as pd
import time
import timeit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
plt.rc('font', size=10)
plt.rcParams['figure.constrained_layout.use'] = True
import sys

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

def func_one(x):
    return 5 * (x[0] - 8)**4 + 6 * (x[1] - 3)**2

def func_two(x):
    return np.maximum(x[0] - 8, 0) + 6 * np.abs(x[1] - 3)

def deriv_one(x):
    return np.array([20 * (x[0] - 8)**3, 12 * x[1] - 36])

def deriv_two(x):
    return np.array([np.heaviside(x[0] - 8, 0.5), 6 * np.sign(x[1] - 3)])

def bad_timing_global_random_search(cost_function, n, l, u, N):
    # set lowest fx to be inf at first so that the first set of parameters is always accepted
    lowest_fx = float('inf')
    lowest_x = []
    lowest_val_per_iter = []

    # Setting up timing
    times = []
    start_time = time.time()
    curr_time = start_time

    for _ in range(N):
        # Choose an element i at random between li and ui
        x = np.random.default_rng().uniform(l, u)
        fx = cost_function(x)
        if fx < lowest_fx:
            lowest_x = x
            lowest_fx = fx
        lowest_val_per_iter.append(lowest_fx)
        curr_time = time.time()
        times.append(curr_time - start_time)

    print(lowest_x, lowest_fx)
    return (lowest_x, lowest_fx, [times, lowest_val_per_iter])


# Parameters:
#   cost_function - the cost function we are trying to minimize
#   n - number of parameters
#   l - vector with mininum parameter value for each parameter
#   u - vector with maximum parameter value for each parameter
#   N - number of samples to take
# Returns:
#   lowest_x - the x value which makes fx the smallest from our samples
#   lowest_fx - the smallest fx we reached
#   fx_history - an array with the timeline of values of lowest_fx, and the corresponding times
def global_random_search(cost_function, n, l, u, N):
    # set lowest fx to be inf at first so that the first set of parameters is always accepted
    lowest_fx = float('inf')
    lowest_x = []
    lowest_val_per_iter = []
    x_hist = []

    for _ in range(N):
        # Choose an element i at random between li and ui
        x = np.random.default_rng().uniform(l, u)
        fx = cost_function(x)
        if fx < lowest_fx:
            lowest_x = x
            lowest_fx = fx
        lowest_val_per_iter.append(lowest_fx)
        x_hist.append(x)

    print(lowest_x, lowest_fx)
    return x_hist, lowest_val_per_iter

def modified_global_random_search(cost_function, n, l, u, N, M):
    # set lowest fx to be inf at first so that the first set of parameters is always accepted
    lowest_fx = float('inf')
    lowest_x = []
    x_hist = []
    all_values = []
    for _ in range(N):
        # Choose an element i at random between li and ui
        x = np.random.default_rng().uniform(l, u)
        fx = cost_function(x)
        if fx < lowest_fx:
            lowest_fx = fx
            lowest_x = x
        all_values.append([fx, x])
        x_hist.append(lowest_x)

    df = pd.DataFrame(all_values, columns=['fx', 'x'])
    df = df.sort_values(by=['fx'])
    m = df[0:M]
    # Now need to do the neighborhood search around m and repeat

    stop = False
    while(not stop):
        all_values = []
        for x in m['x']:
            # Need to generate N points around each x, and then keep the best M
            # Going to generate randomly from -1 to 1 around each point x
            delta = np.random.default_rng().uniform(-0.25, 0.25, M)
            deltaX = [[x[0] + d, x[1] + d] for d in delta]
            # Now get the function values
            fx = list(map(cost_function, deltaX))
            all_values.extend(zip(fx, deltaX))
        df = pd.DataFrame(all_values, columns=['fx', 'x'])
        df = df.sort_values(by=['fx'])
        new_m = df[0:M]
        if new_m['fx'].iloc[0] <= m['fx'].iloc[0]:
            stop = True
        m = new_m
        x_hist.append(m['x'].iloc[0])
    print(m['x'].iloc[0])
    return x_hist    

def gradientDescent(fn, dfdx, x0, alpha, num_iters=100):
    x = x0
    x_history = []
    for _ in range(num_iters):
        step = alpha * dfdx(x)
        x = x - step
        x_history.append(x)

    print(x)
    return x_history

def plot_fx_history(grs_fx_history, gd_fx_history, title):
    plt.plot(grs_fx_history[0], grs_fx_history[1])
    plt.plot(gd_fx_history[0], gd_fx_history[1])
    plt.semilogy()
    plt.xlabel("time"); plt.ylabel('f(x)')
    plt.title(title); plt.legend(['Global Random Search', 'Gradient Descent'])
    plt.show()

def plot_x_over_contour(fn, grs_x_hist, gd_x_hist, mgrs_x_hist, title):
    # Plot contour plot
    buffer = 1
    xMax = max(np.max(grs_x_hist[1][0]), np.max(gd_x_hist[1][0]), np.max(mgrs_x_hist[1][0])) - buffer
    yMax = max(np.max(grs_x_hist[1][1]), np.max(gd_x_hist[1][1]), np.max(mgrs_x_hist[1][1])) + buffer
    xMin = min(np.min(grs_x_hist[1][0]), np.min(gd_x_hist[1][0]), np.min(mgrs_x_hist[1][0])) - buffer
    yMin = min(np.min(grs_x_hist[1][1]), np.min(gd_x_hist[1][1]), np.min(mgrs_x_hist[1][1])) + buffer
    x = np.arange(xMin, xMax, 0.1)
    y = np.arange(yMin, yMax, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = fn([X, Y])

    plt.contour(X, Y, Z)
    plt.xlabel("x_0"); plt.ylabel("x_1")
    plt.plot(grs_x_hist[1][0], grs_x_hist[1][1], alpha=0.8)
    plt.plot(gd_x_hist[1][0], gd_x_hist[1][1], alpha=0.8)
    plt.plot(mgrs_x_hist[1][0], mgrs_x_hist[1][1], alpha=0.8)
    plt.legend(["Global Search", "Gradient Descent", "Modified Global Search"])
    plt.suptitle(title)
    plt.show()

def test_optimizations(func, algo_1, algo_2, plotting_func, derivative, l, u, x0, title):
    start_time = time.time()
    _, grs_fx_history = algo_1(func, len(l), l, u, 100)
    function_duration = time.time() - start_time
    start_time = 0
    grs_fx_history = [np.linspace(start_time, function_duration, num=len(grs_fx_history)), grs_fx_history]
    
    alpha = 0.005
    start_time = time.time()
    gd_x_history = algo_2(func, derivative, x0, alpha, 100)
    function_duration = time.time() - start_time
    start_time = 0
    gd_fx_history = list(map(func, gd_x_history))
    gd_fx_history = [np.linspace(start_time, function_duration, num=len(gd_fx_history)), gd_fx_history]

    # For global random search, we have no derivative evaluations, but we do have 100(or N)
    # funciton evaluations
    # For gradient descent, with num_iters = 100, we have 100(or N) derivative evaluations, but 
    # no function evaluations

    # We can see the effect of this in how difficult the derivative are to do, for func_one, they are
    # easy, however, func_two they are complex(Heaviside)

    plotting_func(grs_fx_history, gd_fx_history, title)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def conv_net(x):
    keras.backend.clear_session()
    batch_size, alpha, beta_1, beta_2, epochs = x
    # Setting up adam to use the hyperparameters that we have provided
    opt = tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=beta_1, beta_2=beta_2)

    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    n=5000
    x_train = x_train[1:n]; y_train=y_train[1:n]
    #x_test=x_test[1:500]; y_test=y_test[1:500]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    use_saved_model = False
    if use_saved_model:
        model = keras.models.load_model("cifar.model")
    else:
        model = keras.Sequential()
        model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
        model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
        # Base compile
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.summary()

        history = LossHistory()

        #batch_size = 128
        #epochs = 20
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        return history.losses[-1]

        # model.save("cifar.model")
        # plt.subplot(211)
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.subplot(212)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss'); plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.show()

    # preds = model.predict(x_train)
    # y_pred = np.argmax(preds, axis=1)
    # y_train1 = np.argmax(y_train, axis=1)
    # print(classification_report(y_train1, y_pred))
    # print(confusion_matrix(y_train1,y_pred))

    # preds = model.predict(x_test)
    # y_pred = np.argmax(preds, axis=1)
    # y_test1 = np.argmax(y_test, axis=1)
    # print(classification_report(y_test1, y_pred))
    # print(confusion_matrix(y_test1,y_pred))

def part_a():
    # Issue with timing, python time library only works when the difference is > 0.01s
    # Since our operation is so quick, we run into problems, however, if we assume that each pass
    # takes the same amout of time, then we can simply time the whole function and then
    # make each point in our output fx_history equally distant apart
    cost_function = func_one
    derivative = deriv_one
    l =  [5, 1]
    u = [10, 5]
    x0 = [11, 1]
    test_optimizations(cost_function, global_random_search, gradientDescent, plot_fx_history, derivative, l, u, x0, title="Function One Comparison")

    l =  [1, 1]
    u = [20, 5]
    x0 = [10, 1]
    cost_function = func_two
    derivative = deriv_two
    test_optimizations(cost_function, global_random_search, gradientDescent, plot_fx_history, derivative, l, u, x0, title="Function Two Comparison")

def part_b():
    #############################
    ###### Cost Function 1 ######
    #############################
    cost_function = func_one
    derivative = deriv_one
    l =  [5, 1]
    u = [10, 5]
    x0 = [11, 1]

    ###### Global Random Search #####
    start_time = time.time()
    x_hist, _ = global_random_search(cost_function, len(l), l, u, 100)
    function_duration = time.time() - start_time
    start_time = 0
    grs_x_history = [np.linspace(start_time, function_duration, num=len(x_hist)), x_hist]

    ###### Modified Global Random Search #####
    start_time = time.time()
    x_hist = modified_global_random_search(cost_function, len(l), l, u, 100, 10)
    function_duration = time.time() - start_time
    start_time = 0
    mgrs_x_history = [np.linspace(start_time, function_duration, num=len(x_hist)), x_hist]

    ###### Gradient Descent #####
    alpha = 0.005
    start_time = time.time()
    gd_x_history = gradientDescent(cost_function, derivative, x0, alpha, 100)
    function_duration = time.time() - start_time
    start_time = 0
    gd_x_history = [np.linspace(start_time, function_duration, num=len(gd_x_history)), gd_x_history]    

    plot_x_over_contour(cost_function, grs_x_history, mgrs_x_history, gd_x_history, "Algorithms applied to Function One over contour")

    # For the modified version, going to plot the highest x of each iteration

    #############################
    ###### Cost Function 2 ######
    #############################
    cost_function = func_one
    derivative = deriv_two
    l =  [1, 1]
    u = [20, 5]
    x0 = [10, 1]

    ###### Global Random Search #####
    start_time = time.time()
    x_hist, _ = global_random_search(cost_function, len(l), l, u, 100)
    function_duration = time.time() - start_time
    start_time = 0
    grs_x_history = [np.linspace(start_time, function_duration, num=len(x_hist)), x_hist]

    ###### Modified Global Random Search #####
    start_time = time.time()
    x_hist = modified_global_random_search(cost_function, len(l), l, u, 100, 10)
    function_duration = time.time() - start_time
    start_time = 0
    mgrs_x_history = [np.linspace(start_time, function_duration, num=len(x_hist)), x_hist]

    ###### Gradient Descent #####
    alpha = 0.005
    start_time = time.time()
    gd_x_history = gradientDescent(cost_function, derivative, x0, alpha, 100)
    function_duration = time.time() - start_time
    start_time = 0
    gd_x_history = [np.linspace(start_time, function_duration, num=len(gd_x_history)), gd_x_history]

    plot_x_over_contour(cost_function, grs_x_history, gd_x_history, mgrs_x_history, "Algorithms applied to Function Two over contour")

def part_c():
    # batch_size, alpha, beta_1, beta_2, epochs
    l = [5, 0.1, 0.01, 0.01, 5]
    u = [250, 10, 0.99, 0.99, 100]

    print(global_random_search(conv_net, len(l), l, u, 100))

    #conv_net(minibatch_size, alpha, beta_1, beta_2, epochs, opt)

#part_a()
#part_b()
part_c()