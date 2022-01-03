import matplotlib.pyplot as plt
import math
import numpy as np
import PIL.Image as Image
<<<<<<< HEAD
from sklearn import dummy
=======
>>>>>>> 49c746edbb6952823efae06a4f90ca5b4ec4c9da
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

def convolve(n, k):
    outputLen = len(n) - len(k) + 1

    convMatrix = []
    localMatrixSizeModifier = len(k)
    for row in range(outputLen):
        convRow = []
        for col in range(outputLen):
            localMatrix = [i[col : col + localMatrixSizeModifier] for i in n[row : row + localMatrixSizeModifier]]
            multMatrices = [[x * y for x, y in zip(a, b)] for a, b in zip(localMatrix, k)]
            flattenedMatrices = sum(multMatrices, [])
            convRow.append(sum(flattenedMatrices))
        convMatrix.append(convRow)
    return convMatrix

def printConvolved(c):
    for row in c:
        toPrint = '|'
        for elem in row:
            if elem != 0:
                newElem = ''
                logged = math.log10(abs(elem))
                if logged % 1 != 0:
                    spaces = 5 - (math.ceil(logged))
                else:
                    spaces = 4 - (math.ceil(logged))
                newElem = str(elem) + (' ' * spaces)
                if elem > 0:
                    newElem = ' ' + newElem
                toPrint += newElem
            else:
                toPrint += ' ' + str(elem) + (' ' * 4)
        print(toPrint + '|')

def modelCreation(x_train, num_classes, c=0.001, pool=False, deeper=False):
    model = keras.Sequential()
    if not deeper:
        if pool:
            model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
            model.add(Conv2D(16, kernel_size=(3,3,), padding="same", activation="relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
        else:
            model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
            model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
            model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(c)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.summary()
    else:
        model.add(Conv2D(8, (3,3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
        model.add(Conv2D(8, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), strides=(2,2),  padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation ='softmax', kernel_regularizer=regularizers.l1(0.0001)))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
    return model

def runModel(n=5000, c=0.001, pool=False, deeper=False, title=''):
    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # takes the first 5000 datapoints in x_train and y_train
    x_train = x_train[1:n]; y_train=y_train[1:n]
    #x_test=x_test[1:500]; y_test=y_test[1:500]

    # images are originaly in 0-255 values, regularize them to 0-1
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    # data currently saved as a vector with numbers 0-9, instead, turn each y into a list of binary numbers, with 9 elements in each list
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = modelCreation(x_train, num_classes, c, pool, deeper)

    batch_size = 128
    epochs = 20
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save("cifar.model")
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1,y_pred))

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1,y_pred))

<<<<<<< HEAD
def baseline():
    num_classes = 10
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # takes the first 5000 datapoints in x_train and y_train
    x_train = x_train[1:5000]; y_train=y_train[1:5000]

    # images are originaly in 0-255 values, regularize them to 0-1
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    model = dummy.DummyClassifier(strategy="most_frequent")
    model.fit(x_train, y_train)

    preds = model.predict(x_train)
    y_train = sum(y_train.tolist(), [])
    totalVals = 0
    correctVals = 0
    for v, i in enumerate(y_train):
        if v == preds[i]:
            correctVals += 1
        totalVals = i
    acc = correctVals / totalVals
    print(acc)

    preds = model.predict(x_test)
    y_test = sum(y_test.tolist(), [])
    totalVals = 0
    correctVals = 0
    for v, i in enumerate(y_test):
        if v == preds[i]:
            correctVals += 1
        totalVals = i
    acc = correctVals / totalVals
    print(acc)

=======
>>>>>>> 49c746edbb6952823efae06a4f90ca5b4ec4c9da
def part1():
    # (a)
    n = [[1, 2, 3, 4, 5], 
        [1, 3, 2, 3, 10], 
        [3, 2, 1, 4, 5], 
        [6, 1, 1, 2, 2], 
        [3, 2, 1, 5, 4]]
    k = [[1, 0, -1], 
        [1, 0, -1], 
        [1, 0, -1]]
    printConvolved(convolve(n, k))

    # (b)
<<<<<<< HEAD
    im = Image.open('shape.png')
    rgb = np.array(im.convert('RGB'))
    r = rgb[:,:,0]
    Image.fromarray(np.uint8(r)).show()

=======
    im = Image.open('square.jpg')
    rgb = np.array(im.convert('RGB'))
    r = rgb[:,:,0]
>>>>>>> 49c746edbb6952823efae06a4f90ca5b4ec4c9da
    kernel1 = [[-1, -1, -1], 
            [-1, 8, -1], 
            [-1, -1, -1]]
    kernel2 = [[0, -1, 0], 
            [-1, 8, -1], 
            [0, -1, 0]]
    con1 = convolve(r, kernel1)
<<<<<<< HEAD
    Image.fromarray(np.uint8(con1)).show()
    #printConvolved(con1)
    print("\n\n")
    con2 = convolve(r, kernel2)
    Image.fromarray(np.uint8(con2)).show()
    #printConvolved(con2)

def part2():
    # (b) (i)
    baseline()

=======
    printConvolved(con1)
    print("\n\n")
    con2 = convolve(r, kernel2)
    printConvolved(con2)

def part2():
>>>>>>> 49c746edbb6952823efae06a4f90ca5b4ec4c9da
    # (b) (iii)
    N = [5000, 10000, 20000, 40000]
    for n in N:
        runModel(n=n, title=f'Model trained with {n} datapoints')

    # (b) (iv)
    Ci = [0, 0.001, 0.1, 1, 10, 100]
    for c in Ci:
        runModel(c=c, title=f'Model with c value of {c}')

    # (c) (i)
    runModel(pool=True, title=f'Model trained with max pooling')

    # (d)
    runModel(deeper=True, n=40000, title=f'Thinner and deeper model')

part1()
<<<<<<< HEAD
#part2()

# TODO: 
#       (i) (b)
#
#       (ii) (c) (ii)
=======
part2()
>>>>>>> 49c746edbb6952823efae06a4f90ca5b4ec4c9da
