# Note: in order to make the provided code work, I needed to add ,0,1 to the first line of
# the data given so that it would correctly identify it as a csv file

# csv first line # id:2--2--2,0,1 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from matplotlib.colors import ListedColormap

# takes in X1, X2 which have all values, regardless if y is 1 or -1
# returns two sets of 3 variables, holding the sets where y is 1 or y is -1
def seperator(df):
    X1pos, X2pos, Ypos = [], [], []
    X1neg, X2neg, Yneg = [], [], []

    for _, row in df.iterrows():
        if row['Y'] == 1:
            X1pos.append(row['X1'])
            X2pos.append(row['X2'])
            Ypos.append(row['Y'])
        else:
            X1neg.append(row['X1'])
            X2neg.append(row['X2'])
            Yneg.append(row['Y'])

    pos = {'X1': X1pos, 'X2': X2pos, 'Y': Ypos}
    neg = {'X1': X1neg, 'X2': X2neg, 'Y': Yneg}
    posData = pd.DataFrame(pos)
    negData = pd.DataFrame(neg)
    return posData, negData

df = pd.read_csv("./data.csv")
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y = df.iloc[:, 2]

pos, neg = seperator(df)

new_set1 = cm.get_cmap('Set1', 2)

hsv_modified = cm.get_cmap('hsv')
newcmp = ListedColormap(hsv_modified(np.linspace(0.3,0.7,256)))

inferno_modified = cm.get_cmap('inferno')
predCmp = ListedColormap(inferno_modified(np.linspace(0.5,0.8,256)))

def plotOriginalData():
    plt.scatter(data=pos, x='X1', y='X2', s=5, c='#00ff00', label='train +1')
    plt.scatter(data=neg, x='X1', y='X2', s=5, c='#0000ff', label='train -1')
    plt.xlabel('X1')
    plt.ylabel('X2')

def plotAgainstOriginal(prediction, decisionBoundary=None):
    plotOriginalData()
    pred = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': prediction})
    newPos = {'X1': [], 'X2': []}
    newNeg = {'X1': [], 'X2': []}
    for index, row in pred.iterrows():
        if y[index] != row['Y'] and y[index] == -1:
            newPos['X1'].append(row['X1'])
            newPos['X2'].append(row['X2'])
        elif y[index] != row['Y'] and y[index] == 1:
            newNeg['X1'].append(row['X1'])
            newNeg['X2'].append(row['X2'])

    plt.scatter(data=newPos, x='X1', y='X2', s=5, c='#a37200', label='train +1')
    plt.scatter(data=newNeg, x='X1', y='X2', s=5, c='#0e1e33', label='train -1')

    if decisionBoundary:
        x = np.linspace(-1.1, 1.1, 100)
        plt.plot(x, decisionBoundary(x), c='#ff0000', linewidth=.8, label='decision boundary')
    plt.legend()

def plotFinal(prediction, decisionBoundary):
    pred = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': prediction})
    predPos, predNeg = seperator(pred)
    
    plt.scatter(data=predPos, x='X1', y='X2', s=5, c='#00ff00', label='pred +1')
    plt.scatter(data=predNeg, x='X1', y='X2', s=5, c='#0000ff', label='pred -1')
    x = np.linspace(-1.1, 1.1, 100)
    plt.plot(x, decisionBoundary(x), c='#ff0000', linewidth=.8, label='decision boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()

# (a)
def questionA():
    print("(a)")

    # (i)
    plotOriginalData()
    plt.legend(['train +1', 'train -1'])
    plt.title('Original Data')
    plt.show()

    # (ii)

    ##### modeling
    bestModel = [LogisticRegression(), 10000]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = LogisticRegression(penalty='none', solver='lbfgs').fit(X[train], y[train])
        ypred = model.predict(X[test])
        squareError = mean_squared_error(y[test], ypred)
        print("intercept %f, slope X1 %f, slope X2 %f, square error %f"%(model.intercept_, model.coef_[0][0], model.coef_[0][1], squareError))
        if bestModel[1] > squareError:
            bestModel[0], bestModel[1] = model, squareError
    model = bestModel[0]

    # (iii)

    ypred = model.predict(X)
    m1, m2 = model.coef_[0]
    c = model.intercept_
    decisionBoundary = lambda x: ((m1 * x) + c) / -(m2)

    plotAgainstOriginal(ypred, decisionBoundary)

    plt.margins(x=0)
    plt.title('Basic Logistic Regression Against Original Data')
    plt.show()

    # (iv)

# (b)
def questionB():
    print("\n(b)\n")

    # (i)
    mean_error=[]; std_error=[]
    meanErrorOfPredictions = []

    smallCBest = [LinearSVC(), 10000]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = LinearSVC(C=0.001).fit(X[train], y[train])
        ypred = model.predict(X[test])
        squareError = mean_squared_error(y[test], ypred)
        if smallCBest[1] > squareError:
            smallCBest[0], smallCBest[1] = model, squareError
        meanErrorOfPredictions.append(squareError)
    mean_error.append(np.array(meanErrorOfPredictions).mean()); 
    std_error.append(np.array(meanErrorOfPredictions).std()); meanErrorOfPredictions = []

    mediumCBest = [LinearSVC(), 10000]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = LinearSVC(C=1).fit(X[train], y[train])
        ypred = model.predict(X[test])
        squareError = mean_squared_error(y[test], ypred)
        if mediumCBest[1] > squareError:
            mediumCBest[0], mediumCBest[1] = model, squareError
        meanErrorOfPredictions.append(squareError)
    mean_error.append(np.array(meanErrorOfPredictions).mean()); 
    std_error.append(np.array(meanErrorOfPredictions).std()); meanErrorOfPredictions = []

    largeCBest = [LinearSVC(), 10000]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = LinearSVC(C=100).fit(X[train], y[train])
        ypred = model.predict(X[test])
        squareError = mean_squared_error(y[test], ypred)
        if largeCBest[1] > squareError:
            largeCBest[0], largeCBest[1] = model, squareError
        meanErrorOfPredictions.append(squareError)
    mean_error.append(np.array(meanErrorOfPredictions).mean())
    std_error.append(np.array(meanErrorOfPredictions).std())

    # (ii)

    smallCModel = smallCBest[0]
    mediumCModel = mediumCBest[0]
    largeCModel = largeCBest[0]
    ypredSmall = smallCModel.predict(X)
    ypredMedium = mediumCModel.predict(X)
    ypredLarge = largeCModel.predict(X)

    print(f'\nFor C=0.001, the intercept is {smallCModel.intercept_[0]}, the slope of X1 is {smallCModel.coef_[0][0]} and the slope of X2 is {smallCModel.coef_[0][1]}')
    print(f'For C=1, the intercept is {mediumCModel.intercept_[0]}, the slope of X1 is {mediumCModel.coef_[0][0]} and the slope of X2 is {mediumCModel.coef_[0][1]}')
    print(f'For C=100, the intercept is {largeCModel.intercept_[0]}, the slope of X1 is {largeCModel.coef_[0][0]} and the slope of X2 is {largeCModel.coef_[0][1]}')

    decisionBoundary = lambda x: ((smallCModel.coef_[0][0] * x) + smallCModel.intercept_) / -(smallCModel.coef_[0][1])
    plotAgainstOriginal(ypredSmall, decisionBoundary)
    plt.title('Prediction With C=0.001 Against Original Data')
    plt.show()

    decisionBoundary = lambda x: ((mediumCModel.coef_[0][0] * x) + mediumCModel.intercept_) / -(mediumCModel.coef_[0][1])
    plotAgainstOriginal(ypredMedium, decisionBoundary)
    plt.title('Prediction With C=1 Against Original Data')
    plt.show()

    decisionBoundary = lambda x: ((largeCModel.coef_[0][0] * x) + largeCModel.intercept_) / -(largeCModel.coef_[0][1])
    plotAgainstOriginal(ypredLarge, decisionBoundary)
    plt.title('Prediction With C=100 Against Original Data')
    plt.show()

    # (iii)

    Ci_range = [0.001, 1, 100]
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel('Ci'); plt.ylabel('Mean square error')
    plt.xlim((-5, 110))
    plt.title('The Mean Square Error for the Different Values of C')
    plt.show()

    # (iv)

# (c)
def questionC():
    print("\n(c)\n")

    # (i)
    XSquared = X * X
    Xnew = np.c_[X, XSquared]

    bestModel = [LogisticRegression(), 10000]
    kf = KFold(n_splits=5)
    for train, test in kf.split(Xnew):
        model = LogisticRegression(penalty='none', solver='lbfgs').fit(Xnew[train], y[train])
        ypred = model.predict(Xnew[test])
        squareError = mean_squared_error(y[test], ypred)
        print("intercept %f, slope X1 %f, slope X2 %f, slope X3 %f, slope X4 %f, square error %f"%(model.intercept_, model.coef_[0][0], model.coef_[0][1], model.coef_[0][2], model.coef_[0][3], squareError))
        if bestModel[1] > squareError:
            bestModel[0], bestModel[1] = model, squareError
    model = bestModel[0]

    # (ii)
    ypred = model.predict(Xnew)

    plotAgainstOriginal(ypred)
    plt.title('Logistic Regression With Feature Engineering Against Original Data')
    plt.show()

    # (iii) baseline predictor, one that predicts most common class simply predicts 1 or -1 everytime depending on which is more common
    mostCommonClass = 0
    sum = 0
    for i in y:
        sum = sum + i
    if sum < 0:
        mostCommonClass = -1
    elif sum > 0:
        mostCommonClass = 1
    else:
        print("More complex baseline required")

    # Get fp, fn, tp and tn's for both predictions
    fpPred, fnPred, tpPred, tnPred = 0, 0, 0, 0
    fpBase, fnBase, tpBase, tnBase = 0, 0, 0, 0
    for i in range(len(y)):
        if (y[i] == ypred[i]) & (y[i] == -1):
            tnPred = tnPred + 1
        elif (y[i] == ypred[i]) & (y[i] == 1):
            tpPred = tpPred + 1
        elif (y[i] != ypred[i]) & y[i] == 1:
            fnPred = fnPred + 1
        else:
            fpPred = fpPred + 1

        if (y[i] == mostCommonClass):
            tnBase = tnBase + 1
        elif (y[i] != mostCommonClass):
            fnBase = fnBase + 1

    print("%d %d %d %d" % (tpPred, tnPred, fpPred, fnPred))
    print("%d %d %d %d" % (tpBase, tnBase, fpBase, fnBase))
    acc = (tnPred + tpPred) / (tnPred + tpPred + fnPred + fpPred)
    accBase = (tnBase + tpBase) / (tnBase + tpBase + fnBase + fpBase)
    print(acc, accBase)

    tpRate = (tpPred) / (tpPred + fnPred)
    tpRateBase = (tpBase) / (tpBase + fnBase)
    print(tpRate, tpRateBase)

    fpRate = (fpPred) / (tnPred + fpPred)
    fpRateBase = (fpBase) / (tnPred + fpPred)
    print(fpRate, fpRateBase)

    precision = (tpPred) / (tpPred + fpPred)
    print(precision)

    accDiff = ((tnPred + tpPred) / (tnPred + tpPred + fnPred + fpPred)) - ((tnBase + tpBase) / (tnBase + tpBase + fnBase + fpBase))
    tpRateDiff = ((tpPred) / (tpPred + fnPred)) - ((tpBase) / (tpBase + fnBase))
    fpRateDiff = ((fpPred) / (tnPred + fpPred)) - ((fpBase) / (tnPred + fpPred))
    precisionDiff = ((tpPred) / (tpPred + fpPred))

    print("%f %f %f %f" % (accDiff, tpRateDiff, fpRateDiff, precisionDiff))

    # (iv)
    m0, m1, m2, m3 = model.coef_[0]
    c = model.intercept_

    # determinant, where c = O3x^2 + O1x + O0
    boundary_f = lambda x: (-m1 - pow(m1*m1 - 4*m3*(m2*x*x + m0*x + c), 0.5)) / 2*m3
    plotFinal(ypred, boundary_f)
    plt.title('Feature Engineering Against Original Data With Decision Boundary')
    plt.show()

questionA()
questionB()
questionC()