import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data.csv', comment='#')
X1 = data.iloc[:,0]
X2 = data.iloc[:,1]
X = np.column_stack((X1, X2))
Y = data.iloc[:,2]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
X1 = np.array([x[0] for x in Xtrain])
X2 = np.array([x[1] for x in Xtrain])
X = np.column_stack((X1, X2))
Y = np.array(Ytrain)

def plotOriginalData(ax=None):
    if ax == None: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X1, X2, Y, color="blue", label='Original Data')
    ax.set_xlabel('X1'); ax.set_ylabel('X2'); ax.set_zlabel('target')
    plt.title("Original Data")

def plotPredictions(model, features, X, ax, name):
    predictions = model.predict(features)
    ax.plot_trisurf(X[:, 0], X[:, 1], predictions, color='g')

def oneA():
    plotOriginalData()
    plt.show()

def oneB(lasso=True):
    # First get the extra polynomial features
    trans = PolynomialFeatures(degree=5)
    # If we just want the combinations and want to ignore the stand alone powers
    #trans = PolynomialFeatures(degree=5, interaction_only=True)
    data = trans.fit_transform(X)
    testData = trans.fit_transform(Xtest)

    models = []
    if lasso == True: Ci_range = [1, 5, 10, 50, 1000]
    else: Ci_range = [0.001, 0.01, 0.1, 1]
    for Ci in Ci_range:
        if lasso: model = Lasso(alpha=1/(2*Ci))
        else: model = Ridge(alpha=1/(2*Ci))
        bestModel = [Lasso(), 10000]
        kf = KFold(n_splits=5)
        for train, test in kf.split(data):
            model.fit(data[train], Y[train])
            ypred = model.predict(data[test])
            mean = mean_squared_error(Y[test], ypred)
            if mean < bestModel[1]:
                bestModel = [model, mean]
        model = bestModel[0]
        pred = model.predict(testData)
        print(f'For a C of {Ci}, I got an intercept of {model.intercept_}, for X1 {model.coef_[1]}, for X2 {model.coef_[2]}, for X1^2 {model.coef_[3]}, for X2^2 {model.coef_[4]}, for X1^3 {model.coef_[5]}, for X2^3 {model.coef_[6]}, for X1^4 {model.coef_[7]}, for X2^4 {model.coef_[8]}, for X1^5 {model.coef_[9]}, for X2^5 {model.coef_[10]} and for X1*X2 {model.coef_[11]}\n')
        print(f'It got an R2 score of {sk.metrics.r2_score(Ytest, pred)}\n')
        models.append([bestModel[0], Ci])

    return models

def oneC(models):
    # this simply makes a grid from -5 to 5, making a jump of .5 every time
    # the grid will have a combination of every possibel X1 and X2
    grid = np.mgrid[-5:5:21j, -5:5:21j].reshape(2, -1).T
    # This is the grid with polynomial features of degree 5
    Xtest = PolynomialFeatures(degree=5).fit_transform(grid)
    
    for m, name in models:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plotPredictions(m, Xtest, grid, ax, name)
        plotOriginalData(ax)
        plt.title(f"C = {name}")
        plt.legend()
        plt.show()

def twoA(lasso=True):
    trans = PolynomialFeatures(degree=5)
    data = trans.fit_transform(X)

    mean_error=[]; std_error=[]
    if lasso == True: Ci_range = [1, 5, 10, 50]
    else: Ci_range = [0.01, 0.1, 1]
    for Ci in Ci_range:
        if lasso: model = Lasso(alpha=1/(2*Ci))
        else: model = Ridge(alpha=1/(2*Ci))
        temp=[]
        kf = KFold(n_splits=5)
        for train, test in kf.split(data):
            model.fit(data[train], Y[train])
            ypred = model.predict(data[test])
            mean = mean_squared_error(Y[test], ypred)
            temp.append(mean)

        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel('Ci')
    plt.ylabel('Mean Square Error')
    if lasso == True: 
        plt.xlim((-10,60))
        plt.title('Lasso')
    else: 
        plt.xlim((-0.01, 1.01))
        plt.title('Ridge Regression')
    plt.show()

def part1():
    print('(i)')

    # (a)
    oneA()

    # (b)
    models = oneB()
    
    # (c)
    oneC(models)

    # (e)
    models = oneB(False)
    oneC(models)

def part2():
    print('(ii)')

    # (a)
    twoA()

    # (c)
    twoA(False)

part1()
part2()