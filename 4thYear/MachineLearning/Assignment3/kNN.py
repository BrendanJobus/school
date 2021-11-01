# id: # id:14--14-14-0 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

dOne = []
dTwo = []

def extractData():
    dataOne = {'X1': [], 'X2': [], 'Y': []}
    dataTwo = {'X1': [], 'X2': [], 'Y': []}
    data = dataOne
    with open("./data.csv", 'r') as f:
        dataset = 0
        for line in f:
            if '#' in line and dataset == 0:
                dataset = 1
            elif '#' in line and dataset == 1:
                dataset = 2
                data = dataTwo
            else:
                words = line.split(sep=',')
                data['X1'].append(float(words[0]))
                data['X2'].append(float(words[1]))
                data['Y'].append(float(words[2].replace('\n','')))
    return pd.DataFrame(dataOne), pd.DataFrame(dataTwo)

def choosePoly(X, Y, c):
    model = LogisticRegression(penalty="l2", C=c)
    kf = KFold(n_splits=5)
    accuracy, mean_error, std_error = [], [], []
    featureLength = range(1, 6)
    for x in featureLength:
        trans = PolynomialFeatures(degree=x)
        data = trans.fit_transform(X)
        acc = []
        scores = []
        for train, test in kf.split(data):
            model.fit(data[train], Y[train])
            ypred = model.predict(data[test])
            acc.append(accuracy_score(Y[test], ypred))
            score = cross_val_score(model, data[test], Y[test], cv=5, scoring='roc_auc')
            scores.append(score)
        accuracy.append(max(acc))
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    plt.errorbar(featureLength, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel('Polynomial'); plt.ylabel('AUC')
    plt.title('AUC of different polynomials')
    plt.show()

    # After looking through the data, I have decided to use a polynomial of degree 2
    return 2
    
def chooseC(X, Y, Ci_range, poly):
    trans = PolynomialFeatures(degree=poly)
    data = trans.fit_transform(X)
    kf = KFold(n_splits=5)
    accuracy, mean_error, std_error = [], [], []
    for Ci in Ci_range:
        acc = []
        scores = []
        model = LogisticRegression(penalty='l2', C=Ci)
        for train, test in kf.split(data):
            model.fit(data[train], Y[train])
            ypred = model.predict(data[test])
            acc.append(accuracy_score(Y[test], ypred))
            score = cross_val_score(model, data[test], Y[test], scoring='roc_auc')
            scores.append(score)
        accuracy.append(max(acc))
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel('Ci'); plt.ylabel('AUC')
    plt.title('AUC of different c values')
    plt.show()

    # By analysing the results, I've deemed 8 to be the best c value, returning it like this so its a little more
    # explicit as to how I get the value
    return 8

def a(X, Y):
    Ci_range = [0.1, 0.5, 1, 5, 10, 50]
    c = Ci_range[2]
    poly = choosePoly(X, Y, c)

    c = chooseC(X, Y, Ci_range, poly)

    return LogisticRegression(penalty='l2', C=c)
    
def b(X, Y):
    kNeighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    kf = KFold(n_splits=5)
    accuracy, mean_error, std_error = [], [], []
    for k in kNeighbors:
        acc, scores = [], []
        for train, test in kf.split(X):
            model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X[train], Y[train])
            ypred = model.predict(X[test])
            acc.append(accuracy_score(Y[test], ypred))
            score = cross_val_score(model, X[test], Y[test], scoring='roc_auc')
            scores.append(score)
        accuracy.append(max(acc))
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    plt.errorbar(kNeighbors, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel('k'); plt.ylabel('AUC')
    plt.title('AUC of different k values')
    plt.show()

    return KNeighborsClassifier(n_neighbors=8, weights='uniform')

def c(X, Y, Xtest, Ytest, logistic, kNN, baseline):
    # here we're gonna want to use the data that we preserved
    model = logistic.fit(X, Y)
    ypred = model.predict(Xtest)
    cnf_matrix = confusion_matrix(Ytest, ypred)
    print(cnf_matrix)

    model = kNN.fit(X, Y)
    ypred = model.predict(Xtest)
    cnf_matrix = confusion_matrix(Ytest, ypred)
    print(cnf_matrix)

    model = baseline.fit(X, Y)
    ypred = model.predict(Xtest)
    cnf_matrix = confusion_matrix(Ytest, ypred)
    print(cnf_matrix)

def d(X, Y, Xtest, Ytest, logistic, kNN, baseline):
    model = baseline.fit(X, Y)
    ypred = model.predict(Xtest)
    RocCurveDisplay.from_predictions(Ytest, ypred, name='Baseline Most Frequent Classifier')

    model = logistic.fit(X, Y)
    fpr, tpr, _ = roc_curve(Ytest, model.decision_function(Xtest))
    plt.plot(fpr, tpr, label='logistic')

    model = kNN.fit(X, Y)
    y_scores = model.predict_proba(Xtest)
    fpr, tpr, _ = roc_curve(Ytest, y_scores[:, 1])
    plt.plot(fpr, tpr, label='kNN')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0,1], [0,1], color='green', linestyle='--', label='Random Classifier')
    plt.legend()
    plt.show()

def part1(part=1):
    if part == 1:
        print("(i)\n")
        X1 = dOne.iloc[:, 0]
        X2 = dOne.iloc[:, 1]
        X = np.column_stack((X1, X2))
        Y = dOne.iloc[:, 2]
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
        X1 = np.array([x[0] for x in Xtrain])
        X2 = np.array([x[1] for x in Xtrain])
        X = np.column_stack((X1, X2))
        Y = np.array(Ytrain)
    else:
        print("(ii)\n")
        X1 = dOne.iloc[:, 0]
        X2 = dOne.iloc[:, 1]
        X = np.column_stack((X1, X2))
        Y = dOne.iloc[:, 2]
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
        X1 = np.array([x[0] for x in Xtrain])
        X2 = np.array([x[1] for x in Xtrain])
        X = np.column_stack((X1, X2))
        Y = np.array(Ytrain)

    logistic = a(X, Y)
    kNN = b(X, Y)
    baseline = DummyClassifier(strategy='most_frequent')
    c(X, Y, Xtest, Ytest, logistic, kNN, baseline)
    d(X, Y, Xtest, Ytest, logistic, kNN, baseline)
    # (e) is typing

dOne, dTwo = extractData()
part1()
part1(2)