<<<<<<< HEAD
from itertools import groupby
import math
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True

def euclidAbsDistance(baseLat, baseLong, xLat, xLong):
    return abs(math.sqrt((baseLat - xLat) ** 2 + (baseLong - xLong) ** 2))

def findLargestContinuoumOfDates(data):
    stationsSeperated = []
    allStationCodes = data.drop_duplicates(subset=['STATION ID'])['STATION ID']
    for station in allStationCodes:
        stationsSeperated.append(data.loc[data['STATION ID'] == station])
    
    # now find which ones have full datasets for the desired range of 2020-01-01 to 2020-02-24

    # to see if we go from 01 01 to 02 02, were going to do (month - 1) * 31 + day, and if there are no gaps, then we can say that this station has all the data
    # should go from 2 to 55
    # apply new function to each station and then remove the duplicates
    unusableStations = []
    for station in stationsSeperated:
        codes = station['TIME'].map(dateToInt)
        data = codes.drop_duplicates()
        
        idx = max(
            (
                list(map(itemgetter(0), g))
                for i, g in groupby(enumerate(np.diff(data) == 1), itemgetter(1))
                if i
            ),
            key = len
        )
        #print(data.iloc[idx[0]], data.iloc[idx[-1]+1])
        # this give us 27 - 60 as the largest continuous range for most
        # remove the stations that don't have this range
        if (not (data.iloc[idx[0]] == 27)) | (not (data.iloc[idx[-1]+1] == 60)):
            unusableStations.append(station['STATION ID'].iloc[0])
    return unusableStations

def dateToInt(date):
    month = int(date[5: 7 : 1]); day = int(date[8 : 10 : 1])
    return (month - 1) * 31 + day

def reduceStations():
    data = pd.read_csv('./data/dublinbikes_20200101_20200401.csv')
    copy = data.copy()

    # largest continuous range of dates = January 27th to February 29th
    unusableStations = findLargestContinuoumOfDates(copy)

    allGeocodes = data.copy()
    allGeocodes = allGeocodes.drop_duplicates(subset=['STATION ID'])
    for station in unusableStations:
        allGeocodes = allGeocodes[allGeocodes['STATION ID'] != station]
    allGeocodes = allGeocodes[ ['STATION ID', 'LATITUDE', 'LONGITUDE'] ]
    base = allGeocodes.loc[allGeocodes['STATION ID'] == 2]
    baseLat = base['LATITUDE'][0]; baseLong = base['LONGITUDE'][0]
    allGeocodes = allGeocodes.drop(index=0)
    dists = allGeocodes.apply(lambda x : euclidAbsDistance(baseLat, baseLong, x['LATITUDE'], x['LONGITUDE']), axis=1)
    furthestStation = data.iloc[dists.idxmax()]['STATION ID']
    reducedData = data.loc[(data['STATION ID'] == 2) | (data['STATION ID'] == furthestStation)].copy()

    # Now just reduce the dates so that they are between January 27th and February 29th
    codes = reducedData['TIME'].map(dateToInt)
    print(codes)
    codes = [True if (x >= 27) & (x <= 60) else False for x in codes]
    reducedData = reducedData[codes]

    ## before we write we need to add in a column for bikes in station
    maxBikes = reducedData['BIKE STANDS']
    availableSpots = reducedData['AVAILABLE BIKE STANDS']
    currentOccupancy = maxBikes - availableSpots
    reducedData['CURRENT OCCUPANCY'] = currentOccupancy
    reducedData.to_csv('./data/bike_data.csv')

def transformData(data, station):
    df = data.loc[(data['NAME'] == station)].copy()
    t = pd.array(pd.DatetimeIndex(df.iloc[:, 2]).view(np.int64)) / 1000000000
    lastUpdated = pd.array(pd.DatetimeIndex(df.iloc[:, 3]).view(np.int64)) / 1000000000
    dt = t[1] - t[0]
    print('data sampling interval is %d secs' %dt)

    df['TIME'] = t; df['LAST UPDATED'] = lastUpdated
    df = df.sort_values(by=['TIME'])
    t = (t - t[0]) / 60 / 60 / 24
    y = np.array(df.loc[:, 'CURRENT OCCUPANCY'].view(np.int64))

    # Turn STATUS into binary: 1 = Open, 0 = Closed
    status = pd.array(df['STATUS'])
    status = [1 if x == 'Open' else 0 for x in status]
    df['STATUS'] = status

    # need to add t back onto the features
    df = df.drop(['CURRENT OCCUPANCY'], axis=1); df = df.drop(['NAME'], axis=1); df = df.drop(['ADDRESS'], axis=1)
    df['TIME IN DAYS'] = t

    return df, y, dt

def plotData(x, y, plot):
    if plot:
        t = x.iloc[:, 12]
        plt.scatter(t, y, color='red', marker='.'); plt.title(f'data'); plt.show()

def plotResults(x, y, xlabel, ylabel, title, axis):
    axis.scatter(x=x, y=y); axis.set(xlabel=xlabel, ylabel=ylabel); axis.set_title(title)

def featurePicker(x, y, modelType):
    plt.rc('font', size=12)
    if modelType == 'Ridge':
        columns = ['Unnamed: 0', 'STATION ID', 'TIME', 'LAST UPDATED', 'BIKE STANDS', 'AVAILABLE BIKE STANDS', 'AVAILABLE BIKES', 'STATUS', 'LATITUDE', 'LONGITUDE', 'TIME IN DAYS']
        r2s = []
        rmses = []
        for c in columns:
            tmpX = x.copy()
            tmpX = tmpX.drop(c, axis=1)
            print(f'\nWithout {c}:')
            r2, rmse = ridgePredictor(1, 1, 3, tmpX, y)
            r2s.append(r2); rmses.append(rmse)
        fig, (ax1, ax2) = plt.subplots(2)
        plotResults(columns, r2s, 'Column Removed', 'R squared', f'R squared change with different column combination', ax1)
        plotResults(columns, rmses, 'Column Removed', 'RMSE', f'RMSE change with different column combination', ax2)

        # drop the elemnts we don't want here
        x = x.drop(['Unnamed: 0', 'STATION ID', 'BIKE STANDS', 'STATUS', 'LONGITUDE', 'LATITUDE'], axis=1)

        columns = ['TIME', 'LAST UPDATED', 'AVAILABLE BIKE STANDS', 'AVAILABLE BIKES', 'TIME IN DAYS']
        r2s = []
        rmses = []
        for c in columns:
            tmpX = x.copy()
            tmpX = tmpX.drop(c, axis=1)
            print(f'\nWithout {c}:')
            r2, rmse = ridgePredictor(1, 1, 3, tmpX, y)
            r2s.append(r2); rmses.append(rmse)
        fig, (ax1, ax2) = plt.subplots(2)
        plotResults(columns, r2s, 'Column Removed', 'R squared', f'R squared change with different column combination', ax1)
        plotResults(columns, rmses, 'Column Removed', 'RMSE', f'RMSE change with different column combination', ax2)
        fig.show()

        testX = x.drop(['AVAILABLE BIKE STANDS', 'AVAILABLE BIKES'], axis=1)

        columns = ['TIME', 'LAST UPDATED', 'TIME IN DAYS']
        r2s = []
        rmses = []
        for c in columns:
            tmpX = testX.copy()
            tmpX = tmpX.drop(c, axis=1)
            print(f'\nWithout {c}:')
            r2, rmse = ridgePredictor(1, 1, 3, tmpX, y)
            r2s.append(r2); rmses.append(rmse)
        fig, (ax1, ax2) = plt.subplots(2)
        plotResults(columns, r2s, 'Column Removed', 'R squared', f'R squared change with different column combination', ax1)
        plotResults(columns, rmses, 'Column Removed', 'RMSE', f'RMSE change with different column combination', ax2)
        fig.show()

        testX = x.drop(['TIME', 'LAST UPDATED', 'TIME IN DAYS'], axis=1)

        columns = ['AVAILABLE BIKE STANDS', 'AVAILABLE BIKES']
        r2s = []
        rmses = []
        for c in columns:
            tmpX = testX.copy()
            tmpX = tmpX.drop(c, axis=1)
            print(f'\nWithout {c}:')
            r2, rmse = ridgePredictor(1, 1, 3, tmpX, y)
            r2s.append(r2); rmses.append(rmse)
        _, (ax1, ax2) = plt.subplots(2)
        plotResults(columns, r2s, 'Column Removed', 'R squared', f'R squared change with different column combination', ax1)
        plotResults(columns, rmses, 'Column Removed', 'RMSE', f'RMSE change with different column combination', ax2)
        plt.show()

        x = x.drop(['TIME', 'LAST UPDATED', 'TIME IN DAYS', 'AVAILABLE BIKE STANDS'], axis=1)

        # test against no base features
        testX = x.drop(['AVAILABLE BIKES'], axis=1)
        r2, rmse = ridgePredictor(1, 1, 3, x, y)
        print(f'\nTest with our best features: {r2}, {rmse}')
        r2, rmse = ridgePredictor(1, 1, 3, testX, y)
        print(f'\nTest with no features: {r2}, {rmse}')
    else:
        testX = x.copy()
        testX = testX.drop(['Unnamed: 0', 'STATION ID', 'BIKE STANDS', 'STATUS', 'LONGITUDE', 'LATITUDE'], axis=1)
        print('\nRemoving the columns that we found to be useless with the Ridge regressor: ', forestPredictor(1, 1, 3, 5, testX, y))

        testX = x.copy()
        testX = testX.drop(['Unnamed: 0', 'STATION ID', 'TIME', 'LAST UPDATED', 'BIKE STANDS', 'AVAILABLE BIKE STANDS', 'STATUS', 'LATITUDE', 'LONGITUDE', 'TIME IN DAYS'], axis=1)
        print('\nUsing just the columns that were best for the Ridge regressor: ', forestPredictor(1, 1, 3, 5, testX, y))

        testX = x.copy()
        testX = testX.drop(testX.columns, axis=1)
        print('\nTesting with just the time series data: ', forestPredictor(1, 1, 3, 5, testX, y))

        x = x.drop(['Unnamed: 0', 'STATION ID', 'TIME', 'LAST UPDATED', 'BIKE STANDS', 'AVAILABLE BIKE STANDS', 'STATUS', 'LATITUDE', 'LONGITUDE', 'TIME IN DAYS'], axis=1)
    plt.rc('font', size=18)
    return x

def hyperParameterPicker(x, y, q, modelType):
    if modelType == 'Ridge':
        r2s = []; rmses = []
        lag = [1, 2, 3, 4, 5, 6]
        for e in lag:
            tmpX = x.copy()
            print(f'\nUsing {e} past times: ')
            r2, rmse = ridgePredictor(q, 1, e, tmpX, y)
            r2s.append(r2), rmses.append(rmse)
        fig, (ax1, ax2) = plt.subplots(2)
        plotResults(lag, r2s, 'Lag', 'R squared', f'R squared with varying lag on {modelType} looking {q} steps ahead', ax1)
        plotResults(lag, rmses, 'Lag', 'RMSE', f'RMSE with varying lag on {modelType} looking {q} steps ahead', ax2)
        fig.show()

        r2s = []
        rmses = []
        ddSetting = [1, 2, 3, 4, 5, 6]
        for dd in ddSetting:
            tmpX = x.copy()
            print(f'\nUsing a dd of {dd}: ')
            r2, rmse = ridgePredictor(q, dd, 1, tmpX, y)
            r2s.append(r2); rmses.append(rmse)
        fig, (ax1, ax2) = plt.subplots(2)
        plotResults(ddSetting, r2s, 'dd', 'R squared', f'R squared with varying immediate trend on {modelType} looking {q} steps ahead', ax1)
        plotResults(ddSetting, rmses, 'dd', 'RMSE', f'RMSE with varying immediate trend on {modelType} looking {q} steps ahead', ax2)
        plt.show()
    else:
        r2s = []; rmses = []
        lag = [1, 2, 3, 4, 5, 6]
        for e in lag:
            tmpX = x.copy()
            r2, rmse = forestPredictor(q, 1, e, 5, tmpX, y)
            r2s.append(r2); rmses.append(rmse)
        fig, (ax1, ax2) = plt.subplots(2)
        plotResults(lag, r2s, 'Lag', 'R squared', f'R squared with varying lag on {modelType} looking {q} steps ahead', ax1)
        plotResults(lag, rmses, 'Lag', 'RMSE', f'RMSE with varying lag on {modelType} looking {q} steps ahead', ax2)
        fig.show()

        r2s = []; rmses = []
        estimators = [1, 5, 10, 50, 100]
        for e in estimators:
            tmpX = x.copy()
            r2, rmse = forestPredictor(q, 1, 3, e, tmpX, y)
            r2s.append(r2); rmses.append(rmse)
        fig, (ax1, ax2) = plt.subplots(2)
        plotResults(estimators, r2s, 'Estimators', 'R squared', f'R squared with varying estimators used for the forest for {q} steps ahead', ax1)
        plotResults(estimators, rmses, 'Estimators', 'RMSE', f'RMSE with varying estimators used for the forest for {q} steps ahead', ax2)
        plt.show()

# Parameters:
#   q: how far into the future we are attempting to predict
#   dd: no fucking clue
#   lag: the number of previous events to take into account when using previous data
#   x: the features
#   y: targets
# Returns:
#   mse
def ridgePredictor(q, dd, lag, x, y, stride = 1):
    x = x.iloc[(lag - 1) * dd : y.size - q - 1 * dd : stride].copy()
    x['BIKE HISTORY 1'] = y[0 : y.size - q - lag * dd : stride]

    for  i in range(1, lag):
        columnName = 'BIKE HISTORY ' + str(i + 1)
        x[columnName] = y[i * dd : y.size - q - (lag - i) * dd : stride]

    y = y[lag * dd + q :: stride]
    scaled_features = StandardScaler().fit_transform(x.values)
    scaledX = pd.DataFrame(scaled_features, index = x.index, columns=x.columns)

    # do cross validation from here, get the mean score and coefficients and then report them at the end
    tscv = TimeSeriesSplit()
    featureTotals = [0] * x.shape[1]
    r2Total = 0
    rmseTotal = 0
    for train, test in tscv.split(x, y):
        model = Ridge(fit_intercept=False).fit(scaledX.iloc[train], y[train])
        featureTotals += model.coef_
        y_pred = model.predict(scaledX.iloc[test])
        r2Total += r2_score(y[test], y_pred)
        rmseTotal += np.sqrt(mean_squared_error(y[test], y_pred))
    featureAvg = featureTotals / ([5] * x.shape[1])
    for i, c in enumerate(scaledX.columns):
        if c == 'Unnamed: 0':
            print(f'Index {featureAvg[i]}')
        else:
            print(f'{c} {featureAvg[i]}')
    return r2Total/5, rmseTotal/5

def dailyAndImmediatePredictor(q, lag, x, y, d, modelType = 'Ridge', estimators = 0, stride=1):
    lngth = y.size - d - lag * d - q

    #x = x.iloc[lag * d + d : lag * d + d + lngth : stride].copy()
    x = x.iloc[(lag - 1) * d : (lag - 1) * d + lngth : stride].copy()

    x['BIKE HISTORY DAY 1'] = y[q:q+lngth:stride]

    for i in range(1, lag):
        columnName = 'BIKE HISTORY ' + str(i + 1)
        x[columnName] = y[i * d + q : i * d + q + lngth : stride]

    for i in range(0, lag):
        columnName = 'BIKE HISTORY DAY' + str(i + 1)
        x[columnName] = y[i : i + lngth : stride]

    print(x)

    y = y[lag * d + d + q : lag * d + d + q + lngth : stride]

    scaled_features = StandardScaler().fit_transform(x.values)
    scaledX = pd.DataFrame(scaled_features, index = x.index, columns=x.columns)

    tscv = TimeSeriesSplit()
    featureTotals = [0] * x.shape[1]
    r2s = 0; rmses = 0

    if modelType == 'Ridge':
        for train, test in tscv.split(x, y):
            model = Ridge(fit_intercept=False).fit(scaledX.iloc[train], y[train])
            featureTotals += model.coef_
            y_pred = model.predict(scaledX.iloc[test])
            r2s += r2_score(y[test], y_pred)
            rmses += np.sqrt(mean_squared_error(y[test], y_pred))
        featureAvg = featureTotals / ([5] * x.shape[1])
        for i, c in enumerate(scaledX.columns):
            if c == 'Unnamed: 0':
                print(f'Index {featureAvg[i]}')
            else:
                print(f'{c} {featureAvg[i]}')
    else:
        for train, test in tscv.split(x, y):
            model = RandomForestRegressor(n_estimators=estimators, random_state=42).fit(scaledX.iloc[train], y[train])
            y_pred = model.predict(scaledX.iloc[test])
            r2s += r2_score(y[test], y_pred)
            rmses += np.sqrt(mean_squared_error(y[test], y_pred))
    return r2s / 5, rmses / 5

def fullSeasonalityPredictor(q, lag, x, y, d, w, modelType = 'Ridge', estimators = 0, stride=1):
    lngth = y.size - w - lag * w - q

    #x = x.iloc[lag * w + w : lag * w + w + lngth : stride].copy()
    x = x.iloc[(lag - 1) * w : (lag - 1) * w + lngth : stride].copy()

    x['BIKE HISTORY WEEK 1'] = y[q:q+lngth:stride]

    for i in range(1, lag):
        columnName = 'BIKE HISTORY WEEK ' + str(i + 1)
        x[columnName] = y[i * w + q : i * w + q + lngth : stride]

    for  i in range(0, lag):
        columnName = 'BIKE HISTORY DAY ' + str(i + 1)
        x[columnName] = y[i * d + q : i * d + q + lngth : stride]

    for i in range(0, lag):
        columnName = 'BIKE HISTORY ' + str(i + 1)
        x[columnName] = y[i : i + lngth : stride]

    y = y[lag * w + w + q : lag * w + w + q + lngth : stride]

    scaled_features = StandardScaler().fit_transform(x.values)
    scaledX = pd.DataFrame(scaled_features, index = x.index, columns=x.columns)

    # do cross validation from here, get the mean score and coefficients and then report them at the end
    tscv = TimeSeriesSplit()
    featureTotals = [0] * x.shape[1]
    r2s = 0; rmses = 0
    if modelType == 'Ridge':
        for train, test in tscv.split(x, y):
            model = Ridge(fit_intercept=False).fit(scaledX.iloc[train], y[train])
            featureTotals += model.coef_
            y_pred = model.predict(scaledX.iloc[test])
            r2s += r2_score(y[test], y_pred)
            rmses += np.sqrt(mean_squared_error(y[test], y_pred))
            featureAvg = featureTotals / ([5] * x.shape[1])
        for i, c in enumerate(scaledX.columns):
            if c == 'Unnamed: 0':
                print(f'Index {featureAvg[i]}')
            else:
                print(f'{c} {featureAvg[i]}')
    else:
        for train, test in tscv.split(x, y):
            model = RandomForestRegressor(n_estimators=estimators, random_state=42).fit(scaledX.iloc[train], y[train])
            y_pred = model.predict(scaledX.iloc[test])
            r2s += r2_score(y[test], y_pred)
            rmses += np.sqrt(mean_squared_error(y[test], y_pred))
    return r2s / 5, rmses / 5
    
def forestPredictor(q, dd, lag, estimators, x, y, stride = 1):
    x = x.iloc[lag * dd : y.size - q : stride].copy()
    x['BIKE HISTORY 1'] = y[0 : y.size - q - lag * dd : stride]

    for  i in range(1, lag):
        columnName = 'BIKE HISTORY ' + str(i + 1)
        x[columnName] = y[i * dd : y.size - q - (lag - i) * dd : stride]

    y = y[lag * dd + q :: stride]

    scaled_features = StandardScaler().fit_transform(x.values)
    scaledX = pd.DataFrame(scaled_features, index = x.index, columns=x.columns)

    # do cross validation from here, get the mean score and coefficients and then report them at the end
    tscv = TimeSeriesSplit()
    r2Total = 0
    rmseTotal = 0
    for train, test in tscv.split(x, y):
        model = RandomForestRegressor(n_estimators=estimators, random_state=42).fit(scaledX.iloc[train], y[train])
        y_pred = model.predict(scaledX.iloc[test])
        r2Total += r2_score(y[test], y_pred)
        rmseTotal += np.sqrt(mean_squared_error(y[test], y_pred))
    return r2Total/5, rmseTotal/5

def findUsefulFeatures(x, y, dt, modelType):
    if modelType == 'Ridge':
        # We're going to run a Ridge regression, seeing which of the variables are actually used, and then removing different features and seing what effect that has
        plotData(x, y, False)

        x = featurePicker(x, y, modelType)

        Q = [2, 6, 12]
        for q in Q:
            # next test the number of past elements
            hyperParameterPicker(x, y, q, modelType)

        stride = [1, 2, 3, 4, 5, 6]
        for q in Q:
            r2s = []; rmses = []
            for s in stride:
                tmpX = x.copy()
                print(f'Testing a stride of {s}')
                r2, rmse = ridgePredictor(q, 1, 3, tmpX, y, stride = s)
                r2s.append(r2); rmses.append(rmse)
            fig, (ax1, ax2) = plt.subplots(2)
            plotResults(stride, r2s, 'Stride', 'R squared', f'R squared with different strides predicting {q} steps ahead', ax1)
            plotResults(stride, rmses, 'Stride', 'RMSE', f'RMSE with different strides predicting {q} steps ahead', ax2)
            fig.show()
        plt.show()

        # Each of the feature sets are better with less past times included

        # Test Seasonality
        daily = math.floor(24 * 60 * 60 / dt)
        weekly = math.floor(7 * 24 * 60 * 60 / dt)

        dd = [1, daily, weekly]
        ddString = ['Immediate', 'Daily', 'Weekly']
        for q in Q:
            r2s = []; rmses = []
            for i, d in enumerate(dd):
                tmpX = x.copy()
                print(f'\nPredicting {q} steps ahead with a {ddString[i]} trend')
                r2, rmse = ridgePredictor(q, d, 3, tmpX, y, stride = 3)
                r2s.append(r2); rmses.append(rmse)
                print(r2, rmse)
            _, (ax1, ax2) = plt.subplots(2)
            plotResults(ddString, r2s, 'Trend', 'R squared', f'Trend usefullness for predicting {q} steps ahead', ax1)
            plotResults(ddString, rmses, 'Trend', 'RMSE', f'Trend usefullness for predicting {q} steps ahead', ax2)
            plt.show()

        # Using multiple trends at once
        ddString = ['Immediate', 'Immediate and Daily', 'All']
        for q in Q:
            r2s = []; rmses = []
            # Going to test immediate, daily and immediate, and then all together

            tmpX = x.copy()
            print(f'\nPredicting with immediate trend')
            r2, rmse = ridgePredictor(q, 1, 3, tmpX, y, stride=3)
            r2s.append(r2); rmses.append(rmse)

            tmpX = x.copy()
            print(f'\nPredicting with immediate and daily trend')
            r2, rmse = dailyAndImmediatePredictor(q, 3, x, y, daily, stride=3)
            r2s.append(r2); rmses.append(rmse)

            tmpX = x.copy()
            print(f'\nPredicting with immediate, daily and weekly trend')
            r2, rmse = fullSeasonalityPredictor(q, 3, x, y, daily, weekly, stride=3)
            r2s.append(r2); rmses.append(rmse)

            _, (ax1, ax2) = plt.subplots(2)
            plotResults(ddString, r2s, 'Trend', 'R squared', f'Trend Combination Scores for {q} steps ahead', ax1)
            plotResults(ddString, rmses, 'Trend', 'RMSE', f'Trend Combination Scores for {q} steps ahead', ax2)
            plt.show()

        # for Ridge, the best trend is immediate in all cases, and the best lag is 1 as well, however, I believe that a lag of 1 will cause us to actually overfit, and not generalize well
        # so I'm goint to use a lag of 3, as the difference in R squared is not that large, and I believe it will generalize better
        return x, 3
    else:
        plotData(x, y, False)

        print(f'Current R squared and RMSE score with all data {forestPredictor(1, 1, 3, 5, x, y)}')

        x = featurePicker(x, y, modelType)

        Q = [2, 6, 12]
        for q in Q:
            print(f'\nLooking for best hyper parameters for forest looking {q} steps ahead: ')
            hyperParameterPicker(x, y, q, modelType)

        stride = [1, 2, 3, 4, 5, 6]
        for q in Q:
            r2s = []; rmses = []
            for s in stride:
                tmpX = x.copy()
                print(f'Testing a stride of {s}')
                r2, rmse = forestPredictor(q, 1, 3, 10, tmpX, y, stride = s)
                r2s.append(r2); rmses.append(rmse)
            fig, (ax1, ax2) = plt.subplots(2)
            plotResults(stride, r2s, 'Stride', 'R squared', f'R squared with different strides predicting {q} steps ahead', ax1)
            plotResults(stride, rmses, 'Stride', 'RMSE', f'RMSE with different strides predicting {q} steps ahead', ax2)
            fig.show()
        plt.show()

        # Test seasonality
        daily = math.floor(24 * 60 * 60 / dt)
        weekly = math.floor(7 * 24 * 60 * 60 / dt)

        dd = [1, daily, weekly]
        ddString = ['Immediately', 'daily', 'weekly']
        for q in Q:
            r2s = []; rmses = []
            
            tmpX = x.copy()
            r2, rmse = forestPredictor(q, 1, 3, 10, tmpX, y)
            r2s.append(r2); rmses.append(rmse)

            tmpX = x.copy()
            print(f'\nPredicting with immediate and daily trend')
            r2, rmse = dailyAndImmediatePredictor(q, 3, x, y, daily, modelType='Forest', estimators=10)
            r2s.append(r2); rmses.append(rmse)

            tmpX = x.copy()
            print(f'\nPredicting with immediate, daily and weekly trend')
            r2, rmse = fullSeasonalityPredictor(q, 3, x, y, daily, weekly, modelType='Forest', estimators=10)
            r2s.append(r2); rmses.append(rmse)

            fig, (ax1, ax2) = plt.subplots(2)
            plotResults(ddString, r2s, 'Trend', 'R squared', f'R squared variance due to seasonality for {q} steps ahead', ax1)
            plotResults(ddString, rmses, 'Trend', 'RMSE', f'RMSE variance due to seasonality for {q} steps ahead', ax2)
            fig.show()
            plt.show()

        return x, 3, 5

def testModel(x, y, model):
    scaled_features = StandardScaler().fit_transform(x.values)
    x = pd.DataFrame(scaled_features, index=x.index, columns=x.columns)

    tscv = TimeSeriesSplit()
    r2Total = 0; rmseTotal = 0
    for train, test in tscv.split(x, y):
        model = model.fit(x.iloc[train], y[train])
        y_pred = model.predict(x.iloc[test])
        r2Total += r2_score(y[test], y_pred)
        rmseTotal += np.sqrt(mean_squared_error(y[test], y_pred))
    r2 = r2Total / 5; rmse = rmseTotal / 5

    print(f'R squared: {r2}\nRMSE: {rmse}')

def createAndTestModels(x, xRidge, xForest, y, optimalEstimators, dt):
    # 10 minute prediction
    q = 2
    print(f'Predicting {q * 5} minutes ahead')
    print('\nDummy model')
    dummyX = x[0 : y.size - q : 1].copy()
    dummyY = y[q :: 1]
    model = DummyRegressor(strategy='mean')
    testModel(dummyX, dummyY, model)
    
    print('\nRidge regressor')
    lag = 1; dd = 1; stride = 3
    xR = xRidge.iloc[lag * dd + q :: stride].copy()
    yR = y[lag * dd + q :: stride]
    model = Ridge(fit_intercept=False)
    testModel(xR, yR, model)

    print('\nRandom forest regressor')
    lag = 1; dd = 1; stride = 1
    xF = xForest.iloc[lag * dd + q :: stride].copy()
    yF = y[lag * dd + q :: stride]
    model = RandomForestRegressor(n_estimators=optimalEstimators, random_state=42)
    testModel(xF, yF, model)
    print('\n')

    # 30 minute prediction
    q = 6
    print(f'Predicting {q * 5} minutes ahead')
    print('\nDummy model')
    dummyX = x.iloc[0 : : 1].copy()
    dummyY = y[q :: 1]
    model = DummyRegressor(strategy='mean')
    testModel(dummyX, dummyY, model)
    
    print('\nRidge regressor')
    lag = 1; dd = 1; stride = 3
    xR = xRidge.iloc[lag * dd + q :: stride].copy()
    yR = y[lag * dd + q :: stride]
    model = Ridge(fit_intercept=False)
    testModel(xR, yR, model)

    print('\nRandom forest regressor')
    lag = 1; dd = 1; stride = 3
    xF = xForest.iloc[lag * dd + q :: stride].copy()
    yF = y[lag * dd + q :: stride]
    model = RandomForestRegressor(n_estimators=optimalEstimators, random_state=42)
    testModel(xF, yF, model)
    print('\n')

    # 60 minute prediction
    q = 12
    print(f'\nPredicting {q * 5} minutes ahead')
    print('\nDummy model')
    dummyX = x.iloc[0 : : 1].copy()
    dummyY = y[q :: 1]
    model = DummyRegressor(strategy='mean')
    testModel(dummyX, dummyY, model)
    
    print('\nRidge regressor')
    lag = 1; dd = 1; stride = 1
    xR = xRidge.iloc[lag * dd + q :: stride].copy()
    yR = y[lag * dd + q :: stride].copy()
    model = Ridge(fit_intercept=False)
    testModel(xR, yR, model)

    print('\nRandom forest regressor')
    lag = 1; dd = 1; stride = 1
    xF = xForest.iloc[lag * dd + q :: stride].copy()
    yF = y[lag * dd + q :: stride]
    model = RandomForestRegressor(n_estimators=optimalEstimators, random_state=42)
    testModel(xF, yF, model)

#reduceStations()
data = pd.read_csv('./data/bike_data.csv', parse_dates=['LAST UPDATED'])

x, y, dt = transformData(data, 'BLESSINGTON STREET')
#xRidge, optimalLagRidge = findUsefulFeatures(x, y, dt, 'Ridge')
xForest, optimalLagForest, optimalEstimators = findUsefulFeatures(x, y, dt, 'Forest')

# createAndTestModels(x, xRidge, xForest, y, optimalEstimators, dt)

# x, y, dt = transformData(data, 'KILMAINHAM GAOL')
# xRidge, optimalLagRidge = findUsefulFeatures(x, y, dt, 'Ridge')
# xForest, optimalLagForest, optimalEstimators = findUsefulFeatures(x, y, dt, 'Forest')

# createAndTestModels(x, xRidge, xForest, y, optimalEstimators, dt)

# A note on the data, the data is extremely inconsistent, there is no guarentee that any day will be recorded
# Going to use Ridge regression and Random Forest Regression
=======
import math
import matplotlib as plt
import numpy as np
import pandas as pd

def euclidAbsDistance(baseLat, baseLong, xLat, xLong):
    print( abs(math.sqrt((baseLat - xLat) + (baseLong - xLong))) )

def reduceStationsAndFilterFeatures():
    data = pd.read_csv('dublinbikes_20200101_20200401.csv')
    allGeocodes = data.copy()
    allGeocodes = allGeocodes.drop_duplicates(subset=['STATION ID'])
    allGeocodes = allGeocodes[['STATION ID', 'LATITIUDE', 'LONGITUDE']]
    base = allGeocodes.loc[allGeocodes['STATION ID'] == 2]
    allGeocodes = allGeocodes.drop(index=0)
    dists = allGeocodes.apply(lambda x : euclidAbsDistance(base['LATITUDE'], base['LONGITUDE'], x['LATITUDE'], x['LONGITUDE']))
    print(dists)
    # furthestStation = maxDists() where it returns the index, then get the station id for that index
    # reducedData = data.loc[(data['STATION ID'] == 2) | (data['STATION ID'] == furthestStation)]
    # reducedData.drop(columns=['NAME', 'ADDRESS', 'LATITUDE', 'LONGITUDE'])
    # reducedData.to_csv('bike_data.csv')

reduceStationsAndFilterFeatures()
>>>>>>> 49c746edbb6952823efae06a4f90ca5b4ec4c9da
