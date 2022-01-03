import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True

def euclidAbsDistance(baseLat, baseLong, xLat, xLong):
    return abs(math.sqrt((baseLat - xLat) ** 2 + (baseLong - xLong) ** 2))

def reduceStations():
    data = pd.read_csv('./data/dublinbikes_20200101_20200401.csv')
    copy = data.copy()


    allGeocodes = data.copy()
    allGeocodes = allGeocodes.drop_duplicates(subset=['STATION ID'])
    allGeocodes = allGeocodes[ ['STATION ID', 'LATITUDE', 'LONGITUDE'] ]
    base = allGeocodes.loc[allGeocodes['STATION ID'] == 2]
    baseLat = base['LATITUDE'][0]; baseLong = base['LONGITUDE'][0]
    allGeocodes = allGeocodes.drop(index=0)
    dists = allGeocodes.apply(lambda x : euclidAbsDistance(baseLat, baseLong, x['LATITUDE'], x['LONGITUDE']), axis=1)
    furthestStation = data.iloc[dists.idxmax()]['STATION ID']
    reducedData = data.loc[(data['STATION ID'] == 2) | (data['STATION ID'] == furthestStation)].copy()

    ## before we write we need to add in a column for bikes in station
    maxBikes = reducedData['BIKE STANDS']
    availableSpots = reducedData['AVAILABLE BIKE STANDS']
    currentOccupancy = maxBikes - availableSpots
    reducedData['CURRENT OCCUPANCY'] = currentOccupancy
    reducedData.to_csv('./data/bike_data.csv')

def extractOccupancyOnly(data, station):
    df = data.loc[(data['NAME'] == station)].copy()
    start = pd.to_datetime("04-02-2020", format='%d-%m-%Y')
    end = pd.to_datetime("14-03-2020", format='%d-%m-%Y')
    t_full = pd.array(pd.DatetimeIndex(df.loc[:, 'TIME']).view(np.int64)) / 1000000000
    dt = t_full[1] - t_full[0]
    t_start = pd.DatetimeIndex([start]).view(np.int64) / 1000000000
    t_end = pd.DatetimeIndex([end]).view(np.int64) / 1000000000
    y = np.extract([(t_full >= t_start) & (t_full <= t_end)], df.loc[:, 'CURRENT OCCUPANCY']).view(np.int64)

    df = data.copy()
    start = pd.to_datetime("04-02-2020", format='%d-%m-%Y')
    end = pd.to_datetime("14-03-2020", format='%d-%m-%Y')
    t_full = pd.array(pd.DatetimeIndex(df.loc[:, 'TIME']).view(np.int64)) / 1000000000
    dt = t_full[1] - t_full[0]
    t_start = pd.DatetimeIndex([start]).view(np.int64) / 1000000000
    t_end = pd.DatetimeIndex([end]).view(np.int64) / 1000000000
    fullY = np.extract([(t_full >= t_start) & (t_full <= t_end)], df.loc[:, 'CURRENT OCCUPANCY']).view(np.int64)

    return y, fullY, dt

def transformData(data, station):
    df = data.loc[(data['NAME'] == station)].copy()
    # first transform all the data, and then take just the data that is in t
    t = pd.array(pd.DatetimeIndex(df.loc[:, 'TIME']).view(np.int64)) / 1000000000
    df['TIME'] = t;

    start = pd.to_datetime("04-02-2020", format='%d-%m-%Y')
    end = pd.to_datetime("14-03-2020", format='%d-%m-%Y')
    t_full = t
    dt = t_full[1] - t_full[0]
    t_start = pd.DatetimeIndex([start]).view(np.int64) / 1000000000
    t_end = pd.DatetimeIndex([end]).view(np.int64) / 1000000000
    t = np.extract([(t_full >= t_start) & (t_full <= t_end)], t_full)
    y = np.extract([(t_full >= t_start) & (t_full <= t_end)], df.loc[:, 'CURRENT OCCUPANCY']).view(np.int64)
    print('data sampling interval is %d secs' %dt)

    df = df[df['TIME'].isin(t)]
    lastUpdated = pd.array(pd.DatetimeIndex(df.loc[:, 'LAST UPDATED']).view(np.int64)) / 1000000000
    df['LAST UPDATED'] = lastUpdated
    df = df.sort_values(by=['TIME'])
    t = (t - t[0]) / 60 / 60 / 24

    # Turn STATUS into binary: 1 = Open, 0 = Closed
    status = pd.array(df['STATUS'])
    status = [1 if x == 'Open' else 0 for x in status]
    df['STATUS'] = status

    # need to add t back onto the features
    df = df.drop(['CURRENT OCCUPANCY'], axis=1); df = df.drop(['NAME'], axis=1); df = df.drop(['ADDRESS'], axis=1)
    df['TIME IN DAYS'] = t

    d = data.copy()
    t = pd.array(pd.DatetimeIndex(d.loc[:, 'TIME']).view(np.int64)) / 1000000000
    d['TIME'] = t;

    start = pd.to_datetime("04-02-2020", format='%d-%m-%Y')
    end = pd.to_datetime("14-03-2020", format='%d-%m-%Y')
    t_full = t
    t_start = pd.DatetimeIndex([start]).view(np.int64) / 1000000000
    t_end = pd.DatetimeIndex([end]).view(np.int64) / 1000000000
    t = np.extract([(t_full >= t_start) & (t_full <= t_end)], t_full)
    y = np.extract([(t_full >= t_start) & (t_full <= t_end)], d.loc[:, 'CURRENT OCCUPANCY']).view(np.int64)

    d = d[d['TIME'].isin(t)]
    lastUpdated = pd.array(pd.DatetimeIndex(d.loc[:, 'LAST UPDATED']).view(np.int64)) / 1000000000
    d['LAST UPDATED'] = lastUpdated
    d = d.sort_values(by=['TIME'])
    t = (t - t[0]) / 60 / 60 / 24

    # Turn STATUS into binary: 1 = Open, 0 = Closed
    status = pd.array(d['STATUS'])
    status = [1 if x == 'Open' else 0 for x in status]
    d['STATUS'] = status

    # need to add t back onto the features
    d = d.drop(['CURRENT OCCUPANCY'], axis=1); d = d.drop(['NAME'], axis=1); d = d.drop(['ADDRESS'], axis=1)
    d['TIME IN DAYS'] = t

    return df, d

def plotResults(x, r2s, rmses, xlabel, title):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(title)
    ax1.scatter(x=x, y=r2s); ax1.set(xlabel=xlabel, ylabel='R squared')
    ax2.scatter(x=x, y=rmses); ax2.set(xlabel=xlabel, ylabel='RMSE')
    fig.show()

def predictor(y, q, dd, lag, modelType='Ridge', stride=1, estimators=20, evl=False):
    #q-step ahead prediction
    XX = y[0 : y.size - q - lag * dd : stride]
    for i in range(1, lag):
        X = y[i * dd : y.size - q - (lag - i) * dd : stride]
        XX = np.column_stack((XX, X))
    yy = y[lag * dd + q :: stride]
    #train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)

    tscv = TimeSeriesSplit()
    if lag == 1:
        featureTotals = 0
    else:
        featureTotals = [0] * XX.shape[1]

    if lag == 1:
        XX = StandardScaler().fit_transform(XX.reshape(-1, 1))
    else:
        XX = StandardScaler().fit_transform(XX)

    r2Total = 0
    rmseTotal = 0
    maeTotal = 0
    if modelType == 'Ridge':
        for train, test in tscv.split(XX, yy):
            model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
            featureTotals += model.coef_
            y_pred = model.predict(XX[test])
            r2Total += r2_score(yy[test], y_pred); rmseTotal += np.sqrt(mean_squared_error(yy[test], y_pred)); maeTotal += mean_absolute_error(yy[test], y_pred)
        print(featureTotals / ([5] * XX.shape[1]))
    else:
        for train, test in tscv.split(XX, yy):
            model = RandomForestRegressor(n_estimators=estimators, random_state=42).fit(XX[train], yy[train])
            y_pred = model.predict(XX[test])
            r2Total += r2_score(yy[test], y_pred); rmseTotal += np.sqrt(mean_squared_error(yy[test], y_pred)); maeTotal += mean_absolute_error(yy[test], y_pred)

    if evl:
        return r2Total / 5, rmseTotal / 5, maeTotal / 5
    return r2Total / 5, rmseTotal / 5

def fullPredictor(x, y, q, dd, lag, modelType='Ridge'):
    stride=1

    x = x.iloc[(lag - 1) * dd : y.size - q - (lag - 2) * dd : stride].copy()

    XX = y[0 : y.size - q - lag * dd : stride]
    for i in range(0, lag):
        X = y[i * dd : y.size - q - (lag - i) * dd : stride]
        XX = np.vstack((XX, X))

    for i, X in enumerate(XX):
        x['BIKE HISTORY ' + str(i + 1)] = X
    scaled_features = StandardScaler().fit_transform(x.values)
    scaledX = pd.DataFrame(scaled_features, index = x.index, columns=x.columns)
    yy = y[lag * dd + q :: stride]
    #train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)

    tscv = TimeSeriesSplit()
    featureTotals = [0] * scaledX.shape[1]
    r2Total = 0
    rmseTotal = 0
    if modelType == 'Ridge':
        for train, test in tscv.split(scaledX, yy):
            model = Ridge(fit_intercept=False).fit(scaledX.iloc[train], yy[train])
            featureTotals += model.coef_
            y_pred = model.predict(scaledX.iloc[test])
            r2Total += r2_score(yy[test], y_pred); rmseTotal += np.sqrt(mean_squared_error(yy[test], y_pred))
        print(featureTotals / ([5] * scaledX.shape[1]))
    else:
        for train, test in tscv.split(scaledX, yy):
            model = RandomForestRegressor(n_estimators=20, random_state=42).fit(scaledX.iloc[train], yy[train])
            y_pred = model.predict(scaledX.iloc[test])
            r2Total += r2_score(yy[test], y_pred); rmseTotal += np.sqrt(mean_squared_error(yy[test], y_pred))

    return r2Total / 5, rmseTotal / 5

def seasonalityPredictor(y, q, d, w, lag, modelType='Ridge', stride=1, estimators=20, evl=False):
    lngth = y.size - w - lag * w - q
    XX = y[q : q + lngth : stride]
    for i in range(1, lag):
        X = y[i * w + q : i * w + q + lngth : stride]
        XX = np.column_stack((XX, X))
    for i in range(0, lag):
        X = y[i * d + q : i * d + q + lngth : stride]
        XX = np.column_stack((XX, X))
    for i in range(0, lag):
        X = y[i : i + lngth : stride]
        XX = np.column_stack((XX, X))
    yy = y[lag * w + w + q : lag * w + w + q + lngth : stride]
    #train, test  = train_test_split(np.arange(0, yy.size), test_size=0.2)

    tscv = TimeSeriesSplit()
    featureTotals = [0] * XX.shape[1]
    XX = StandardScaler().fit_transform(XX)
    r2Total = 0
    rmseTotal = 0
    maeTotal = 0
    if modelType == 'Ridge':
        for train, test in tscv.split(XX, yy):
            model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
            featureTotals += model.coef_
            y_pred = model.predict(XX[test])
            r2Total += r2_score(yy[test], y_pred); rmseTotal += np.sqrt(mean_squared_error(yy[test], y_pred)); maeTotal += mean_absolute_error(yy[test], y_pred)
        print(featureTotals / ([5] * XX.shape[1]))
    else:
        for train, test in tscv.split(XX, yy):
            model = RandomForestRegressor(n_estimators=estimators, random_state=42).fit(XX[train], yy[train])
            y_pred = model.predict(XX[test])
            r2Total += r2_score(yy[test], y_pred); rmseTotal += np.sqrt(mean_squared_error(yy[test], y_pred)); maeTotal += mean_absolute_error(yy[test], y_pred)

    if evl:
        return r2Total / 5, rmseTotal / 5, maeTotal / 5
    return r2Total / 5, rmseTotal / 5

def findUsefulFeatures(x, y, dt, modelType, x1=0, y1=0):
    d = math.floor(24 * 60 * 60 / dt)
    w = math.floor(7 * 24 * 60 * 60 / dt)
    Q = [2, 6, 12]

    if modelType == 'Ridge':
        # Going to plot q vs r2 and q vs rmse, with two colors, one for just time series data, and one with full data
        r2s, r2sFull, rmses, rmsesFull = [], [], [], []
        r2s_kil, r2sFull_kil, rmses_kil, rmsesFull_kil = [], [], [], []
        for q in Q:
            r2, rmse = predictor(y, q, 1, 3)
            r2s.append(r2); rmses.append(rmse)
            r2, rmse = fullPredictor(x, y, q, 1, 3)
            r2sFull.append(r2); rmsesFull.append(rmse)

            r2, rmse = predictor(y1, q, 1, 3)
            r2s_kil.append(r2); rmses_kil.append(rmse)
            r2, rmse = fullPredictor(x1, y1, q, 1, 3)
            r2sFull_kil.append(r2); rmsesFull_kil.append(rmse)
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(f'Comparison of time series only model vs timeseries with extra data predictor using Ridge')
        axs[0, 0].scatter(x=Q, y=r2s); axs[0, 0].scatter(x=Q, y=r2sFull); axs[0, 0].set(xlabel='q', ylabel='R squared'); #axs[0, 0].legend(['Time Series', 'Full Dataset']); 
        axs[0, 0].set_title("Blessington Street")
        axs[1, 0].scatter(x=Q, y=rmses); axs[1, 0].scatter(x=Q, y=rmsesFull); axs[1, 0].set(xlabel='q', ylabel='RMSE'); #axs[1, 0].legend(['Time Series', 'Full Dataset'])
        axs[0, 1].scatter(x=Q, y=r2s_kil); axs[0, 1].scatter(x=Q, y=rmses_kil); axs[0, 1].set(xlabel='q', ylabel='R squared'); #axs[0, 1].legend(['Time Series', 'Full Dataset']); 
        axs[0, 1].set_title("Kilmainham Gaol")
        axs[1, 1].scatter(x=Q, y=rmses_kil); axs[1, 1].scatter(x=Q, y=rmsesFull_kil); axs[1, 1].set(xlabel='q', ylabel='RMSE'); #axs[1, 1].legend(['Time Series', 'Full Dataset'])
        fig.legend(['Time Series', 'Full Dataset'])
        plt.show()

        streets = ['Blessington Street', 'Kilmainham Gaol']

        ### Find best lag
        lag = [1, 2, 3, 4, 5, 6]
        r2s, rmses, r2s_kil, rmses_kil = [], [], [], []
        for l in lag:
            r2, rmse = predictor(y, q, 1, l)
            r2s.append(r2); rmses.append(rmse)

            r2, rmse = predictor(y1, q, 1, l)
            r2s_kil.append(r2); rmses_kil.append(rmse)
        plotResults(lag, r2s, rmses, 'lag', f'Effect of lag on Ridge Regression predictions for Blessington Street')
        plotResults(lag, r2s_kil, rmses_kil, 'lag', f'Effect of lag on Ridge Regression predictions for Kilmainham Gaol')

        ### Find best stride
        stride = [1, 2, 3, 4, 5, 6]
        r2s, rmses, r2s_kil, rmses_kil = [], [], [], []
        for s in stride:
            r2, rmse = predictor(y, q, 1, 3, stride=s)
            r2s.append(r2); rmses.append(rmse)

            r2, rmse = predictor(y1, q, 1, 3, stride=s)
            r2s_kil.append(r2); rmses_kil.append(rmse)
        plotResults(stride, r2s, rmses, 'stride', f'Effect of stride on Ridge Regression predictions for Blessington Street')
        plotResults(stride, r2s_kil, rmses_kil, 'stride', f'Effect of stride on Ridge Regression predictions for Kilmainham Gaol')
        plt.show()

        immediateR2, dailyR2, weeklyR2, allR2 = [], [], [], []
        immediateR2_kil, dailyR2_kil, weeklyR2_kil, allR2_kil = [], [], [], []
        immediateRMSE, dailyRMSE, weeklyRMSE, allRMSE = [], [], [], []
        immediateRMSE_kil, dailyRMSE_kil, weeklyRMSE_kil, allRMSE_kil = [], [], [], []
        seasonality = ['Immediate', 'Daily', 'Weekly', 'All']
        for q in Q:
            print('Models with only occupancy data')
            print(f'Predicting {q} steps ahead')
            print('Immediate Trend')
            r2, rmse = predictor(y, q, 1, 3)
            immediateR2.append(r2); immediateRMSE.append(rmse)
            print('Daily Seasonality')
            r2, rmse = predictor(y, q, d, 3)
            dailyR2.append(r2); dailyRMSE.append(rmse)
            print('Weekly Seasonality')
            r2, rmse = predictor(y, q, w, 3)
            weeklyR2.append(r2); weeklyRMSE.append(rmse)
            print('All Together')
            r2, rmse = seasonalityPredictor(y, q, d, w, 3)
            allR2.append(r2); allRMSE.append(rmse)
            print('\n')

            print('Immediate Trend')
            r2, rmse = predictor(y1, q, 1, 3)
            immediateR2_kil.append(r2); immediateRMSE_kil.append(rmse)
            print('Daily Seasonality')
            r2, rmse = predictor(y1, q, d, 3)
            dailyR2_kil.append(r2); dailyRMSE_kil.append(rmse)
            print('Weekly Seasonality')
            r2, rmse = predictor(y1, q, w, 3)
            weeklyR2_kil.append(r2); weeklyRMSE_kil.append(rmse)
            print('All Together')
            r2, rmse = seasonalityPredictor(y1, q, d, w, 3)
            allR2_kil.append(r2); allRMSE_kil.append(rmse)
            print('\n')
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(f'Comparison of using different seasonalities using Ridge Regression')
        axs[0, 0].scatter(x=Q, y=immediateR2); axs[0, 0].scatter(x=Q, y=dailyR2); axs[0, 0].scatter(x=Q, y=weeklyR2); axs[0, 0].scatter(x=Q, y=allR2); axs[0, 0].set(xlabel='q', ylabel='R squared'); #axs[0, 0].legend(seasonality); 
        axs[0, 0].set_title("Blessington Street")
        axs[1, 0].scatter(x=Q, y=immediateRMSE); axs[1, 0].scatter(x=Q, y=dailyRMSE); axs[1, 0].scatter(x=Q, y=weeklyRMSE); axs[1, 0].scatter(x=Q, y=allRMSE); axs[1, 0].set(xlabel='q', ylabel='RMSE'); #axs[1, 0].legend(seasonality)
        axs[0, 1].scatter(x=Q, y=immediateR2_kil); axs[0, 1].scatter(x=Q, y=dailyR2_kil); axs[0, 1].scatter(x=Q, y=weeklyR2_kil); axs[0, 1].scatter(x=Q, y=allR2_kil); axs[0, 1].set(xlabel='q', ylabel='R squared'); #axs[0, 1].legend(seasonality); 
        axs[0, 1].set_title("Kilmainham Gaol")
        axs[1, 1].scatter(x=Q, y=immediateRMSE_kil); axs[1, 1].scatter(x=Q, y=dailyRMSE_kil); axs[1, 1].scatter(x=Q, y=weeklyRMSE_kil); axs[1, 1].scatter(x=Q, y=allRMSE_kil); axs[1, 1].set(xlabel='q', ylabel='RMSE'); #axs[1, 1].legend(seasonality)
        fig.legend(seasonality)
        plt.show()
    else:
        r2s, r2sFull, rmses, rmsesFull = [], [], [], []
        for q in Q:
            r2, rmse = predictor(y, q, 1, 3, 'Forest')
            r2s.append(r2); rmses.append(rmse)
            r2, rmse = fullPredictor(x, y, q, 1, 3, 'Forest')
            r2sFull.append(r2); rmsesFull.append(rmse)
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Comparison of time series only model vs timeseries with extra data predictor on Random Forest Model')
        ax1.scatter(x=Q, y=r2s); ax1.scatter(x=Q, y=r2sFull); ax1.set(xlabel='q', ylabel='R squared'); #ax1.legend(['Time Series', 'Full Dataset'])
        ax2.scatter(x=Q, y=rmses); ax2.scatter(x=Q, y=rmsesFull); ax2.set(xlabel='q', ylabel='RMSE'); #ax2.legend(['Time Series', 'Full Dataset'])
        fig.legend(['Time Series', 'Full Dataset'])
        plt.show()

        ### Find best lag
        lag = [1, 2, 3, 4, 5, 6]
        r2s, rmses = [], []
        for l in lag:
            r2, rmse = predictor(y, q, 1, l)
            r2s.append(r2); rmses.append(rmse)
        plotResults(lag, r2s, rmses, 'lag', f'Effect of lag on on Random Forest Model predictions')

        ### Find best stride
        stride = [1, 2, 3, 4, 5, 6]
        r2s, rmses = [], []
        for s in stride:
            r2, rmse = predictor(y, q, 1, 3, stride=s)
            r2s.append(r2); rmses.append(rmse)
        plotResults(stride, r2s, rmses, 'stride', f'Effect of stride on on Random Forest Model predictions')

        ### Find best no. estimators
        estimators = [1, 5, 10, 20, 50]
        r2s, rmses = [], []
        for e in estimators:
            r2, rmse = predictor(y, q, 1, 3, 'Forest', estimators=e)
            r2s.append(r2); rmses.append(rmse)
        plotResults(estimators, r2s, rmses, 'estimators', 'Effect of estimators on Random Forest Model predictions')
        plt.show()

        immediateR2, dailyR2, weeklyR2, allR2 = [], [], [], []
        immediateRMSE, dailyRMSE, weeklyRMSE, allRMSE = [], [], [], []
        seasonality = ['Immediate', 'Daily', 'Weekly', 'All']
        for q in Q:
            print('Models with only occupancy data')
            print(f'Predicting {q} steps ahead')
            print('Immedieate Trend')
            r2, rmse = predictor(y, q, 1, 6, 'Forest', stride=4)
            immediateR2.append(r2); immediateRMSE.append(rmse)
            print('Daily Seasonality')
            r2, rmse = predictor(y, q, d, 6, 'Forest', stride=4)
            dailyR2.append(r2); dailyRMSE.append(rmse)
            print('Weekly Seasonality')
            r2, rmse = predictor(y, q, w, 6, 'Forest', stride=4)
            weeklyR2.append(r2); weeklyRMSE.append(rmse)
            print('All Together')
            r2, rmse = seasonalityPredictor(y, q, d, w, 6, 'Forest', stride=4)
            allR2.append(r2); allRMSE.append(rmse)
            print('\n')
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Comparison of using different seasonalities on Random Forest Model')
        ax1.scatter(x=Q, y=immediateR2); ax1.scatter(x=Q, y=dailyR2); ax1.scatter(x=Q, y=weeklyR2); ax1.scatter(x=Q, y=allR2); ax1.set(xlabel='q', ylabel='R squared'); #ax1.legend(seasonality)
        ax2.scatter(x=Q, y=immediateRMSE); ax2.scatter(x=Q, y=dailyRMSE); ax2.scatter(x=Q, y=weeklyRMSE); ax2.scatter(x=Q, y=allRMSE); ax2.set(xlabel='q', ylabel='RMSE'); #ax2.legend(seasonality)
        fig.legend(seasonality)
        plt.show()

def evalDummy(model, y, q, lag, stride):
    XX = y[0 : y.size - q - lag * 1 : stride]
    for i in range(1, lag):
        X = y[i * 1 : y.size - q - (lag - i) * 1 : stride]
        XX = np.column_stack((XX, X))
    yy = y[lag * 1 + q :: stride]
    
    tscv = TimeSeriesSplit()
    featureTotals = [0] * XX.shape[1]
    XX = StandardScaler().fit_transform(XX)

    r2Total = 0
    rmseTotal = 0
    maeTotal = 0

    for train, test in tscv.split(XX, yy):
        model.fit(XX[train], yy[train])
        y_pred = model.predict(XX[test])
        r2Total += r2_score(yy[test], y_pred); rmseTotal += np.sqrt(mean_squared_error(yy[test], y_pred)); maeTotal += mean_absolute_error(yy[test], y_pred)
    return r2Total / 5, rmseTotal / 5, maeTotal / 5

def testModels(y1, y2, combinedY, dt):
    d = math.floor(24 * 60 * 60 / dt)
    w = math.floor(7 * 24 * 60 * 60 / dt)

    # When testing the models, we're going to use R squared, RMSE, MAE
    q = 2
    print(f'Evaluations for predicting {q * 5} minutes ahead')

    ### Dummy
    dummyY = combinedY[q :: 1]
    model = DummyRegressor(strategy='mean')
    r2, rmse, mae = evalDummy(model, dummyY, 1, 3, 1)
    print(f'\nDummy model: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    ### Ridge Seasonality: 1  lag: 1 for Blessington and 6 for Kilmainham    Stride: 2 and 1
    r2, rmse, mae = predictor(y1, q, 1, 1, stride=2, evl=True)
    print(f'\nRidge model on Blessington Street: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')
    r2, rmse, mae = predictor(y2, q, 1, 6, stride=1, evl=True)
    print(f'\nRidge model: Kilmainham Gaol\nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    ### Forest Seasonality: 1 Lag: 6    Stride: 4
    r2, rmse, mae = predictor(combinedY, q, 1, 6, modelType='Forest', stride=4, estimators=20, evl=True)
    print(f'\nRandom forest regressor: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    q = 6
    print(f'Evaluations for predicting {q * 5} minutes ahead')

    ### Dummy
    dummyY = combinedY[q :: 1]
    model = DummyRegressor(strategy='mean')
    r2, rmse, mae = evalDummy(model, dummyY, 1, 3, 1)
    print(f'\nDummy model: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    ### Ridge Seasonality: 1  lag:    Stride: 
    r2, rmse, mae = predictor(y1, q, 1, 1, stride=2, evl=True)
    print(f'\nRidge model on Blessington Street: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')
    r2, rmse, mae = predictor(y2, q, 1, 6, stride=1, evl=True)
    print(f'\nRidge model: Kilmainham Gaol\nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    ### Forest Seasonality: Lag:    Stride: 
    r2, rmse, mae = predictor(combinedY, q, 1, 6, modelType='Forest', stride=4, estimators=20, evl=True)
    print(f'\nRandom forest regressor: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    q = 12
    print(f'Evaluations for predicting {q * 5} minutes ahead')

    ### Dummy
    dummyY = combinedY[q :: 1]
    model = DummyRegressor(strategy='mean')
    r2, rmse, mae = evalDummy(model, dummyY, 1, 3, 1)
    print(f'\nDummy model: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    ### Ridge Seasonality: 1  lag:    Stride: 
    r2, rmse, mae = predictor(y1, q, 1, 1, stride=2, evl=True)
    print(f'\nRidge model on Blessington Street: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')
    r2, rmse, mae = predictor(y2, q, 1, 6, stride=1, evl=True)
    print(f'\nRidge model: Kilmainham Gaol\nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    ### Forest Seasonality: Lag:    Stride: 
    r2, rmse, mae = predictor(combinedY, q, 1, 6, modelType='Forest', stride=4, estimators=20, evl=True)
    print(f'\nRandom forest regressor: \nR2 = {r2}, RMSE = {rmse}, MAE = {mae}')

    ### Finally test Kilmainham on the Blessington Ridge model and Blessington on the Kilmainham Ridge model

#reduceStations()
data = pd.read_csv('./data/bike_data.csv', parse_dates=['TIME'])
xExtraBless, _ = transformData(data, 'BLESSINGTON STREET')
yBless, _, dt = extractOccupancyOnly(data, 'BLESSINGTON STREET')

xExtraKil, combinedExtraX = transformData(data, 'KILMAINHAM GAOL')
yKil, combinedY, dt = extractOccupancyOnly(data, 'KILMAINHAM GAOL')

findUsefulFeatures(xExtraBless, yBless, dt, 'Ridge', x1=xExtraKil, y1=yKil)

findUsefulFeatures(combinedExtraX, combinedY, dt, 'Forest')

testModels(yBless, yKil, combinedY, dt)