import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True

def euclidAbsDistance(baseLat, baseLong, xLat, xLong):
    return abs(math.sqrt((baseLat - xLat) ** 2 + (baseLong - xLong) ** 2))

def reduceStations():
    data = pd.read_csv('./data/dublinbikes_20200101_20200401.csv')
    allGeocodes = data.copy()
    allGeocodes = allGeocodes.drop_duplicates(subset=['STATION ID'])
    allGeocodes = allGeocodes[ ['STATION ID', 'LATITUDE', 'LONGITUDE'] ]
    base = allGeocodes.loc[allGeocodes['STATION ID'] == 2]
    baseLat = base['LATITUDE'][0]; baseLong = base['LONGITUDE'][0]
    allGeocodes = allGeocodes.drop(index=0)
    dists = allGeocodes.apply(lambda x : euclidAbsDistance(baseLat, baseLong, x['LATITUDE'], x['LONGITUDE']), axis=1)
    furthestStation = data.iloc[dists.idxmax()]['STATION ID']
    reducedData = data.loc[(data['STATION ID'] == 2) | (data['STATION ID'] == furthestStation)]

    ## before we write we need to add in a column for bikes in station
    maxBikes = reducedData['BIKE STANDS']
    availableSpots = reducedData['AVAILABLE BIKE STANDS']
    currentOccupancy = maxBikes - availableSpots
    reducedData['CURRENT OCCUPANCY'] = currentOccupancy
    reducedData.to_csv('./data/bike_data.csv')

def transformData(data, station):
    df = data[ data['NAME'] == station ]
    t = pd.array(pd.DatetimeIndex(df.iloc[:, 2]).view(np.int64)) / 1000000000
    lastUpdated = pd.array(pd.DatetimeIndex(df.iloc[:, 3]).view(np.int64)) / 1000000000
    dt = t[1] - t[0]
    print('data sampling interval is %d secs' %dt)

    df['TIME'] = t; df['LAST UPDATED'] = lastUpdated
    t = (t - t[0]) / 60 / 60 / 24
    y = np.array(df.iloc[:, 12].view(np.int64))

    # Turn STATUS into binary: 1 = Open, 0 = Closed
    status = pd.array(df['STATUS'])
    status = [1 if x == 'Open' else 0 for x in status]
    df['STATUS'] = status

    # need to add t back onto the features
    df = df.drop(['CURRENT OCCUPANCY'], axis=1); df = df.drop(['NAME'], axis=1); df = df.drop(['ADDRESS'], axis=1)
    df['TIME IN DAYS'] = t

    return df, y, dt

def plotData(x, y, station, plot):
    if plot:
        t = x.iloc[:, 12]
        plt.scatter(t, y, color='red', marker='.'); plt.title(f'{station.lower()} data'); plt.show()

# Parameters:
# q: how far into the future we are attempting to predict
# 
# lag: the number of previous events to take into account when using previous data
# 
# Returns:
# 
def predictor(q, dd, lag, plot, x, y, dt):
    ## Want to add cross validation and f1 or f2 score to this to finish off

    stride = 1

    # merge XX and x and then do to 
    x = x.iloc[lag * dd + q :: stride]
    x['BIKE HISTORY 1'] = y[0 : y.size - q - lag * dd : stride]

    for  i in range(1, lag):
        columnName = 'BIKE HISTORY ' + str(i + 1)
        x[columnName] = y[i * dd : y.size - q - (lag - i) * dd : stride]

    t = np.array(x['TIME IN DAYS'])
    yy = y[lag * dd + q :: stride]
    y = y[lag * dd + q :: stride]

    scaled_features = StandardScaler().fit_transform(x.values)
    scaledX = pd.DataFrame(scaled_features, index = x.index, columns=x.columns)

    # do cross validation from here, get the mean score and coefficients and then report them at the end
    kf = KFold(n_splits=5)
    tscv = TimeSeriesSplit()
    featureTotals = [0] * x.shape[1]
    mseTotal = 0
    for train, test in tscv.split(x, yy):
        model = Ridge(fit_intercept=False).fit(scaledX.iloc[train], yy[train])
        featureTotals += model.coef_
        mseTotal += mean_squared_error(yy[test], model.predict(scaledX.iloc[test]))
    print(mseTotal / 5)
    featureAvg = featureTotals / ([5] * x.shape[1])
    for i, c in enumerate(scaledX.columns):
        if c == 'Unnamed: 0':
            print(f'Index {featureAvg[i]}')
        else:
            print(f'{c} {featureAvg[i]}')

    # train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
    # model = Ridge(fit_intercept=False).fit(scaledX.iloc[train], yy[train])
    # print(model.coef_)
    # print(mean_squared_error(yy[test], model.predict(scaledX.iloc[test])))
    # for i, c in enumerate(scaledX.columns):
    #     if c == 'Unnamed: 0':
    #         print(f'Index {model.coef_[i]}')
    #     else:
    #         print(f'{c} {model.coef_[i]}')
    # if plot:
    #     # need to scale t to make plotting work
    #     y_pred = model.predict(x)
    #     plt.scatter(t, y, color='black'); plt.scatter(t, y_pred, color='blue')
    #     plt.xlabel("time(days)"); plt.ylabel('#bikes')
    #     plt.legend(["training data", "predictions"], loc='upper right')
    #     day = math.floor(24*60*60/dt)
    #     plt.xlim(((lag * dd + q) / day, (lag * dd + q) / day + 2))
    #     plt.show()
    # else:
    #     # print the accuracy

    #     pass
    
def findUsefulFeatures(x, y, dt, station):
    # We're going to run a basic linear regression on this, seeing which of the variables are actually used, and then removing different features and seing what effect that has
    plotData(x, y, station, False)

    # Should probably just use the most basic of predictors first with an immediate trend predictor
    # short term trend prediciton
    predictor(1, 1, 3, False, x, y, dt)

    columns = ['Unnamed: 0', 'STATION ID', 'TIME', 'LAST UPDATED', 'BIKE STANDS', 'AVAILABLE BIKE STANDS', 'AVAILABLE BIKES', 'STATUS', 'LATITUDE', 'LONGITUDE']
    for c in columns:
        tmpX = x.copy()
        tmpX = tmpX.drop(c, axis=1)
        predictor(1, 1, 3, False, tmpX, y, dt)

def trainModel(modelName):
    pass

def testModel():
    pass

#reduceStations()
data = pd.read_csv('./data/bike_data.csv', parse_dates=['LAST UPDATED'])
#plotData(data, 'BLESSINGTON STREET')
#plotData(data, 'KILMAINHAM GAOL')
x1, y1, dt1 = transformData(data, 'BLESSINGTON STREET')
x2, y2, dt2 = transformData(data, 'KILMAINHAM GAOL')
findUsefulFeatures(x1, y1, dt1, 'BLESSINGTON STREET')
# Remember that we are comparing two different ML approaches
#model = trainModel('')
#testModel(model)
#model = trainModel('')
#testModel(model)


### What we're looking for is the number of bikes in the station

### Now that we've got a way to get the last q instances, we need to add this to the full feature set