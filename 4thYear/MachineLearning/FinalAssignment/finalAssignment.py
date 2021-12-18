import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

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
    reducedData.to_csv('./data/bike_data.csv')

# Need to transform the time into unix timestamp in seconds
def timeTransformer():
    pass

def predictor(feautres, tragets):
    pass

def findUsefulFeatures(data):
    # We're going to run a basic linear regression on this, seeing which of the variables are actually used, and then removing different features and seing what effect that has
    targets = data['AVAILABLE BIKE STANDS']
    features = data.drop(['AVAILABLE BIKE STANDS'], axis=1)

    predictor(features, targets)

def trainModel(modelName):
    pass

def testModel():
    pass

#reduceStations()
data = pd.read_csv('./data/bike_data.csv')
findUsefulFeatures(data)
# Remember that we are comparing two different ML approaches
#model = trainModel('')
#testModel(model)
#model = trainModel('')
#testModel(model)