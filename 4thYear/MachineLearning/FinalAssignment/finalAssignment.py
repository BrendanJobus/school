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