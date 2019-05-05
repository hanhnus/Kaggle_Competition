import time
import numpy as np 
import pandas as pd 
import gc
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
import os
import subprocess
import math
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from time import time, localtime, strftime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


df_train        = pd.DataFrame()
df_test         = pd.DataFrame()




df_train = pd.read_csv('train.csv', nrows = 5400000, index_col = "key")
df_train = df_train.dropna()
df_test  = pd.read_csv('test.csv', index_col = "key")
testdex  = df_test.index





####################################################################################
#                        Data Cleansing Function Definition                        #
####################################################################################

def data_cleansing(longitude_latitude_set,
                   flag_clean_fare_less_than_two_point_five, 
                   flag_abs_and_clean_fare_less_than_two_point_five, 
                   flag_clean_fare_count_less_equal,
                   flag_clean_itude_out_of_range, 
                   flag_clean_point_in_water):
    
    global df_train

    # Data Cleansing 1 Function
    def clean_missing_data():
        global df_train
        df_train = df_train.dropna(how = 'any', axis = 'rows')
        print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Data Cleansing 1  ', ' After drop rows with missing data :                 ', len(df_train))

    # Data Cleansing 2-1 Function
    def clean_fare_less_than(threshold):
        global df_train
        df_train = df_train[df_train.fare_amount >= threshold]
        print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Data Cleansing 2  ', ' After drop fare_amount <', threshold , 'directly:              ', len(df_train))

    # Data Cleansing 2-2 Function
    def abs_and_clean_fare_less_than(threshold):
        global df_train
        df_train['fare_amount'] = df_train.fare_amount.apply(lambda x: abs(x))   
        df_train = df_train[df_train.fare_amount >= threshold]
        print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Data Cleansing 2-2', ' After drop fare_amount ABS <', threshold , ':                  ', len(df_train))

    # Data Cleansing 3 Function
    def clean_fare_count_less_equal(flag_clean_fare_count_less_equal):
        global df_train
        df_count = pd.DataFrame(df_train.groupby('fare_amount').passenger_count.count())
        df_count.columns.values[0] = 'fare_amount_count'
        df_train = pd.merge(df_train, df_count, how = 'left', on = 'fare_amount').set_index(df_train.index)     
        del df_count
        df_train = df_train[df_train.fare_amount_count > flag_clean_fare_count_less_equal]
        df_train.drop('fare_amount_count', axis = 1, inplace = True)
        print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Data Cleansing 3  ', ' After drop fare_amount count <=', flag_clean_fare_count_less_equal, ':                 ', len(df_train))

    # Data Cleansing 4 Function
    def clean_itude_out_of_range():
        global df_train
        global longitude_min
        global longitude_range  
        global latitude_min
        global latitude_range       
        # Longitude x 118km
        # Latitude  y 144km
        if longitude_latitude_set == 1:
            longitude_min   = -74.3
            longitude_max   = -72.9
            latitude_min    =  40.5
            latitude_max    =  41.8
            longitude_range =   1.4 
            latitude_range  =   1.3

        elif longitude_latitude_set == 2:
            longitude_min   = -74.1
            longitude_max   = -73.7
            latitude_min    =  40.5
            latitude_max    =  41
            longitude_range =   0.4 
            latitude_range  =   0.5
#       print('Longitude  (x) Range: ', longitude_range, 'deg')
#       print('Latitude   (y) Range: ', latitude_range,  'deg')
        df_train = df_train[df_train.pickup_longitude  > longitude_min]
        df_train = df_train[df_train.pickup_longitude  < longitude_max]
        df_train = df_train[df_train.dropoff_longitude > longitude_min]
        df_train = df_train[df_train.dropoff_longitude < longitude_max]
        df_train = df_train[df_train.pickup_latitude   > latitude_min]
        df_train = df_train[df_train.pickup_latitude   < latitude_max]
        df_train = df_train[df_train.dropoff_latitude  > latitude_min]
        df_train = df_train[df_train.dropoff_latitude  < latitude_max]
        print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Data Cleansing 4  ', ' After drop data out of test_set coordinate range :  ', len(df_train))

    # Data Cleansing 5 Function
    def clean_pickup_dropoff_same_trip():
        global df_train
        df_train = df_train[~((df_train.pickup_longitude == df_train.dropoff_longitude) & (df_train.pickup_latitude == df_train.dropoff_latitude))]  
        print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Data Cleansing 5  ', ' After drop pickup dropoff same coordinate :         ', len(df_train))

    # Data Cleansing 6 Function
    def clean_point_in_water():
        global df_train
        def remove_datapoints_from_water():
            global df_train
            def lonlat_to_xy(longitude, latitude, dx, dy, BB):
                return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype('int'), (dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])).astype('int')
            BB                   = (-74.5, -72.8, 40.5, 41.8)
            nyc_mask             = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
            pickup_x,  pickup_y  = lonlat_to_xy(df_train.pickup_longitude,  df_train.pickup_latitude,  nyc_mask.shape[1], nyc_mask.shape[0], BB)
            dropoff_x, dropoff_y = lonlat_to_xy(df_train.dropoff_longitude, df_train.dropoff_latitude, nyc_mask.shape[1], nyc_mask.shape[0], BB)
            idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
            return df_train[idx]
        df_train = remove_datapoints_from_water()
        print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Data Cleansing 6  ', ' After drop data in water :                          ', len(df_train))

    def clean_passenger_count_oor():
        global df_train
        df_train = df_train[df_train.passenger_count  <= 6]
        df_train = df_train[df_train.passenger_count  >= 1]
        print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Data Cleansing 7  ', ' After drop passenger count <1 and >6:               ', len(df_train))



    
    clean_missing_data()                                                   # Data Cleansing 1: drop rows with missing data

    if flag_clean_fare_less_than_two_point_five == 1:                      # Data Cleansing 2: drop fare_amount < 2.5
        clean_fare_less_than(2.5)
    elif flag_abs_and_clean_fare_less_than_two_point_five == 1: 
        abs_and_clean_fare_less_than(2.5)

    if flag_clean_fare_count_less_equal > 0:                               # Data Cleansing 3: drop fare_amount count <= 1
        clean_fare_count_less_equal(flag_clean_fare_count_less_equal)
    
    if flag_clean_itude_out_of_range == 1:                                 # Data Cleansing 4: drop data out of test_set coordinate range
        clean_itude_out_of_range()
    
    clean_pickup_dropoff_same_trip()                                       # Data Cleansing 5: drop pickup dropoff same coordinate
    
    if flag_clean_point_in_water == 1:                                     # Data Cleansing 6: drop data in water
        clean_point_in_water()
    
#    clean_passenger_count_oor()

    print('-----')



####################################################################################
#                        Data Cleansing Function Definition                        #
####################################################################################




def prepare_distance_features():
    global df_train
    global df_test	
    df_train['longitude_distance'] = abs(df_train['pickup_longitude'] - df_train['dropoff_longitude'])
    df_train['latitude_distance'] = abs(df_train['pickup_latitude'] - df_train['dropoff_latitude'])
    df_train['distance_travelled'] = (df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5
    df_train['distance_travelled_sin'] = np.sin((df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5)
    df_train['distance_travelled_cos'] = np.cos((df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5)
    df_train['distance_travelled_sin_sqrd'] = np.sin((df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5) ** 2
    df_train['distance_travelled_cos_sqrd'] = np.cos((df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5) ** 2
    R = 6371e3                                                # Equator R(m)
    phi1      = np.radians(df_train['pickup_latitude' ])
    phi2      = np.radians(df_train['dropoff_latitude'])
    delta_phi = np.radians(df_train['dropoff_latitude' ] - df_train['pickup_latitude' ])
    delta_lmd = np.radians(df_train['dropoff_longitude'] - df_train['pickup_longitude'])
    a         = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lmd / 2) * np.sin(delta_lmd / 2)
    c         = 2 * np.arctan2(a ** 0.5, (1-a) ** 0.5)
    d         = R * c
    df_train['haversine']          = d                                                                       # Haversine Distance
    y         = np.sin(delta_lmd * np.cos(phi2))
    x         = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lmd)
    df_train['bearing']            = np.arctan2(y, x)    
    #------------------------------------
    df_test['longitude_distance'] = abs(df_test['pickup_longitude'] - df_test['dropoff_longitude'])
    df_test['latitude_distance'] = abs(df_test['pickup_latitude'] - df_test['dropoff_latitude'])
    df_test['distance_travelled'] = (df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5
    df_test['distance_travelled_sin'] = np.sin((df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5)
    df_test['distance_travelled_cos'] = np.cos((df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5)
    df_test['distance_travelled_sin_sqrd'] = np.sin((df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5) ** 2
    df_test['distance_travelled_cos_sqrd'] = np.cos((df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5) ** 2
    R = 6371e3                                                # Equator R(m)
    phi1      = np.radians(df_test['pickup_latitude' ])
    phi2      = np.radians(df_test['dropoff_latitude'])
    delta_phi = np.radians(df_test['dropoff_latitude' ] - df_test['pickup_latitude' ])
    delta_lmd = np.radians(df_test['dropoff_longitude'] - df_test['pickup_longitude'])
    a         = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lmd / 2) * np.sin(delta_lmd / 2)
    c         = 2 * np.arctan2(a ** 0.5, (1-a) ** 0.5)
    d         = R * c
    df_test['haversine']          = d                                                                       # Haversine Distance
    y         = np.sin(delta_lmd * np.cos(phi2))
    x         = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lmd)
    df_test['bearing']            = np.arctan2(y, x)  
    


def prepare_time_features():
    global df_train
    global df_test	
    df_train['pickup_datetime'] = df_train['pickup_datetime'].str.replace(" UTC", "")
    df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    df_train['hour_of_day']     = df_train.pickup_datetime.dt.hour
    df_train['day_of_week']     = df_train.pickup_datetime.dt.weekday
    df_train['day_of_month']    = df_train.pickup_datetime.dt.day
    df_train['day_of_year']     = df_train.pickup_datetime.dt.dayofyear
    df_train['week_of_year']    = df_train.pickup_datetime.dt.weekofyear
    df_train['month']           = df_train.pickup_datetime.dt.month
    df_train['quarter']         = df_train.pickup_datetime.dt.quarter
    df_train['year']            = df_train.pickup_datetime.dt.year
    def period_of_day(hour):
        if hour >= 6 and hour < 16:
            return 1
        elif hour >= 16 and hour < 20:
            return 2
        elif hour >= 20 or hour < 6:
            return 3
    df_train['period_of_day']   = df_train.pickup_datetime.apply(lambda t: period_of_day(t.hour))
    #------------------------------------
    df_test['pickup_datetime'] = df_test['pickup_datetime'].str.replace(" UTC", "")
    df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    df_test['hour_of_day']     = df_test.pickup_datetime.dt.hour
    df_test['day_of_week']     = df_test.pickup_datetime.dt.weekday
    df_test['day_of_month']    = df_test.pickup_datetime.dt.day
    df_test['day_of_year']     = df_test.pickup_datetime.dt.dayofyear
    df_test['week_of_year']    = df_test.pickup_datetime.dt.weekofyear
    df_test['month']           = df_test.pickup_datetime.dt.month
    df_test['quarter']         = df_test.pickup_datetime.dt.quarter
    df_test['year']            = df_test.pickup_datetime.dt.year
    def period_of_day(hour):
        if hour >= 6 and hour < 16:
            return 1
        elif hour >= 16 and hour < 20:
            return 2
        elif hour >= 20 or hour < 6:
            return 3
    df_test['period_of_day']   = df_test.pickup_datetime.apply(lambda t: period_of_day(t.hour))


def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):  
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    return distance

def airport_feats():
    longitude_min   = -74.3
    longitude_max   = -72.9
    latitude_min    =  40.5
    latitude_max    =  41.8
    longitude_range =   1.4 
    latitude_range  =   1.3
    global df_train
    global df_test		
    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    def itude_mht_dist_60_deg(x1, y1, x2, y2):
        if x1 != x2:
            k = (y1 - y2) / (x1 - x2)              
            alfa = math.degrees(math.atan(k))
        else:
            alfa = 90
        if alfa < 0:
            alfa = 180 + alfa
        d = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))                
        if alfa >= 0 and alfa < 90:
            return (d * (math.sin(math.radians(abs(alfa -60))) + math.cos(math.radians(abs(alfa - 60)))))
        elif alfa >= 90 and alfa < 180:
            return (d * (math.sin(math.radians(abs(alfa -150))) + math.cos(math.radians(abs(alfa - 150))))) 

    df_train['itude_mht_dist_60_deg'] = df_train.apply(lambda row: itude_mht_dist_60_deg(row['pickup_longitude'], row['pickup_latitude'], 
        row['dropoff_longitude'], row['dropoff_latitude'])  , axis = 1)
    df_train['mht60_dist_pkp_to_ctr'] = df_train.apply(lambda row: itude_mht_dist_60_deg(nyc[0], nyc[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
    df_train['mht60_dist_ctr_to_dpf'] = df_train.apply(lambda row: itude_mht_dist_60_deg(nyc[0], nyc[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
    df_train['mht60_dist_pkp_to_jfk'] = df_train.apply(lambda row: itude_mht_dist_60_deg(jfk[0], jfk[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
    df_train['mht60_dist_jfk_to_dpf'] = df_train.apply(lambda row: itude_mht_dist_60_deg(jfk[0], jfk[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
    df_train['mht60_dist_pkp_to_ewr'] = df_train.apply(lambda row: itude_mht_dist_60_deg(ewr[0], ewr[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
    df_train['mht60_dist_ewr_to_dpf'] = df_train.apply(lambda row: itude_mht_dist_60_deg(ewr[0], ewr[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
    df_train['mht60_dist_pkp_to_lgr'] = df_train.apply(lambda row: itude_mht_dist_60_deg(lgr[0], lgr[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
    df_train['mht60_dist_lgr_to_dpf'] = df_train.apply(lambda row: itude_mht_dist_60_deg(lgr[0], lgr[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
    def longitude_to_box(longitude, long_resolution, lat_resolution):
        long_idx = math.floor((longitude - longitude_min) / (longitude_range / long_resolution))
        return long_idx
    def latitude_to_box(latitude,    long_resolution, lat_resolution):
        lat_idx  = math.floor((latitude  - latitude_min) /  (latitude_range  / lat_resolution))
        return lat_idx
    df_train['pickup_long_blk_idx']  = df_train.apply(lambda row: longitude_to_box(row['pickup_longitude'],  600, 600), axis = 1)  
    df_train['pickup_lat_blk_idx']   = df_train.apply(lambda row: latitude_to_box(row['pickup_latitude'],    600, 600), axis = 1)
    df_train['dropoff_long_blk_idx'] = df_train.apply(lambda row: longitude_to_box(row['dropoff_longitude'], 600, 600), axis = 1)  
    df_train['dropoff_lat_blk_idx']  = df_train.apply(lambda row: latitude_to_box(row['dropoff_latitude'],   600, 600), axis = 1)




    df_test['itude_mht_dist_60_deg'] = df_test.apply(lambda row: itude_mht_dist_60_deg(row['pickup_longitude'], row['pickup_latitude'], 
        row['dropoff_longitude'], row['dropoff_latitude'])  , axis = 1)
    df_test['mht60_dist_pkp_to_ctr'] = df_test.apply(lambda row: itude_mht_dist_60_deg(nyc[0], nyc[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
    df_test['mht60_dist_ctr_to_dpf'] = df_test.apply(lambda row: itude_mht_dist_60_deg(nyc[0], nyc[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
    df_test['mht60_dist_pkp_to_jfk'] = df_test.apply(lambda row: itude_mht_dist_60_deg(jfk[0], jfk[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
    df_test['mht60_dist_jfk_to_dpf'] = df_test.apply(lambda row: itude_mht_dist_60_deg(jfk[0], jfk[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
    df_test['mht60_dist_pkp_to_ewr'] = df_test.apply(lambda row: itude_mht_dist_60_deg(ewr[0], ewr[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
    df_test['mht60_dist_ewr_to_dpf'] = df_test.apply(lambda row: itude_mht_dist_60_deg(ewr[0], ewr[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
    df_test['mht60_dist_pkp_to_lgr'] = df_test.apply(lambda row: itude_mht_dist_60_deg(lgr[0], lgr[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
    df_test['mht60_dist_lgr_to_dpf'] = df_test.apply(lambda row: itude_mht_dist_60_deg(lgr[0], lgr[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
    def longitude_to_box(longitude, long_resolution, lat_resolution):
        long_idx = math.floor((longitude - longitude_min) / (longitude_range / long_resolution))
        return long_idx
    def latitude_to_box(latitude,    long_resolution, lat_resolution):
        lat_idx  = math.floor((latitude  - latitude_min) /  (latitude_range  / lat_resolution))
        return lat_idx
    df_test['pickup_long_blk_idx']  = df_test.apply(lambda row: longitude_to_box(row['pickup_longitude'],  600, 600), axis = 1)  
    df_test['pickup_lat_blk_idx']   = df_test.apply(lambda row: latitude_to_box(row['pickup_latitude'],    600, 600), axis = 1)
    df_test['dropoff_long_blk_idx'] = df_test.apply(lambda row: longitude_to_box(row['dropoff_longitude'], 600, 600), axis = 1)  
    df_test['dropoff_lat_blk_idx']  = df_test.apply(lambda row: latitude_to_box(row['dropoff_latitude'],   600, 600), axis = 1)


    '''
    df_train['distance_to_center'] = dist(nyc[1], nyc[0], df_train['pickup_latitude'], df_train['pickup_longitude'])
    df_train['pickup_distance_to_jfk'] = dist(jfk[1], jfk[0], df_train['pickup_latitude'], df_train['pickup_longitude'])
    df_train['dropoff_distance_to_jfk'] = dist(jfk[1], jfk[0], df_train['dropoff_latitude'], df_train['dropoff_longitude'])
    df_train['pickup_distance_to_ewr'] = dist(ewr[1], ewr[0],  df_train['pickup_latitude'], df_train['pickup_longitude'])
    df_train['dropoff_distance_to_ewr'] = dist(ewr[1], ewr[0], df_train['dropoff_latitude'], df_train['dropoff_longitude'])
    df_train['pickup_distance_to_lgr'] = dist(lgr[1], lgr[0], df_train['pickup_latitude'], df_train['pickup_longitude'])
    df_train['dropoff_distance_to_lgr'] = dist(lgr[1], lgr[0], df_train['dropoff_latitude'], df_train['dropoff_longitude'])
    #----------
    df_test['distance_to_center'] = dist(nyc[1], nyc[0], df_test['pickup_latitude'], df_test['pickup_longitude'])
    df_test['pickup_distance_to_jfk'] = dist(jfk[1], jfk[0], df_test['pickup_latitude'], df_test['pickup_longitude'])
    df_test['dropoff_distance_to_jfk'] = dist(jfk[1], jfk[0], df_test['dropoff_latitude'], df_test['dropoff_longitude'])
    df_test['pickup_distance_to_ewr'] = dist(ewr[1], ewr[0],  df_test['pickup_latitude'], df_test['pickup_longitude'])
    df_test['dropoff_distance_to_ewr'] = dist(ewr[1], ewr[0], df_test['dropoff_latitude'], df_test['dropoff_longitude'])
    df_test['pickup_distance_to_lgr'] = dist(lgr[1], lgr[0], df_test['pickup_latitude'], df_test['pickup_longitude'])
    df_test['dropoff_distance_to_lgr'] = dist(lgr[1], lgr[0], df_test['dropoff_latitude'], df_test['dropoff_longitude'])
    '''


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

# Build ime Aggregate Features
def time_agg(vars_to_agg, vars_be_agg):
    global df_train
    global df_test	
    for var in vars_to_agg:
        agg = df_train.groupby(var)[vars_be_agg].agg(["sum","mean","std","skew",percentile(80),percentile(20)])
        if isinstance(var, list):
            agg.columns = pd.Index(["fare_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
        else:
            agg.columns = pd.Index(["fare_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 
        df_train = pd.merge(df_train,agg, on=var, how= "left")
        df_test = pd.merge(df_test,agg, on=var, how= "left")
    


# Clean dataset from https://www.kaggle.com/gunbl4d3/xgboost-ing-taxi-fares
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45)]
print("Cleaning Functions Defined..")



print("Percent of Training Set with Zero and Below Fair: ", round(((df_train.loc[df_train["fare_amount"] <= 0, "fare_amount"].shape[0]/df_train.shape[0]) * 100),5))
print("Percent of Training Set 200 and Above Fair: ", round((df_train.loc[df_train["fare_amount"] >= 200, "fare_amount"].shape[0]/df_train.shape[0]) * 100,5))
df_train = df_train.loc[(df_train["fare_amount"] > 0) & (df_train["fare_amount"] <= 200),:]
print("\nPercent of Training Set with Zero and Below Passenger Count: ", round((df_train.loc[df_train["passenger_count"] <= 0, "passenger_count"].shape[0]/df_train.shape[0]) * 100,5))
print("Percent of Training Set with Nine and Above Passenger Count: ", round((df_train.loc[df_train["passenger_count"] >= 9, "passenger_count"].shape[0]/df_train.shape[0]) * 100,5))
df_train = df_train.loc[(df_train["passenger_count"] > 0) & (df_train["passenger_count"] <= 9),:]

# Clean Training Set
df_train = clean_df(df_train)



#data_cleansing(1, 1, 0, 1, 1, 1)       # parameter1      : longitude_latitude_set (1 - large box; 2 - small box)
                                       # parameter2      : flag_clean_fare_less_than_two_point_five
                                       # parameter3      : flag_abs_and_clean_fare_less_than_two_point_five
                                       # parameter4      : flag_clean_fare_count_less_equal
                                       # parameter5      : flag_clean_itude_out_of_range
                                       # parameter6      : flag_clean_point_in_water



# Distance Features
prepare_distance_features()
airport_feats()

# Time Features
prepare_time_features()


# Ratios
df_train["fare_to_dist_ratio"] = df_train["fare_amount"] / ( df_train["distance_travelled"]+0.0001)
df_train["fare_npassenger_to_dist_ratio"] = (df_train["fare_amount"] / df_train["passenger_count"]) /( df_train["distance_travelled"]+0.0001)

# Time Aggregate Features
time_agg(vars_to_agg  = ["passenger_count", 'day_of_week', "quarter", "month", "year", "hour_of_day",
                                          ['day_of_week', "month", "year"], ["hour_of_day", 'day_of_week', "month", "year"]],
                          vars_be_agg = "fare_amount")


train_time_start = df_train.pickup_datetime.min()
train_time_end = df_train.pickup_datetime.max()
print("Train Time Starts: {}, Ends {}".format(train_time_start,train_time_end))
test_time_start = df_test.pickup_datetime.min()
test_time_end = df_test.pickup_datetime.max()
print("Test Time Starts: {}, Ends {}".format(test_time_start,test_time_end))

# Keep Relevant Variables..
y          = df_train.fare_amount.copy()
df_test.drop("pickup_datetime", axis = 1, inplace=True)
df_train   = df_train[df_test.columns]
print("Does Train feature equal test feature?: ", all(df_train.columns == df_test.columns))
trainshape = df_train.shape
testshape  = df_test.shape

# LGBM Dataset Formating
training_data = lgb.Dataset(df_train, label=y, free_raw_data=False)

print("Light Gradient Boosting Regressor: ")
lgbm_params =  {'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse'}

folds              = KFold(n_splits = 5, shuffle = True, random_state = 1)
test_set_predict   = np.zeros(testshape[0])
test_split_predict = np.zeros(trainshape[0])
training_data.construct()

# Fit 5 Folds
#modelstart = time.time()
for trn_idx, val_idx in folds.split(df_train):
    LightGBM_model = lgb.train(
        params                = lgbm_params,
        train_set             = training_data.subset(trn_idx),
        valid_sets            = training_data.subset(val_idx),
        num_boost_round       = 4000, 
        early_stopping_rounds = 125,
        verbose_eval          = 500
    )
    test_split_predict[val_idx] = LightGBM_model.predict(training_data.data.iloc[val_idx])
    test_set_predict += LightGBM_model.predict(df_test) / folds.n_splits
    print(mean_squared_error(y.iloc[val_idx], test_split_predict[val_idx]) ** .5)

submission = pd.DataFrame(test_set_predict,columns=["fare_amount"],index=testdex)
submission.to_csv("LightGBM_5400000rows.csv",index=True,header=True)














