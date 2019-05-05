import time
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

################################## Parameters ##################################





####################################################################################
#                         Data Reading Function Definition                         #
####################################################################################

def training_data_reading(read_rows):
	global df_train
	print('Training Set Data Reading Start: ', strftime('%Y-%m-%d %H:%M:%S', localtime()))
	if read_rows > 0:
		# read 2 million rows of raw data		
		print('Reading', read_rows, 'rows...')
		df_train =  pd.read_csv('train.csv', nrows = read_rows, index_col = "key")
		df_train['pickup_datetime'] = df_train['pickup_datetime'].str.slice(0, 16)
		df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'], utc = True, format = '%Y-%m-%d %H:%M')
		print("info:")
		print(df_train.info())
		print("describe:")
		print(df_train.describe())
		print("isnull.sum:")
		print(df_train.isnull().sum())
		print(df_train.head())
		print(df_train.tail())

	elif read_rows == 0:
		# read all raw data
		print('Reading WHOLE dataset...')   
		df_train = dd.read_csv('train.csv').compute()
		df_train['pickup_datetime'] = df_train['pickup_datetime'].str.slice(0, 16)
		df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'], utc = True, format = '%Y-%m-%d %H:%M')
		df_train.index = pd.RangeIndex(start = 0, stop = len(df_train))
		print(df_train.info())
		print(df_train.describe())
		print(df_train.isnull().sum())
		print(df_train.head())
		print(df_train.tail())

	else:
		print('Wrong data reading information input!')
		os._exit()

	print('Training Set Data Reading Finish:', strftime('%Y-%m-%d %H:%M:%S', localtime()))
	print('----------')


def test_data_reading():
	global df_test
	print('Test Set Data Reading Start: ', strftime('%Y-%m-%d %H:%M:%S', localtime()))
	df_test =  pd.read_csv('test.csv', index_col = "key")
	df_test['pickup_datetime'] = df_test['pickup_datetime'].str.slice(0, 16)
	df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'], utc = True, format = '%Y-%m-%d %H:%M')
	print(df_test.info())
	print(df_test.describe())
	print(df_test.head())
	print(df_test.tail())
	print('Test Set Data Reading Finish:', strftime('%Y-%m-%d %H:%M:%S', localtime()))
	print('----------')	

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
		print('Longitude  (x) Range: ', longitude_range, 'deg')
		print('Latitude   (y) Range: ', latitude_range,  'deg')
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
	
	clean_passenger_count_oor()

	print('-----')

####################################################################################
#                     Feature Manipulation Function Definition                     #
####################################################################################


def feature_manipulation_trainingset():
	global df_train

	def add_time_features():
		global df_train
		df_train['pickup_datetime'] = df_train['pickup_datetime'].replace(' UTC', '')
		df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
		df_train['hour_of_day']     = df_train.pickup_datetime.dt.hour
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   1 added:', ' hour_of_day')
		df_train['day_of_week']     = df_train.pickup_datetime.dt.weekday
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   2 added:', ' day_of_week')
		df_train['day_of_month']    = df_train.pickup_datetime.dt.day
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   3 added:', ' day_of_month')
		df_train['day_of_year']     = df_train.pickup_datetime.dt.dayofyear
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   4 added:', ' day_of_year')
		df_train['week_of_year']    = df_train.pickup_datetime.dt.weekofyear
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   5 added:', ' week_of_year')
		df_train['month']           = df_train.pickup_datetime.dt.month
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   6 added:', ' month')
		df_train['quarter']         = df_train.pickup_datetime.dt.quarter
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   7 added:', ' quarter')
		df_train['year']            = df_train.pickup_datetime.dt.year
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   8 added:', ' year')
		def period_of_day(hour):
			if hour >= 6 and hour < 16:
				return 1
			elif hour >= 16 and hour < 20:
				return 2
			elif hour >= 20 or hour < 6:
				return 3
		df_train['period_of_day']   = df_train.pickup_datetime.apply(lambda t: period_of_day(t.hour))
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   9 added:', ' period_of_day')
		df_train.drop('pickup_datetime', axis=1, inplace=True)
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' pickup_datetime dropped')
		print('-----')

	def add_distance_features():
		global df_train
		global longitude_min
		global longitude_range  
		global latitude_min
		global latitude_range
		df_train['longitude_distance']            =     abs(df_train['pickup_longitude'  ]   -    df_train['dropoff_longitude'])                    
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   1 added:', ' longitude_distance')
		df_train['latitude_distance' ]            =     abs(df_train['pickup_latitude'   ]   -    df_train['dropoff_latitude' ])                   
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   2 added:', ' latitude_distance')
		df_train['itude_dist']          =        (df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5         
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   3 added:', ' itude_dist')
		R = 6371e3                                                # Equator R(m)
		phi1      = np.radians(df_train['pickup_latitude' ])
		phi2      = np.radians(df_train['dropoff_latitude'])
		delta_phi = np.radians(df_train['dropoff_latitude' ] - df_train['pickup_latitude' ])
		delta_lmd = np.radians(df_train['dropoff_longitude'] - df_train['pickup_longitude'])
		a         = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lmd / 2) * np.sin(delta_lmd / 2)
		c         = 2 * np.arctan2(a ** 0.5, (1-a) ** 0.5)
		d         = R * c
		df_train['haversine']          = d                                                                       # Haversine
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   4 added:', ' haversine')
		y         = np.sin(delta_lmd * np.cos(phi2))
		x         = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lmd)
		df_train['bearing']            = np.degrees(np.arctan2(y, x))                                                          # Bearing
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   5 added:', ' bearing')

		if box == 1:
			def longitude_to_box(longitude, long_resolution, lat_resolution):
				long_idx = math.floor((longitude - longitude_min) / (longitude_range / long_resolution))
				return long_idx
			def latitude_to_box(latitude,    long_resolution, lat_resolution):
				lat_idx  = math.floor((latitude  - latitude_min) /  (latitude_range  / lat_resolution))
				return lat_idx
			df_train['pickup_long_blk_idx']  = df_train.apply(lambda row: longitude_to_box(row['pickup_longitude'],  600, 600), axis = 1)  
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   6 added:', ' pickup_long_blk_idx')          
			df_train['pickup_lat_blk_idx']   = df_train.apply(lambda row: latitude_to_box(row['pickup_latitude'],    600, 600), axis = 1)
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   7 added:', ' pickup_lat_blk_idx') 
			df_train['dropoff_long_blk_idx'] = df_train.apply(lambda row: longitude_to_box(row['dropoff_longitude'], 600, 600), axis = 1)  
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   8 added:', ' dropoff_long_blk_idx') 
			df_train['dropoff_lat_blk_idx']  = df_train.apply(lambda row: latitude_to_box(row['dropoff_latitude'],   600, 600), axis = 1)
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   9 added:', ' dropoff_lat_blk_idx') 

		if sin_cos == 1:
			df_train['itude_dist_sin']      = np.sin((df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5)        # sin()
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  10 added:', ' itude_dist_sin')
			df_train['itude_dist_cos']      = np.cos((df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5)        # cos()
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  11 added:', ' itude_dist_cos')
			df_train['itude_dist_sin_sqrd'] = np.sin((df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5) ** 2   # sin平方()
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  12 added:', ' itude_dist_sin_sqrd')
			df_train['itude_dist_cos_sqrd'] = np.cos((df_train['longitude_distance'] ** 2 + df_train['latitude_distance'] ** 2) ** .5) ** 2   # cos平方()
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  13 added:', ' itude_dist_cos_sqrd')
		
		if mht_d == 1:
			df_train['itude_mht_distance'] = df_train.apply(lambda row: (np.abs(row['dropoff_latitude'] - row['pickup_latitude']) +
				np.abs(row['dropoff_longitude'] - row['pickup_longitude'])), axis = 1)  
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  14 added:', ' itude_mht_distance')

		if mht_d60 == 1:
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
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  15 added:', ' itude_mht_dist_60_deg')







		###
		def add_popular_zone_and_distance_features():
			global df_train
			nyc = (-74.0063889,    40.7141667)
			jfk = (-73.786578,     40.64879)
			ewr = (-74.178894,     40.6928)
			lgr = (-73.872825,     40.77387)
			def popular_zones(x, y):
				def within_manhattan(x, y):
					if ((y < x * 1.929503    + 183.553291) & \
					    (y > x * 0.215615    +  56.657966) & \
					    (y > x * 8.238748    + 650.131497) & \
					    (y > x * 1.387465    + 143.369265) & \
					    (y < x * (-5.700678) - 380.638322)) == True:
						return True
					else:
						return False
				def itude_dist(x1, y1, x2, y2):
					return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
				if within_manhattan(x, y) == True:
					return 0
				elif itude_dist(x, y, jfk[0], jfk[1]) < 0.0437:
					return 1
				elif itude_dist(x, y, ewr[0], ewr[1]) < 0.0421:
					return 2
				elif itude_dist(x, y, lgr[0], lgr[1]) < 0.0187:
					return 3
				else:
					return 4
			
			df_train['pickup_popular_zone' ] = df_train.apply(lambda row: popular_zones(row['pickup_longitude' ], row['pickup_latitude' ]), axis = 1)
			print('pickup_popular_zone')
			df_train['dropoff_popular_zone'] = df_train.apply(lambda row: popular_zones(row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
			print('dropoff_popular_zone')
			def trip_route(pickup_zone, dropoff_zone):
				return pickup_zone * 5 + dropoff_zone
			df_train['trip_route'          ] = df_train.apply(lambda row: trip_route(row['pickup_popular_zone'], row['dropoff_popular_zone']), axis = 1)
			print('trip_route')
			df_train.drop('pickup_popular_zone', axis = 1, inplace = True)
			df_train.drop('dropoff_popular_zone', axis = 1, inplace = True)
		###
			



			def mht_dist(pickup_long, pickup_lat, dropoff_long, dropoff_lat):
				distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)		    
				return distance
			
			if mht_d == 1:
				df_train['mht_dist_pkp_to_ctr'] = mht_dist(nyc[0], nyc[1], df_train['pickup_longitude' ], df_train['pickup_latitude' ])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  16 added:', ' mht_dist_pkp_to_ctr') 
				df_train['mht_dist_ctr_to_dpf'] = mht_dist(nyc[0], nyc[1], df_train['dropoff_longitude'], df_train['dropoff_latitude'])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  17 added:', ' mht_dist_ctr_to_dpf') 
				df_train['mht_dist_pkp_to_jfk'] = mht_dist(jfk[0], jfk[1], df_train['pickup_longitude' ], df_train['pickup_latitude' ])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  18 added:', ' mht_dist_pkp_to_jfk') 
				df_train['mht_dist_jfk_to_dpf'] = mht_dist(jfk[0], jfk[1], df_train['dropoff_longitude'], df_train['dropoff_latitude'])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  19 added:', ' mht_dist_jfk_to_dpf') 
				df_train['mht_dist_pkp_to_ewr'] = mht_dist(ewr[0], ewr[1], df_train['pickup_longitude' ], df_train['pickup_latitude' ])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  20 added:', ' mht_dist_pkp_to_ewr') 
				df_train['mht_dist_ewr_to_dpf'] = mht_dist(ewr[0], ewr[1], df_train['dropoff_longitude'], df_train['dropoff_latitude'])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  21 added:', ' mht_dist_ewr_to_dpf') 
				df_train['mht_dist_pkp_to_lgr'] = mht_dist(lgr[0], lgr[1], df_train['pickup_longitude' ], df_train['pickup_latitude' ])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  22 added:', ' mht_dist_pkp_to_lgr') 
				df_train['mht_dist_lgr_to_dpf'] = mht_dist(lgr[0], lgr[1], df_train['dropoff_longitude'], df_train['dropoff_latitude'])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  23 added:', ' mht_dist_lgr_to_dpf') 
			
			if mht_d60 == 1:
				df_train['mht60_dist_pkp_to_ctr'] = df_train.apply(lambda row: itude_mht_dist_60_deg(nyc[0], nyc[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  24 added:', ' mht60_dist_pkp_to_ctr') 
				df_train['mht60_dist_ctr_to_dpf'] = df_train.apply(lambda row: itude_mht_dist_60_deg(nyc[0], nyc[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  25 added:', ' mht60_dist_ctr_to_dpf') 
				df_train['mht60_dist_pkp_to_jfk'] = df_train.apply(lambda row: itude_mht_dist_60_deg(jfk[0], jfk[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  26 added:', ' mht60_dist_pkp_to_jfk') 
				df_train['mht60_dist_jfk_to_dpf'] = df_train.apply(lambda row: itude_mht_dist_60_deg(jfk[0], jfk[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  27 added:', ' mht60_dist_jfk_to_dpf') 
				df_train['mht60_dist_pkp_to_ewr'] = df_train.apply(lambda row: itude_mht_dist_60_deg(ewr[0], ewr[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  28 added:', ' mht60_dist_pkp_to_ewr') 
				df_train['mht60_dist_ewr_to_dpf'] = df_train.apply(lambda row: itude_mht_dist_60_deg(ewr[0], ewr[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  29 added:', ' mht60_dist_ewr_to_dpf') 
				df_train['mht60_dist_pkp_to_lgr'] = df_train.apply(lambda row: itude_mht_dist_60_deg(lgr[0], lgr[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  30 added:', ' mht60_dist_pkp_to_lgr') 
				df_train['mht60_dist_lgr_to_dpf'] = df_train.apply(lambda row: itude_mht_dist_60_deg(lgr[0], lgr[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  31 added:', ' mht60_dist_lgr_to_dpf') 
		add_popular_zone_and_distance_features()
		print('-----')

	add_time_features()
	add_distance_features()








#######################################################
# A better large df transfer machanism should be find out later
#########################################################


def feature_manipulation_testset():
	global df_test

	def add_time_features():
		global df_test
		df_test['pickup_datetime'] = df_test['pickup_datetime'].replace(' UTC', '')
		df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
		df_test['hour_of_day']     = df_test.pickup_datetime.dt.hour
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   1 added:', ' hour_of_day')
		df_test['day_of_week']     = df_test.pickup_datetime.dt.weekday
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   2 added:', ' day_of_week')
		df_test['day_of_month']    = df_test.pickup_datetime.dt.day
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   3 added:', ' day_of_month')
		df_test['day_of_year']     = df_test.pickup_datetime.dt.dayofyear
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   4 added:', ' day_of_year')
		df_test['week_of_year']    = df_test.pickup_datetime.dt.weekofyear
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   5 added:', ' week_of_year')
		df_test['month']           = df_test.pickup_datetime.dt.month
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   6 added:', ' month')
		df_test['quarter']         = df_test.pickup_datetime.dt.quarter
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   7 added:', ' quarter')
		df_test['year']            = df_test.pickup_datetime.dt.year
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   8 added:', ' year')
		def period_of_day(hour):
			if hour >= 6 and hour < 16:
				return 1
			elif hour >= 16 and hour < 20:
				return 2
			elif hour >= 20 or hour < 6:
				return 3
		df_test['period_of_day']   = df_test.pickup_datetime.apply(lambda t: period_of_day(t.hour))
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Time Feature   9 added:', ' period_of_day')
		df_test.drop('pickup_datetime', axis=1, inplace=True)
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' pickup_datetime dropped')
		print('-----')

	def add_distance_features():
		global df_test
		global longitude_min
		global longitude_range  
		global latitude_min
		global latitude_range
		df_test['longitude_distance']            =     abs(df_test['pickup_longitude'  ]   -    df_test['dropoff_longitude'])                    
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   1 added:', ' longitude_distance')
		df_test['latitude_distance' ]            =     abs(df_test['pickup_latitude'   ]   -    df_test['dropoff_latitude' ])                    
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   2 added:', ' latitude_distance')
		df_test['itude_dist']          =        (df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5        
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   3 added:', ' itude_dist')
		R = 6371e3                                                # 赤道半径(m)
		phi1      = np.radians(df_test['pickup_latitude' ])
		phi2      = np.radians(df_test['dropoff_latitude'])
		delta_phi = np.radians(df_test['dropoff_latitude' ] - df_test['pickup_latitude' ])
		delta_lmd = np.radians(df_test['dropoff_longitude'] - df_test['pickup_longitude'])
		a         = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lmd / 2) * np.sin(delta_lmd / 2)
		c         = 2 * np.arctan2(a ** 0.5, (1-a) ** 0.5)
		d         = R * c
		df_test['haversine']          = d                                                                       # Haversine
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   4 added:', ' haversine')
		y         = np.sin(delta_lmd * np.cos(phi2))
		x         = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lmd)
		df_test['bearing']            = np.degrees(np.arctan2(y, x))                                                          # Bearing
		print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   5 added:', ' bearing')

		if box == 1:
			def longitude_to_box(longitude, long_resolution, lat_resolution):
				long_idx = math.floor((longitude - longitude_min) / (longitude_range / long_resolution))
				return long_idx
			def latitude_to_box(latitude,    long_resolution, lat_resolution):
				lat_idx  = math.floor((latitude  - latitude_min) /  (latitude_range  / lat_resolution))
				return lat_idx
			df_test['pickup_long_blk_idx']  = df_test.apply(lambda row: longitude_to_box(row['pickup_longitude'],  600, 600), axis = 1)  
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   6 added:', ' pickup_long_blk_idx')          
			df_test['pickup_lat_blk_idx']   = df_test.apply(lambda row: latitude_to_box(row['pickup_latitude'],    600, 600), axis = 1)
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   7 added:', ' pickup_lat_blk_idx') 
			df_test['dropoff_long_blk_idx'] = df_test.apply(lambda row: longitude_to_box(row['dropoff_longitude'], 600, 600), axis = 1)  
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   8 added:', ' dropoff_long_blk_idx') 
			df_test['dropoff_lat_blk_idx']  = df_test.apply(lambda row: latitude_to_box(row['dropoff_latitude'],   600, 600), axis = 1)
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature   9 added:', ' dropoff_lat_blk_idx') 

		if sin_cos == 1:
			df_test['itude_dist_sin']      = np.sin((df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5)        # sin()
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  10 added:', ' itude_dist_sin')
			df_test['itude_dist_cos']      = np.cos((df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5)        # cos()
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  11 added:', ' itude_dist_cos')
			df_test['itude_dist_sin_sqrd'] = np.sin((df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5) ** 2   # sin2()
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  12 added:', ' itude_dist_sin_sqrd')
			df_test['itude_dist_cos_sqrd'] = np.cos((df_test['longitude_distance'] ** 2 + df_test['latitude_distance'] ** 2) ** .5) ** 2   # cos2()
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  13 added:', ' itude_dist_cos_sqrd')

		if mht_d == 1:
			df_test['itude_mht_distance'] = df_test.apply(lambda row: (np.abs(row['dropoff_latitude'] - row['pickup_latitude']) +
				np.abs(row['dropoff_longitude'] - row['pickup_longitude'])), axis = 1)  
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  14 added:', ' itude_mht_distance')
		
		if mht_d60 == 1:
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
			df_test['itude_mht_dist_60_deg'] = df_test.apply(lambda row: itude_mht_dist_60_deg(row['pickup_longitude'], row['pickup_latitude'], 
				row['dropoff_longitude'], row['dropoff_latitude'])  , axis = 1)
			print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  15 added:', ' itude_mht_dist_60_deg')



		###
		def add_popular_zone_and_distance_features():
			global df_test
			nyc = (-74.0063889,    40.7141667)
			jfk = (-73.786578,     40.64879)
			ewr = (-74.178894,     40.6928)
			lgr = (-73.872825,     40.77387)
			def popular_zones(x, y):
				def within_manhattan(x, y):
					if ((y < x * 1.929503    + 183.553291) & \
					    (y > x * 0.215615    +  56.657966) & \
					    (y > x * 8.238748    + 650.131497) & \
					    (y > x * 1.387465    + 143.369265) & \
					    (y < x * (-5.700678) - 380.638322)) == True:
						return True
					else:
						return False
				def itude_dist(x1, y1, x2, y2):
					return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
				if within_manhattan(x, y) == True:
					return 0
				elif itude_dist(x, y, jfk[0], jfk[1]) < 0.0437:
					return 1
				elif itude_dist(x, y, ewr[0], ewr[1]) < 0.0421:
					return 2
				elif itude_dist(x, y, lgr[0], lgr[1]) < 0.0187:
					return 3
				else:
					return 4
#			df_test['pickup_popular_zone' ] = df_test.apply(lambda row: popular_zones(row['pickup_longitude' ], row['pickup_latitude' ]), axis = 1)
#			print('pickup_popular_zone')
#			df_test['dropoff_popular_zone'] = df_test.apply(lambda row: popular_zones(row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
#			print('dropoff_popular_zone')
###			def trip_route(pickup_zone, dropoff_zone):
###				return pickup_zone * 5 + dropoff_zone
###			df_test['trip_route'          ] = df_test.apply(lambda row: trip_route(row['pickup_popular_zone'], row['dropoff_popular_zone']), axis = 1)
###			print('trip_route')

		###




			def mht_dist(pickup_long, pickup_lat, dropoff_long, dropoff_lat):
				distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)		    
				return distance

			if mht_d == 1:
				df_test['mht_dist_pkp_to_ctr'] = mht_dist(nyc[0], nyc[1], df_test['pickup_longitude' ], df_test['pickup_latitude' ])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  16 added:', ' mht_dist_pkp_to_ctr') 
				df_test['mht_dist_ctr_to_dpf'] = mht_dist(nyc[0], nyc[1], df_test['dropoff_longitude'], df_test['dropoff_latitude'])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  17 added:', ' mht_dist_ctr_to_dpf') 
				df_test['mht_dist_pkp_to_jfk'] = mht_dist(jfk[0], jfk[1], df_test['pickup_longitude' ], df_test['pickup_latitude' ])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  18 added:', ' mht_dist_pkp_to_jfk') 
				df_test['mht_dist_jfk_to_dpf'] = mht_dist(jfk[0], jfk[1], df_test['dropoff_longitude'], df_test['dropoff_latitude'])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  19 added:', ' mht_dist_jfk_to_dpf') 
				df_test['mht_dist_pkp_to_ewr'] = mht_dist(ewr[0], ewr[1], df_test['pickup_longitude' ], df_test['pickup_latitude' ])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  20 added:', ' mht_dist_pkp_to_ewr') 
				df_test['mht_dist_ewr_to_dpf'] = mht_dist(ewr[0], ewr[1], df_test['dropoff_longitude'], df_test['dropoff_latitude'])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  21 added:', ' mht_dist_ewr_to_dpf') 
				df_test['mht_dist_pkp_to_lgr'] = mht_dist(lgr[0], lgr[1], df_test['pickup_longitude' ], df_test['pickup_latitude' ])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  22 added:', ' mht_dist_pkp_to_lgr') 
				df_test['mht_dist_lgr_to_dpf'] = mht_dist(lgr[0], lgr[1], df_test['dropoff_longitude'], df_test['dropoff_latitude'])
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  23 added:', ' mht_dist_lgr_to_dpf') 
			
			if mht_d60 == 1:
				df_test['mht60_dist_pkp_to_ctr'] = df_test.apply(lambda row: itude_mht_dist_60_deg(nyc[0], nyc[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  24 added:', ' mht60_dist_pkp_to_ctr') 
				df_test['mht60_dist_ctr_to_dpf'] = df_test.apply(lambda row: itude_mht_dist_60_deg(nyc[0], nyc[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  25 added:', ' mht60_dist_ctr_to_dpf') 
				df_test['mht60_dist_pkp_to_jfk'] = df_test.apply(lambda row: itude_mht_dist_60_deg(jfk[0], jfk[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  26 added:', ' mht60_dist_pkp_to_jfk') 
				df_test['mht60_dist_jfk_to_dpf'] = df_test.apply(lambda row: itude_mht_dist_60_deg(jfk[0], jfk[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  27 added:', ' mht60_dist_jfk_to_dpf') 
				df_test['mht60_dist_pkp_to_ewr'] = df_test.apply(lambda row: itude_mht_dist_60_deg(ewr[0], ewr[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  28 added:', ' mht60_dist_pkp_to_ewr') 
				df_test['mht60_dist_ewr_to_dpf'] = df_test.apply(lambda row: itude_mht_dist_60_deg(ewr[0], ewr[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  29 added:', ' mht60_dist_ewr_to_dpf') 
				df_test['mht60_dist_pkp_to_lgr'] = df_test.apply(lambda row: itude_mht_dist_60_deg(lgr[0], lgr[1], row['pickup_longitude'],  row['pickup_latitude']),  axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  30 added:', ' mht60_dist_pkp_to_lgr') 
				df_test['mht60_dist_lgr_to_dpf'] = df_test.apply(lambda row: itude_mht_dist_60_deg(lgr[0], lgr[1], row['dropoff_longitude'], row['dropoff_latitude']), axis = 1)
				print(strftime('%Y-%m-%d %H:%M:%S', localtime()), ' Dist Feature  31 added:', ' mht60_dist_lgr_to_dpf') 
			



		add_popular_zone_and_distance_features()
		print('-----')

	add_time_features()
	add_distance_features()


#######################################################
# A better large df transfer machanism should be find out later
#########################################################


def feature_manipulation_for_both_sets():
	global df_train
	global df_test
	# Percentile
	def percentile(n):
	    def percentile_(x):
	        return np.percentile(x, n)
	    percentile_.__name__ = 'percentile_%s' % n
	    return percentile_

	# Build Time Aggregate Features
	def time_agg(vars_to_agg, vars_be_agg):
		global df_train
		global df_test
		for var in vars_to_agg:
			agg = df_train.groupby(var)[vars_be_agg].agg(["sum","mean","std","skew",percentile(80),percentile(20)])
			if isinstance(var, list):
				agg.columns = pd.Index(["fare_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
			else:
				agg.columns = pd.Index(["fare_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 
			df_train = pd.merge(df_train, agg, on=var, how= "left").set_index(df_train.index)
			df_test  = pd.merge(df_test,  agg, on=var, how= "left").set_index(df_test.index) 


	time_agg(vars_to_agg = ["passenger_count",                                # Group 1
							"day_of_week",                                    # Group 2
							"quarter",                                        # Group 3
							"month",                                          # Group 4
							"year",                                           # Group 5
							"hour_of_day",                                    # Group 6
							["day_of_week", "month", "year"],                 # Group 7
							["hour_of_day", "day_of_week", "month", "year"]   # Group 8
						   ],
			 vars_be_agg = "fare_amount")






####################################################################################
#                      Data Visualization Function Definition                      #
####################################################################################

def data_visualization():

	def plot_training_set_distribution():
		df_train[(df_train.fare_amount <= 200) & (df_train.fare_amount >= 0)].fare_amount.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Fare Amount ($USD)')
		plt.title('Distribution - Fare Amount ($USD)')
		plt.show()
		df_train[(df_train.pickup_longitude <= -72.9) & (df_train.pickup_longitude >= -74.3)].pickup_longitude.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Pickup Longitude')
		plt.title('Distribution - Pickup Longitude')
		plt.show()
		df_train[(df_train.pickup_latitude <= 41.8) & (df_train.pickup_latitude >= 40.5)].pickup_latitude.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Pickup Latitude')
		plt.title('Distribution - Pickup Latitude')
		plt.show()
		df_train[(df_train.dropoff_longitude <= -72.9) & (df_train.dropoff_longitude >= -74.3)].dropoff_longitude.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Dropoff Longitude')
		plt.title('Distribution - Dropoff Longitude')
		plt.show()
		df_train[(df_train.dropoff_latitude <= 41.8) & (df_train.dropoff_latitude >= 40.5)].dropoff_latitude.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Dropoff Latitude')
		plt.title('Distribution - Dropoff Latitude')
		plt.show()
		df_train[(df_train.passenger_count <= 6) & (df_train.passenger_count >= 0)].passenger_count.hist(bins = 100, figsize = (14, 3))
		plt.xlabel('Passenger Count')
		plt.title('Distribution - Passenger Count')
		plt.show()

	def plot_test_set_distribution():
		df_test.pickup_longitude.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Pickup Longitude')
		plt.title('Distribution - Pickup Longitude')
		plt.show()
		df_test.pickup_latitude.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Pickup Latitude')
		plt.title('Distribution - Pickup Latitude')
		plt.show()
		df_test.dropoff_longitude.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Dropoff Longitude')
		plt.title('Distribution - Dropoff Longitude')
		plt.show()
		df_test.dropoff_latitude.hist(bins = 1000, figsize = (14, 3))
		plt.xlabel('Dropoff Latitude')
		plt.title('Distribution - Dropoff Latitude')
		plt.show()
		df_test.passenger_count.hist(bins = 100, figsize = (14, 3))
		plt.xlabel('Passenger Count')
		plt.title('Distribution - Passenger Count')
		plt.show()

	def plot_hires(df, BB, figsize=(12, 12), ax=None, c=('r', 'b')):
	    def select_within_boundingbox(df, BB):
		    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
		           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
		           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
		           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
	    if ax == None:
	        fig, ax = plt.subplots(1, 1, figsize=figsize)
	    idx = select_within_boundingbox(df, BB)
	    ax.scatter(df[idx].pickup_longitude, df[idx].pickup_latitude, c=c[0], s=0.01, alpha=0.5)
	    ax.scatter(df[idx].dropoff_longitude, df[idx].dropoff_latitude, c=c[1], s=0.01, alpha=0.5)
	    plt.show()

	# Download NYC Map
	BB = (-74.5, -72.8, 40.5, 41.8)
	#nyc_map = plt.imread('https://aiblog.nl/download/nyc_-74.5_-72.8_40.5_41.8.png')
	nyc_map = plt.imread('nyc_-74.5_-72.8_40.5_41.8.png')
	# Download NYC Map (Zoom-in)
	BB_zoom = (-74.3, -73.7, 40.5, 40.9)
	#nyc_map_zoom = plt.imread('https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png')
	nyc_map_zoom = plt.imread('nyc_-74.3_-73.7_40.5_40.9.png')
	# this function will be used more often to plot data on the NYC map
	def plot_on_map(df, BB, nyc_map, s=10, alpha=0.2):
	    fig, axs = plt.subplots(1, 2, figsize=(16,10))
	    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=alpha, c='r', s=s)
	    axs[0].set_xlim((BB[0], BB[1]))
	    axs[0].set_ylim((BB[2], BB[3]))
	    axs[0].set_title('Pickup locations')
	    axs[0].imshow(nyc_map, zorder=0, extent=BB)

	    axs[1].scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=alpha, c='r', s=s)
	    axs[1].set_xlim((BB[0], BB[1]))
	    axs[1].set_ylim((BB[2], BB[3]))
	    axs[1].set_title('Dropoff locations')
	    axs[1].imshow(nyc_map, zorder=0, extent=BB)
	    plt.show()
	# plot training data on map
	plot_on_map(df_train, BB, nyc_map, s=1, alpha=0.3)
	# plot training data on map zoomed in
	plot_on_map(df_train, BB_zoom, nyc_map_zoom, s=1, alpha=0.3)
	# plot test data on map
	plot_on_map(df_test, BB, nyc_map, alpha=1.0, s=1)






#	plot_training_set_distribution()
#	plot_test_set_distribution()
#	plot_hires(df_train, (-74.2, -73.5, 40.5, 41.2))
#	plot_hires(df_test, (-74.2, -73.7, 40.5, 41))







####################################################################################
#                             Main Function Definition                             #
####################################################################################


for dtc in range(1):  # for test

	mht_d   = 0   # = 0 better
	mht_d60 = 1   # = 1 better
	sin_cos = 1   # = 1 better
	box     = 1   # = 1 better

	longitude_min   = 0
	longitude_range = 0  
	latitude_min    = 0
	latitude_range  = 0

	df_train        = pd.DataFrame()
	df_test         = pd.DataFrame()

	training_data_reading(300000)         # 3000000         : read 3million rows
									       # 0               : read the whole dataset
	test_data_reading()
#	data_visualization()
	data_cleansing(1, 1, 0, 1, 1, 1)       # parameter1      : longitude_latitude_set (1 - large box; 2 - small box)
										   # parameter2      : flag_clean_fare_less_than_two_point_five
										   # parameter3      : flag_abs_and_clean_fare_less_than_two_point_five
										   # parameter4      : flag_clean_fare_count_less_equal
										   # parameter5      : flag_clean_itude_out_of_range
										   # parameter6      : flag_clean_point_in_water
	feature_manipulation_trainingset()
	feature_manipulation_testset()
	feature_manipulation_for_both_sets()
#	modelling()

	# Keep Relevant Variables..
	y          = df_train.fare_amount.copy()
	#df_test.drop("pickup_datetime", axis = 1, inplace=True)
	df_train   = df_train[df_test.columns]
	trainshape = df_train.shape
	testshape  = df_test.shape

	print("Does Train feature equal test feature?: ", all(df_train.columns == df_test.columns))
	print("df_train Shape: ", df_train.shape)
	print("df_test  Shape: ", df_test.shape)

	print(df_train.head())
	print(df_test.head())

	train_data = lgb.Dataset(df_train, label = y, free_raw_data = False)


	print("----------")
	print("Light Gradient Boosting Regressor: ")
	lgbm_params =  { 'task':          'train',
	                 'boosting_type': 'gbdt',
	                 'objective':     'regression',
	                 'metric':        'rmse' }
	folds       = KFold(n_splits = 5, shuffle = True, random_state = 1)   # shuffle
	test_split_predict   = np.zeros(trainshape[0])   
	test_set_predict     = np.zeros(testshape[0])   

	train_data.construct()
	for training_index, validation_index in folds.split(df_train):      # for 5 times, each of the two _index is a list
	    LightGBM_model = lgb.train(
	        params                = lgbm_params,
	        train_set             = train_data.subset(training_index),
	        valid_sets            = train_data.subset(validation_index),
	        num_boost_round       = 3500,                       # boosting looping times, default = 100
	        early_stopping_rounds = 125,                        # model will train till validation score no longer improve, have to improve in 125 round, otherwise stop
	        verbose_eval          = 500                         # print out eval metric for test split every 500 rounds "valid_0's rmse: 17.7844"
	    )
	    test_split_predict[validation_index] = LightGBM_model.predict(train_data.data.iloc[validation_index])
	    test_set_predict += LightGBM_model.predict(df_test) / folds.n_splits
	    print("RMSE: ", mean_squared_error(y.iloc[validation_index], test_split_predict[validation_index]) ** .5)



	# 14
	test_index = df_test.index
	submission_file = pd.DataFrame(test_set_predict, columns = ["fare_amount"], index = test_index)
	submission_file.to_csv("LightGBM_" + str(strftime('%Y-%m-%d_%H-%M-%S', localtime())) + 
				 "_countclean_" + str(dtc+2) + 
				 ".csv", index = True, header = True)










'''
                X = df_train[features].values
                y = df_train['fare_amount'].values



                #61 generate train and test split
            #    print("-----61 -----")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
                # 训练15000行, 测试5000行




#                model_knn = Pipeline((
#                        ('scale', MinMaxScaler()),
#                        ('normalize', Normalizer()),
#                        ("standard_scaler", StandardScaler()),
#                        ("knn", KNeighborsRegressor(n_neighbors = nbs, n_jobs = -1, weights = "distance")),
#                    ))



                #model_knn = KNeighborsRegressor(p=1)
                model_knn = KNeighborsRegressor(n_neighbors = nbs, n_jobs = -1, weights="distance")




                # modelling
                model_knn.fit(X_train, y_train)

                # predit
                y_train_pred = model_knn.predict(X_train)
                y_test_pred = model_knn.predict(X_test)
                #print(y_train_pred.shape, y_test_pred.shape)

                rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            #    evs = explained_variance_score(y_train, y_train_pred)
            #    avg = np.mean(y_train - y_train_pred)
            #    std = np.std(y_train - y_train_pred)
                test_avg = np.mean(y_test - y_test_pred)
                test_std = np.std(y_test - y_test_pred)
                print("rmse: ", rmse)
            #    print("evs : ", evs)
                print("avg : ", test_avg)
                print("std : ", test_std)


                #df_test_home =  pd.read_csv('test_home.csv', nrows = 6000, parse_dates=["pickup_datetime"], dtype=traintypes)
                #print(df_test_home.info())



                #######################################
                # Generate Kaggle submission
                # The code below can be used to generate a Kaggle submission file.
                #######################################

                # Read Testset
                df_test = pd.read_csv('test.csv', parse_dates=["pickup_datetime"])

                #57
                #print("-----57-----")
                # add new column to dataframe
                #df_test['distance_miles'] = distance(df_test.pickup_latitude, df_test.pickup_longitude, \
                #                                     df_test.dropoff_latitude, df_test.dropoff_longitude)
                df_test['year']              = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
                df_test['month']             = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).month)
                df_test['day']               = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).day)
                df_test['weekday']           = df_test.pickup_datetime.apply(lambda t: t.weekday())
                df_test['hour']              = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
                df_test['period_of_day']     = df_test.pickup_datetime.apply(lambda t: period_of_day(t.hour))

                df_test['pickup_long_idx']   = df_test.apply(lambda row: longitude_to_box(row['pickup_longitude'], resolution, resolution), axis = 1)  
                df_test['pickup_lat_idx']    = df_test.apply(lambda row: latitude_to_box(row['pickup_latitude'], resolution, resolution), axis = 1)
                df_test['dropoff_long_idx']  = df_test.apply(lambda row: longitude_to_box(row['dropoff_longitude'], resolution, resolution), axis = 1)  
                df_test['dropoff_lat_idx']   = df_test.apply(lambda row: latitude_to_box(row['dropoff_latitude'], resolution, resolution), axis = 1)

                df_test['idx_box_distance']  = df_test.apply(lambda row: (math.sqrt(
                    math.pow((latitude_to_box(row['dropoff_latitude'],  resolution, resolution) - latitude_to_box(row['pickup_latitude'],  resolution, resolution)), 2) +
                    math.pow((latitude_to_box(row['dropoff_longitude'], resolution, resolution) - latitude_to_box(row['pickup_longitude'], resolution, resolution)), 2))), axis = 1)
                df_test['geo_distance']      = df_test.apply(lambda row: (math.sqrt(
                    math.pow((row['dropoff_latitude']  - row['pickup_latitude']),  2) +
                    math.pow((row['dropoff_longitude'] - row['pickup_longitude']), 2))), axis = 1)
                df_test['mht_distance']      = df_test.apply(lambda row: (np.abs(row['dropoff_latitude'] - row['pickup_latitude']) +
                    np.abs(row['dropoff_longitude'] - row['pickup_longitude'])), axis = 1)            
                df_test['real_mht_distance'] = df_test.apply(lambda row: real_mht_distance(row['pickup_longitude'], row['pickup_latitude'], 
                    row['dropoff_longitude'], row['dropoff_latitude'], kkk)  , axis = 1)
                
                print(df_test.head())

                #65
                # define dataset
                XTEST = df_test[features].values


                #66
                y_pred_final = model_knn.predict(XTEST)
                submission = pd.DataFrame(
                    {'key': df_test.key, 'fare_amount': y_pred_final},
                    columns = ['key', 'fare_amount'])
                temp_file_name = "submission_model_@" + str(strftime('%Y-%m-%d_%H-%M-%S', localtime())) + "resolution" + str(resolution) + "read_line" + str(read_line) + "nbs" + str(nbs) + "dropcount" + str(1) + "kkk" + str(kkk) + ".csv"
                submission.to_csv(temp_file_name, index = False)
                submission
                print("End: ", strftime('%Y-%m-%d %H:%M:%S', localtime()))

'''