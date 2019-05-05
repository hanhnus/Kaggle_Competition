ZonesDevelopment.py
is only for functions development, especially for the NYC zoning function.

OriginalKNN.py
has all the data reading, data cleaning, feature engineering part of the program. The original KNN method was commented out at the end of the file, because after switching to LightGBM, the previous parts of data process were also updated accordingly. The inferface was not maintained after changing to new algorithm.

FinalLightGBM.py
This program generated the .csv file got 2.99 Kaggle score.

LightGBM_5400000rows.csv
This file is the one submitted to Kaggle.

the 3 .png files
needed for all .py to print out points on maps

train.csv & test.csv
removed. too big to upload.
