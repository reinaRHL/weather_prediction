####################################################
# Author: Hwayoung(reina) Lee
# CMPT 318 Final Project
####################################################

import pandas as pd
import numpy as np
import sys
from timeit import default_timer as timer
from scipy import stats, misc
import glob
import os
from os import getcwd
from os.path import join, abspath
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier




# Extract date from the filepath
def getDate(filepath):
    date = os.path.basename(filepath)
    date = os.path.splitext(date)[0]
    date_without_pre = date.lstrip("katkam-")
    date_without_zeros = date_without_pre[:-4]
    date_final = date_without_zeros[0:4] + '-' + date_without_zeros[4:6] + '-' + date_without_zeros[6:8]
    return date_final

def getWeatherDate(date):
    return date[0:10]

def isRained(data):
    flag = 'Rain' in data or 'Snow' in data or 'Drizzle' in data
    if flag is True:
        return 1
    else:
        return 0


def main():

    ########################################################################################
    ############## READ FILE and Clean Data - tide data and image data #####################
    ########################################################################################
    
    # Reading multiple files in the same folder. 
    # Code is from https://stackoverflow.com/questions/39568925/python-read-files-from-directory-and-concatenate-that
    path = sys.argv[1]
    allFiles = glob.glob(abspath(getcwd())+ '/' + path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, header=14)
        list_.append(df)
    frame = pd.concat(list_, ignore_index=True)

    # clean the data
    frame = frame[frame['Weather'].notnull()]
    
    columns = ['Date/Time', 'Year', 'Month', 'Day', 'Weather']
    weather_df = pd.DataFrame(frame, columns=columns)
    
    weather_df['Date/Time'] = weather_df['Date/Time'].apply(getWeatherDate)
    
    # group by a certain col and concat string in another col
    # from https://stackoverflow.com/questions/27298178/concatenate-strings-from-several-rows-using-pandas-groupby
    weather_df = weather_df.groupby(['Date/Time'])['Weather'].apply(lambda x: ','.join(x)).reset_index()
    
    weather_df['Today'] = weather_df['Weather'].apply(isRained)
    weather_df['Tmr'] = weather_df['Today'].shift(-1)
    weather_df = weather_df[weather_df['Tmr'].notnull()]

    columns = ['Date/Time', 'Tmr']
    weather_df = pd.DataFrame(weather_df, columns=columns)

    # read image file from the folder
    # idea from https://stackoverflow.com/questions/39165992/converting-appended-images-list-in-pandas-dataframe
    path = sys.argv[2]
    allFiles = glob.glob(abspath(getcwd())+ '/' + path + "/*.jpg")
    list1_ = []
    list2_ = []
    image_date = pd.DataFrame()
    print ("Reading image file...")
    for file_ in allFiles:
        list1_.append(file_)
        list2_.append(misc.imread(file_).reshape(-1))
    image_date= pd.DataFrame(list1_, columns=['Date/Time'])
    image_date['Date/Time'] = image_date['Date/Time'].apply(getDate)
    image_date.set_index('Date/Time')
    
    image_df = pd.DataFrame.from_records(list2_)

    # Combine Date/time columns and image data
    # and set index as 'Date/Time' so that we can merge this df with weather data on this feature. 
    image_df = pd.concat([image_date, image_df], axis=1)
    image_df.set_index('Date/Time')
    
    # Merge image dataframe and weather data frame 
    # It will keep rows that has matching data from both dataFrame
    weather_df = weather_df.merge(image_df, on='Date/Time')

    # ################################################################
    # ############## Training and Testing Data   #####################
    # ################################################################

    #Build models for tide prediction
    # gaussian yielded most accurate result
    gsModel = GaussianNB()

    # input data
    X = weather_df[weather_df.columns[2:147459]]

    # target data, tide height is either 0 (false) 1 (true)
    y = weather_df['Tmr'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    gsModel.fit(X_train, y_train)

    print ("\nAccuracy score for tide height prediction:")
    print (gsModel.score(X_test, y_test))

    # # ####################################################
    # # ############## Sample input  #######################
    # # ####################################################

    # Predict tide and time using some test input
    path = sys.argv[3]
    allFiles = glob.glob(abspath(getcwd())+  "/" + path + "/*.jpg")
    list3_ = []
    for file_ in allFiles:
        list3_.append(misc.imread(file_).reshape(-1))

    test_df = pd.DataFrame.from_records(list3_)

    X_pre = test_df[test_df.columns[0:147456]]

    print ("\nWeather Forecast prediction from the sample input:")
    predictions = gsModel.predict(X_pre)
    print (predictions)



if __name__=='__main__':
    main()