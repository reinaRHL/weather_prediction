####################################################
# Author: Hwayoung(reina) Lee
# CMPT 318 Final Project
####################################################

import pandas as pd
import numpy as np
import sys
import gzip
from timeit import default_timer as timer
from scipy import stats, misc
import glob
import os
from os import getcwd
from os.path import join, abspath
import csv
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC

def simplify_cat(weather):
    #simplify (clean) category
    weather = weather.replace('Moderate Snow','Snow')
    weather = weather.replace('Freezing Fog','Fog')
    weather = weather.replace('Freezing Rain','Rain')
    weather = weather.replace('Moderate Rain Showers','Rain Showers')
    weather = weather.replace('Moderate Rain','Rain')
    weather = weather.replace('Mainly Clear','Clear')
    weather = weather.replace('Moderate Snow','Snow')
    weather = weather.replace('Mostly Cloudy','Cloudy')
    weather = weather.replace('Snow Pellets','Ice Pellets')
    return weather

# Extract date (YYYYmmDD-HH) from the filepath
def getDate(filepath):
    date = os.path.basename(filepath)
    date = os.path.splitext(date)[0]
    date_without_pre = date.lstrip("katkam-")
    date_without_zeros = date_without_pre[:-4]
    date_final = date_without_zeros[0:4] + '-' + date_without_zeros[4:6] + '-' + date_without_zeros[6:8] + ' '+ date_without_zeros[8:10]+ ":00"
    return date_final


def main():

    ##################################################################################
    ############## READ INPUT FILE - weather data and image data #####################
    ##################################################################################

    # Reading multiple files in the same folder. 
    # Code is from https://stackoverflow.com/questions/39568925/python-read-files-from-directory-and-concatenate-that
    path = sys.argv[1]
    allFiles = glob.glob(abspath(getcwd())+ "/" + path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, header=14)
        list_.append(df)
    frame = pd.concat(list_, ignore_index=True)
    
    # idea from https://stackoverflow.com/questions/39165992/converting-appended-images-list-in-pandas-dataframe
    path =sys.argv[2]
    allFiles = glob.glob(abspath(getcwd())+  "/" + path + "/*.jpg")
    list1_ = []
    list2_ = []
    image_date = pd.DataFrame()
    for file_ in allFiles:
        list1_.append(os.path.basename(file_))
        list2_.append(misc.imread(file_).reshape(-1))

    # image_date contains 'Date/Time' info
    image_date= pd.DataFrame(list1_, columns=['Date/Time'])
    image_date['Date/Time'] = image_date['Date/Time'].apply(getDate)

    # image_df contains image information read from input
    image_df = pd.DataFrame.from_records(list2_)

    # Combine Date/time columns and image data
    # and set index as 'Date/Time' so that we can merge this df with weather data on this feature. 
    image_df = pd.concat([image_date, image_df], axis=1)
    image_df.set_index('Date/Time')


    ####################################################
    ############## Cleaning Data   #####################
    ####################################################

    # The input weather data is quite big. 
    # So, extract meaningful columns
    columns = ['Date/Time', 'Year', 'Month', 'Day', 'Temp (°C)', 'Weather']
    weather_df = pd.DataFrame(frame, columns=columns)
    
    # Remove rows whose 'weather' description is 'nan'
    weather_df = weather_df[weather_df['Weather'].notnull()]

    # Simplify the categories. For example, 'moderate rain' to 'rain'
    weather_df['simple cat'] = weather_df['Weather'].apply(simplify_cat)
    
    # Again, keep only meaningful columns
    columns = ['Date/Time', 'Year', 'Month', 'Day', 'Temp (°C)', 'simple cat']
    weather_df = pd.DataFrame(weather_df, columns=columns)  
    
    # Merge image dataframe and weather data frame 
    # It will keep rows that has matching data from both dataFrame    
    weather_df = weather_df.merge(image_df, on='Date/Time')
    weather_df['simple cat'] = weather_df['simple cat'].str.split(',')


    ################################################################
    ############## Training and Testing Data   #####################
    ################################################################

    # Build a model
    # Multi label classification: 
    # Part of code is from https://stackoverflow.com/questions/10526579/use-scikit-learn-to-classify-into-multiple-categories
    X = weather_df[weather_df.columns[6:147462]]
    y = weather_df['simple cat'].values
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    model = OneVsRestClassifier(LinearSVC(random_state=0))

    model.fit(X_train, y_train)
    print ("\nAccuracy score for weather label prediction:")
    print (model.score(X_test, y_test))


    ####################################################
    ############## Predict Sample input ################
    ####################################################

    # Predict weather using some test input
    path = sys.argv[3]
    allFiles = glob.glob(abspath(getcwd())+  "/" + path + "/*.jpg")
    list3_ = []
    for file_ in allFiles:
        list3_.append(misc.imread(file_).reshape(-1))

    test_df = pd.DataFrame.from_records(list3_)

    X_pre = test_df[test_df.columns[0:147456]]

    # Part of code is from https://stackoverflow.com/questions/10526579/use-scikit-learn-to-classify-into-multiple-categories
    predictions = model.predict(X_pre)
    origin_label = mlb.inverse_transform(predictions)
    print ("\nWeather Label Prediction from the sample input:")
    print (origin_label)
    pd.Series(origin_label).to_csv("labels.csv", index=False)




if __name__=='__main__':
    main()