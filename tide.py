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



def classify_tide(data, avg, std):
    if data < avg - std:
        return 'low'
    elif data >= avg-std and data < avg+std:
        return 'medium'
    else:
        return 'high'

def classify_time(time):
    morning = [4,5,6,7,8,9,10,11]
    afternoon = [12,13,14,15,16,17]
    evening = [18,19,20,21]
    night = [22,23,0, 1,2,3]

    if time in morning:
        return 'morning'
    elif time in afternoon:
        return 'afternoon'
    elif time in evening:
        return 'evening'
    else:
        return 'night'


# Extract date from the filepath
def getDate(filepath):
    date = os.path.basename(filepath)
    date = os.path.splitext(date)[0]
    date_without_pre = date.lstrip("katkam-")
    date_without_zeros = date_without_pre[:-4]
    date_final = date_without_zeros[0:4] + '-' + date_without_zeros[4:6] + '-' + date_without_zeros[6:8] + ' '+ date_without_zeros[8:10]+ ":00"
    return date_final

def year(t):
    year = t[0:4]
    return year
def month(t):
    month = t[5:7]
    return month
def date(t):
    date = t[8:10]
    return date

def time(t):
    time = t[11:13]
    return time

def datetime(t):
    date_final = t[0:13] + ":00"
    return date_final


def main():

    ########################################################################################
    ############## READ FILE and Clean Data - tide data and image data #####################
    ########################################################################################
    
    tide_df = pd.read_csv('tideData.csv')

    # Clean the tide height data
    tide_df = pd.DataFrame(tide_df, columns=['TIME_TAG PST (Z+8)', 'ENCODER1'])
    tide_df.columns = ['Date/Time', 'Tide']
    tide_df['Date/Time'] = tide_df['Date/Time'].apply(datetime)
    avg_tide = tide_df['Tide'].mean()
    std_tide = tide_df['Tide'].std()
    tide_df_new = tide_df.groupby(tide_df['Date/Time']).mean()
    tide_df_new['Tide'] = tide_df_new['Tide'].astype(float).apply(classify_tide,  args = (avg_tide, std_tide,))
    tide_df_new = tide_df_new.reset_index()
    tide_df_new["year"]= tide_df_new['Date/Time'].apply(year)
    tide_df_new["month"]= tide_df_new['Date/Time'].apply(month)
    tide_df_new["date"]= tide_df_new['Date/Time'].apply(date)
    tide_df_new["time"]= tide_df_new['Date/Time'].apply(time)
    tide_df_new['time'] = tide_df_new['time'].astype(int).apply(classify_time)

    # read image file from the folder
    # idea from https://stackoverflow.com/questions/39165992/converting-appended-images-list-in-pandas-dataframe
    path ='/katkam-scaled_700'
    allFiles = glob.glob(abspath(getcwd())+ path + "/*.jpg")
    list1_ = []
    list2_ = []
    image_date = pd.DataFrame()
    for file_ in allFiles:
        list1_.append(os.path.basename(file_))
        list2_.append(misc.imread(file_).reshape(-1))

    # image_date contains 'Date/Time' of the image
    image_date= pd.DataFrame(list1_, columns=['Date/Time'])
    image_date['Date/Time'] = image_date['Date/Time'].apply(getDate)

    # image_df contains image information read from input
    image_df = pd.DataFrame.from_records(list2_)

    # Combine Date/time columns and image data
    # and set index as 'Date/Time' so that we can merge this df with tide data on this feature. 
    image_df = pd.concat([image_date, image_df], axis=1)
    image_df.set_index('Date/Time')

    # Merge image dataframe and tide data frame 
    # It will keep rows that has matching date/time from both dataFrame
    tide_df_new = tide_df_new.merge(image_df, on='Date/Time')


    # ################################################################
    # ############## Training and Testing Data   #####################
    # ################################################################

    #Build models for tide prediction

    #gsModel_tide = make_pipeline(StandardScaler(), GaussianNB())

    #knn_model_tide = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11))
    
    # svc model is most accurate for this data
    svc_model_tide = make_pipeline(StandardScaler(), SVC())

    # input data
    X = tide_df_new[tide_df_new.columns[6:147463]]

    # target data, tide height is one of 'low', 'medium', 'high'
    y = tide_df_new['Tide'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #gsModel_tide.fit(X_train, y_train)
    #knn_model_tide.fit(X_train, y_train)
    svc_model_tide.fit(X_train, y_train)

    print ("Accuracy score for tide height prediction:")
    # print ("Gaussian-")
    # print (gsModel_tide.score(X_test, y_test))
    # print ("KNN-")
    # print (knn_model_tide.score(X_test, y_test))
    print (svc_model_tide.score(X_test, y_test))


    # Build a model for time prediction
    
    #gsModel_time = GaussianNB()

    # This time, knn was the most accurate model
    knn_model_time = KNeighborsClassifier(n_neighbors=11)
    
    #svc_model_time = SVC()

    # input
    X = tide_df_new[tide_df_new.columns[6:147463]]
    
    # target, time is one of 'morning', 'afternoon', 'evening', 'night'
    y = tide_df_new['time'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #gsModel_time.fit(X_train, y_train)
    knn_model_time.fit(X_train, y_train)
    #svc_model_time.fit(X_train, y_train)

    print ("Accuracy score for time prediction:")
    # print ("Gaussian-")
    # print (gsModel_time.score(X_test, y_test))
    print (knn_model_time.score(X_test, y_test))
    # print ("SVC-")
    # print (svc_model_time.score(X_test, y_test))


    # ####################################################
    # ############## Verify Data   #######################
    # ####################################################

    # # Predict weather using some test input
    # path ='/testing_img'
    # allFiles = glob.glob(abspath(getcwd())+ path + "/*.jpg")
    # list3_ = []
    # for file_ in allFiles:
    #     list3_.append(misc.imread(file_).reshape(-1))

    # test_df = pd.DataFrame.from_records(list3_)

    # X_pre = test_df[test_df.columns[0:147456]]


    # predictions = gsModel.predict(X_pre)
    # print ("gspredict")
    # print (predictions)
    # predictions = knn_model.predict(X_pre)
    # print ("knn")
    # print (predictions)
    # predictions = svc_model.predict(X_pre)
    # print ("svc")
    # print (predictions)



if __name__=='__main__':
    main()