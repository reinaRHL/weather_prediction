####################################################
# Author: Hwayoung(reina) Lee
# CMPT 318 Final Project
####################################################

from timeit import default_timer as timer
import pandas as pd
import numpy as np
import sys
import gzip
import math as Math
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy import stats, misc
import glob
import os
from os import getcwd
from os.path import join, abspath
import csv

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
    start = timer()


    ##################################################################################
    ############## READ INPUT FILE - weather data and image data #####################
    ##################################################################################

    # Reading multiple files in the same folder. 
    # Code is from https://stackoverflow.com/questions/39568925/python-read-files-from-directory-and-concatenate-that
    path ='/yvr-weather'
    allFiles = glob.glob(abspath(getcwd())+ path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, header=14)
        list_.append(df)
    frame = pd.concat(list_, ignore_index=True)
    
    path ='/katkam-scaled'
    allFiles = glob.glob(abspath(getcwd())+ path + "/*.jpg")
    list1_ = []
    list2_ = []
    image_date = pd.DataFrame()
    print ("Reading image file...")
    for file_ in allFiles:
        list1_.append(os.path.basename(file_))
        list2_.append(misc.imread(file_).reshape(-1))
    print ("read image file...")

    #image_date contains 'Date/Time' info
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
    print (weather_df)

    end = timer()
    print (end-start)



if __name__=='__main__':
    main()