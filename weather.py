####################################################
# Author: Hwayoung(reina) Lee
# CMPT 318 Final Project
####################################################

import pandas as pd
import numpy as np
import sys
import gzip
import math as Math
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy import stats
import glob
from os import getcwd
from os.path import join, abspath

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

def main():

    ####################################################
    ############## READ INPUT FILE #####################
    ####################################################
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
    
    ####################################################
    ############## Cleaning Data   #####################
    ####################################################
    # Extract meaningful columns
    columns = ['Date/Time', 'Year', 'Month', 'Day', 'Temp (°C)', 'Weather']
    weather_df = pd.DataFrame(frame, columns=columns)
    
    # Remove row whose weather column is 'nan'
    weather_df = weather_df[weather_df['Weather'].notnull()]

    # Simplify the category. For example, 'moderate rain' to 'rain'
    weather_df['simple cat'] = weather_df['Weather'].apply(simplify_cat)
    
    # Again, keep only meaningful columns
    columns = ['Date/Time', 'Year', 'Month', 'Day', 'Temp (°C)', 'simple cat']
    weather_df = pd.DataFrame(weather_df, columns=columns)  
    

    print (weather_df.groupby(['simple cat'])['Temp (°C)'].mean())
    print (weather_df)




if __name__=='__main__':
    main()