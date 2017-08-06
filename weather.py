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

    # reading multiple files in the same folder. 
    #Code is from https://stackoverflow.com/questions/39568925/python-read-files-from-directory-and-concatenate-that
    path ='/yvr-weather' # use your path
    allFiles = glob.glob(abspath(getcwd())+ path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, header=14)
        list_.append(df)
    frame = pd.concat(list_, ignore_index=True)
    
    frame = frame[frame['Weather'].notnull()]
    frame['simple cat'] = frame['Weather'].apply(simplify_cat)  
    print (frame.groupby(['simple cat'])['Temp (°C)'].mean())


if __name__=='__main__':
    main()