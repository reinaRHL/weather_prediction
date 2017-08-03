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



def main():
    path ='/yvr-weather' # use your path
    allFiles = glob.glob(abspath(getcwd())+ path + "/*.csv")
    data = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, header=14)
        list_.append(df)
    data = pd.concat(list_, ignore_index=True)
    print (data)
      



if __name__=='__main__':
    main()