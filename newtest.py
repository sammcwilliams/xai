import pandas as pd
import numpy as np

import sys, os

def check_same(df):
    pass

def main():
    one = pd.read_csv("../happiness/2015.csv")
    one["Year"] = 2015
    two = pd.read_csv("../happiness/2016.csv")
    two["Year"] = 2016
    three = pd.read_csv("../happiness/2017.csv")
    three["Year"] = 2017
    print(one.columns)
    print(two.columns)
    print(len(three.columns))
    x = [one,two,three]
    df = pd.concat(x, ignore_index=True)
    
    for i in list(df.columns):
        if df[i].dtype != np.float64:
            print("{}: ".format(i), df[i].value_counts())
        

if __name__ == "__main__":
    main()