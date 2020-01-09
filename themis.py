import pandas as pd
import numpy as np

from themis_ml.metrics import mean_difference
from utilities import join_x_with_y

class Themis():
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
        
    def mean_diff(self, feature): 
        """
        ONLY BINARY TARGET VARIABLES FFS
        """
        df = join_x_with_y(self.train_x, self.train_y, self.xai.features)
        
        target = df["Target"]
        
        feature_column = df[feature]
        
        mean, lower_b, upper_b = mean_difference(target, feature_column)
        
        print("\nThe mean difference between target value and protected class {0}:\n".format(feature))
        print("\tMean:\t\t\t{0}".format(mean))
        print("\tLower Bound:\t\t{0}".format(lower_b))
        print("\tUpper Bound:\t\t{0}".format(upper_b))
        
        return [mean, lower_b, upper_b]