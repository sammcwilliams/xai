import numpy as np
import pandas as pd

import pysbrl

class Rules():
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
        
    def generate(self):
        ##What is this data format ??????? https://github.com/myaooo/pysbrl/blob/master/data/ttt_test.out
        rule_ids, outputs, rule_strings = pysbrl.train_sbrl("data/ttt_train.out", "data/ttt_train.label", 20.0, eta=2.0, max_iters=2000, nchain=10, alphas=[1,1])