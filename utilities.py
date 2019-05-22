# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:40:57 2019

@author: smcwilliams1
"""

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, BaggingRegressor, BaggingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import normalize

import xgboost as xgb

from sklearn.utils import shuffle
from sklearn.externals import joblib

from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D

from sklearn.datasets import load_boston, load_iris, load_wine, load_breast_cancer, load_diabetes

import numpy as np
import pandas as pd

from shap.explainers.tree import Tree

def sample_from_large_data(orig_x, orig_y, size=3000):
    max_idx = len(orig_x) -1
    for i in range(size):
        idx = np.random.randint(max_idx)
        
        x = orig_x[idx]
        if i != 0:
            x = x.reshape(1,-1)
            new_x = np.append(new_x, x, axis=0)
        else:
            new_x = x.reshape(1,-1)
        orig_x = np.delete(orig_x, idx, 0)
        
        y = orig_y[idx]
        if i != 0:
            y = y.reshape(1,-1)
            new_y = np.append(new_y, y, axis=0)
        else:
            new_y = y.reshape(1,-1)
        orig_y = np.delete(orig_y, idx, 0)
        
        max_idx -= 1
    return new_x, new_y
    
def train_poker():
    prior = [0.50117739, 0.42256903, 0.4753902, 0.2112845, 0.0392465, 0.019654, 0.0144058, 0.002401, 0.0001385, 0.0000154]
    choice = "mnb"
    if choice == "mlp":
        model = MLPClassifier((200,200), activation="relu", solver ="adam", early_stopping=True, verbose=True)
    elif choice == "rtree":
        model = RandomForestClassifier(n_estimators=900)
    elif choice == "etree":
        model = ExtraTreesClassifier(n_estimators=900)
    elif choice == "gbr":
        model = GradientBoostingClassifier(n_estimators=1500)
    elif choice == "br":
        model = BaggingClassifier(n_estimators=900)
    elif choice == "mnb":
        model = MultinomialNB(fit_prior=True, class_prior=prior)
        
    classes = [0,1,2,3,4,5,6,7,8,9]
    
    
    chunksize = 1000
    x_arr = np.memmap("poker_x.memmap", mode="w+", shape=(100000,	10), dtype="int32")
    y_arr = np.memmap("poker_y.memmap", mode="w+", shape=(100000,	1), dtype="int32")
    for i, chunk in enumerate(pd.read_csv("poker/poker.data", chunksize=chunksize)):
        if i < 25:
            model.partial_fit(np.array(chunk)[:,:-1], np.array(chunk)[:,-1:], classes)
        else:
            try:
                x_arr[i*chunksize:(i+1)*chunksize,:] = np.array(chunk)[:,:-1]
                y_arr[i*chunksize:(i+1)*chunksize,:] = np.array(chunk)[:,-1:]
            except:
                pass
        
    
    predict = model.predict(x_arr)
    error = r2_score(y_arr, predict)
    print("Error of trained model is {0}\n".format(error))
    
    save = int(input("Save the model, true (1) or false (0): "))
    if save == 1:
        joblib.dump(model, "poker_trained_model.pkl")

def train_census():
    chunksize = 1000
    x_arr = np.memmap("cali_x.memmap", mode="w+", shape=(20639,	8), dtype="float64")
    y_arr = np.memmap("cali_y.memmap", mode="w+", shape=(20639,	1), dtype="float64")
    for i, chunk in enumerate(pd.read_csv("CaliforniaHousing/cal_housing.data", chunksize=chunksize)):
        print(np.array(chunk)[0])
        try:
            x_arr[i*chunksize:(i+1)*chunksize,:] = np.array(chunk)[:,:-1]
            y_arr[i*chunksize:(i+1)*chunksize,:] = np.array(chunk)[:,-1:]
        except:
            pass
    choice = "br"
    if choice == "mlp":
        model = MLPRegressor((200,200), activation="relu", solver ="adam", early_stopping=True, verbose=True)
    elif choice == "rtree":
        model = RandomForestRegressor(n_estimators=300)
    elif choice == "etree":
        model = ExtraTreesRegressor(n_estimators=300)
    elif choice == "gbr":
        model = GradientBoostingRegressor(n_estimators=1500)
    elif choice == "br":
        model = BaggingRegressor(n_estimators=900)
        
    
    cutoff = int(0.8*(len(x_arr)))
    print("cutoff is {}".format(cutoff))
    train_x = x_arr[:cutoff]
    train_y = y_arr[:cutoff]
    test_x = x_arr[cutoff:]
    test_y = y_arr[cutoff:]
    
    model.fit(train_x, train_y)
    
    predict = model.predict(test_x)
    error = r2_score(test_y, predict)
    print("Error of trained model is {0}\n".format(error))
    
    save = int(input("Save the model, true (1) or false (0): "))
    if save == 1:
        joblib.dump(model, "cali_trained_model.pkl")
    
        

class DataFormatter():
    def __init__(self, dataset, features, learn_type, data_type, class_names, training_ratio=0.7):
        cutoff = int(training_ratio*len(dataset.data))
        feat = dataset.data
        targ = dataset.target
        feat, targ = shuffle(feat, targ)
        self.X = dataset.data
        self.Y = dataset.target
        self.Ytrain = np.array([x for x in targ[:cutoff]])
        self.Ytest = np.array([x for x in targ[cutoff:]])
        self.Xtrain = np.array([x for x in feat[:cutoff]])
        self.Xtest = np.array([x for x in feat[cutoff:]])
        self.dataset= []
        for i,j in zip(dataset.data,dataset.target):
            self.dataset.append(np.append(i,j))
        self.dataset = np.array(self.dataset)
        self.data_type = data_type
        self.learn_type = learn_type
        self.class_names = class_names
        self.training_ratio = training_ratio
        self.features = features
        #self.dataframe = pd.DataFrame(self.dataset, columns=(self.features.append(class_name)))

def get_model(model_type, learn_type, input_shape=None, output_size=None, data=None):
    if model_type == "svm" and learn_type == "classification":
        model = SVC(probability=True)
    elif model_type == "svm" and learn_type == "regression":
        model = SVR()
    elif model_type == "xgb" and learn_type =="regression":
        assert data is not None, "XGB DMatrix requires the data to be initialised"
        model = xgb.DMatrix(data=data.Xtrain, label=data.Ytrain)
    elif model_type == "nn" and learn_type == "regression":
        assert input_shape is not None and output_size is not None, "Need to know input and output size to build neural net"
        model = Sequential()
        model.add(Dense(128, kernel_initializer="glorot_normal", activation="relu", input_shape=(input_shape[1],)))
        model.add(Dense(256, kernel_initializer="glorot_normal", activation="relu"))
        model.add(Dense(output_size, kernel_initializer="glorot_normal", activation="linear"))
        model.compile(optimizer="adam", loss="mean_squared_error")
    elif model_type == "nn" and learn_type == "classification":
        assert input_shape is not None and output_size is not None, "Need to know input and output size to build neural net"
        model = Sequential()
        model.add(Dense(128, kernel_initializer="glorot_normal", activation="relu", input_shape=(input_shape[1],)))
        model.add(Dense(256, kernel_initializer="glorot_normal", activation="relu"))
        model.add(Dense(256, kernel_initializer="glorot_normal", activation="relu"))
        model.add(Dense(output_size, kernel_initializer="glorot_normal", activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy")
    elif model_type == "tree" and learn_type == "classification":
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "tree" and learn_type == "regression":
        model = RandomForestRegressor(n_estimators=100)
    elif model_type == "linear" or learn_type == "regression":
        model = LinearRegression()
    elif model_type == "logistic" or learn_type == "classification":
        model = LinearRegression()
    else:
        raise ValueError("please give proper model type")
    return model, model_type

def create_permuted(dataset, targets, feature_no, faster = False):
    Xpermuted = []
    Ypermuted = []
    if faster:
        Xfirst = dataset[:int(len(dataset)/2)]
        Xsecond = dataset[int(len(dataset)/2):]
        Yfirst = targets[:int(len(dataset)/2)]
        Ysecond = targets[int(len(dataset)/2):]
        
        if len(Xfirst) > len(Xsecond):
            Xfirst = Xfirst[:-1]
            Yfirst = Yfirst[:-1]
        elif len(Xfirst) < len(Xsecond):
            Xsecond = Xsecond[:-1]
            Ysecond = Ysecond[:-1]

        for i, j in zip(Xfirst,Xsecond):
            temp = list(i)
            temp[feature_no] = j[feature_no]
            Xpermuted.append(temp)
            temp = list(j)
            temp[feature_no] = i[feature_no]
            Xpermuted.append(temp)
        for i, j in zip(Yfirst,Ysecond):
            Ypermuted.append(i)
            Ypermuted.append(j)

    else:
        for num, i in enumerate(dataset):
            for j in range(len(dataset)):
                if num != j:
                    temp = list(i)
                    temp[feature_no] = dataset[j][feature_no]
                    Xpermuted.append(temp)
                    Ypermuted.append(targets[num])

    Xpermuted = np.array(Xpermuted)
    return Xpermuted, Ypermuted

def is_shapley_tree(model):
    is_tree = False
    if type(model) == list and type(model[0]) == Tree:
        is_tree = True
    elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("skopt.learning.forest.RandomForestRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("skopt.learning.forest.ExtraTreesRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
        is_tree = True
    elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
        is_tree = True
    elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesClassifier'>"):
        is_tree = True
    elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>"):
        is_tree = True
    elif str(type(model)).endswith("xgboost.core.Booster'>"):
        is_tree = True
    elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
        is_tree = True
    elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("xgboost.sklearn.XGBRanker'>"):
        is_tree = True
    elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
        is_tree = True
    elif str(type(model)).endswith("lightgbm.sklearn.LGBMRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("lightgbm.sklearn.LGBMRanker'>"):
        is_tree = True
    elif str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
        is_tree = True
    elif str(type(model)).endswith("catboost.core.CatBoostRegressor'>"):
        is_tree = True
    elif str(type(model)).endswith("catboost.core.CatBoostClassifier'>"):
        is_tree = True
    elif str(type(model)).endswith("catboost.core.CatBoost'>"):
        is_tree = True
    elif str(type(model)).endswith("imblearn.ensemble._forest.BalancedRandomForestClassifier'>"):
        is_tree = True
    return is_tree

def get_dataset(name):
    if name == "wine":
        dataset = load_wine()
        classes = dataset.target_names
        learn_type = "classification"
        features = dataset.feature_names
    elif name == "iris":
        dataset = load_iris()
        classes = dataset.target_names
        learn_type = "classification"
        features = dataset.feature_names
    elif name == "boston":
        dataset = load_boston()
        classes = "Value"
        learn_type = "regression"
        features = dataset.feature_names
    elif name == "bc":
        dataset = load_breast_cancer()
        classes = dataset.target_names
        learn_type = "classification"
        features = dataset.feature_names
    elif name == "diabetes":
        dataset = load_diabetes()
        classes = "Value"
        learn_type = "regression"
        features = dataset.feature_names
    else:
        raise ValueError("not a dataset yet matey")
    
    return dataset, features, classes, learn_type
