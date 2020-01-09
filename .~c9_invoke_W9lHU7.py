import numpy as np
import pandas as pd

import sys, pprint

from explainme import XAI

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris, load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical

from utilities import join_x_with_y_split

def instance_to_df(instance, features):
    instance = instance.reshape(1,-1)
    instance = pd.DataFrame(instance, columns=features)
    return instance
    
def load_bikes():
    df = pd.read_csv("day.csv")
    return df

raw = load_bikes()

remove = ["cnt","registered", "casual", "dteday", "season"]

target = np.array(raw["cnt"])
####
#TEST WITH CERTAIN VARIABLES AND WITHOUT (SEASON, YEAR, MONTH, HOLIDAY, WEEKDAY, WORKINGDAY)

data = np.array(raw.drop(remove, axis=1))
for i, inst in enumerate(data):
    if inst[4] == 0:
        data[i,4] = 7
        
#TEST WITH CERTAIN VARIABLES AND WITHOUT
####

feature_names = list(raw.columns)
for i in remove:
    feature_names.remove(i)

idx = list(feature_names).index("weathersit")
ohe = OneHotEncoder(categorical_features = [idx])
data = ohe.fit_transform(data).toarray()

feature_names.remove("weathersit")
feature_names.insert(0,"weather_wet")
feature_names.insert(0,"weather_misty")
feature_names.insert(0,"weather_clear")

np.random.seed(42)

target = np.array(target)

train_x, test_x, train_y, test_y = train_test_split(data, target)


#clf = SVR()
#clf = RandomForestClassifier(n_estimators=200)
clf = RandomForestRegressor(n_estimators=400)

clf.fit(train_x, train_y)

score = clf.score(test_x, test_y)
print("The r2 score is {}\n".format(score))

ohe_categories = {"weather": ["weather_wet","weather_misty","weather_clear"]}

#xai = XAI(model=clf, train_x=train_x, train_y=train_y, mode="classification", features=feature_names, n_classes=3)
xai = XAI(model=clf, train_x=train_x, train_y=train_y, mode="regression", features=feature_names, n_classes=1, ohe_categories=ohe_categories)

## Running explainers
print("\nTree surrogate:")
xai.surrogate.global_surrogate(filename="bikes-tree-surrogate")
sys.exit()
print("\nShapley Values:")
shap, expect = xai.shap.shapley_tree(test_x)
xai.shap.plot_shapley(shap[8], expect, filename="bikes-one-shap")
print("Expected Value = {}".format(expect))
print("\nLIME explanation:")
exp = xai.lime.tabular_explanation(test_x[0], filename="bikes-zero-lime")
print("\nALE plot:")
xai.ale.plot_single("weekday", filename="bikes-instance-ale")
print("Single PDP:")
xai.pdp.calculate_single_pdp(feature_name="weekday", filename="bikes-holiday-pdp")
print("\nDouble PDP:")
xai.pdp.calculate_double_pdp(feature_names=["hum", "temp"], filename="bikes-temp-weather-pdp")
print("\nShapley Values:")
shap, expect = xai.shap.shapley_tree(test_x[:20])
xai.shap.standard_plot(shap[2], expect, filename="bikes-shap")
print("\nDouble PDP:")
xai.pdp.calculate_double_pdp(feature_names=["mnth", "temp"], filename="bikes-instance-weather-pdp")
print("\nCorrelation Matrix:")
corr = xai.correlation("correlation_matrix")
print("\nTree surrogate:")
xai.surrogate.global_surrogate(filename="bikes-tree-surrogate")
print("\nICE plots:")
xai.ceteris.plot_instances(test_x[:20], test_y[:20], selected_variables=("temp",))
print("\nFeature Importance:")
imp = xai.feature_importance(method="all")
print("\nAnchor Explanation:")
xai.anchor.anchor_explanation(test_x[0])
