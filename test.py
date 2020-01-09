import numpy as np
import pandas as pd

import sys, pprint, time, os

from explainme import XAI, BohemianWrapsody

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris, load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb

from utilities import join_x_with_y_split

def instance_to_df(instance, features):
    return pd.DataFrame(instance.reshape(1,-1), columns=features)
    
def load_bikes():
    return pd.read_csv("day.csv")

raw = load_bikes()


#FEATURES TO REMOVE
remove = ["cnt","registered", "casual", "dteday", "season"]

#EXTRACT TARGET VARIABLE AND CONVERT TO ARRAY
target = np.array(raw["cnt"])
target = np.array(target)

#REMOVE REMOVABLE FEATURES AND CONVERT TO ARRAY
data = np.array(raw.drop(remove, axis=1))

#MAKE 1=MONDAY IN DATA RATHER THAN 1=SUNDAY BECAUSE REASONS
for i, inst in enumerate(data):
    if inst[4] == 0:
        data[i,4] = 7
        
#REMOVE REMOVABLE FEATURES FROM THE FEATURE LIST
feature_names = list(raw.columns)
for i in remove:
    feature_names.remove(i)

#CONVERT THE WEATHER FEATURE TO A CATEGORICAL VARIABLE THROUGH ONEHOTENCODING
idx = list(feature_names).index("weathersit")
ohe = OneHotEncoder(categorical_features = [idx])
data = ohe.fit_transform(data).toarray()
feature_names.remove("weathersit")
feature_names.insert(0,"weather_wet")
feature_names.insert(0,"weather_misty")
feature_names.insert(0,"weather_clear")

#ESTABLISH IDENTIFYING NAME FOR THE THREE NEW CATEGORIES
ohe_categories = {"weather": ["weather_wet","weather_misty","weather_clear"]}

np.random.seed(42)

#SPLIT DATA
train_x, test_x, train_y, test_y = train_test_split(data, target)

#TRAIN AND SCORE THE MODEL (sklearn)
clf = RandomForestRegressor(n_estimators=400)
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
res = mean_squared_error(test_y, y_pred)
score = clf.score(test_x, test_y)

#TRAIN AND SCORE THE MODEL (keras)
if "best_model.h5" in os.listdir():
    model = load_model("best_model.h5")
else:
    model = Sequential()
    model.add(Dense(128, input_dim=len(feature_names), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])
    es = EarlyStopping(monitor='val_loss', patience=75, verbose=1)
    model.fit(train_x, train_y, epochs=4000, validation_split=0.1, callbacks=[es])
    model.save("best_model.h5")
score = model.evaluate(test_x, test_y)

#TRAIN AND SCORE THE MODEL (xgboost)
predictor = xgb.XGBRegressor(max_depth=4)
predictor.fit(train_x, train_y)

#TRAIN AND SCORE THE MODEL (catboost)
if "best_cat_model" in os.listdir():
    guessbox = CatBoostRegressor()
    guessbox.load_model("best_cat_model")
else:
    cat_train = Pool(train_x, label=train_y)
    
    
    guessbox = CatBoostRegressor(iterations=100, 
                              depth=4, 
                              learning_rate=1, 
                              loss_function='RMSE')
                              
    guessbox.fit(cat_train)
    guessbox.save_model("best_cat_model")
    
#TRAIN AND SCORE THE MODEL (LightGBM)
if "best_lgm_model" in os.listdir():
    sasquatch = lgb.Booster(model_file='best_lgm_model')
else:
    lgb_train = lgb.Dataset(train_x, label=train_y, feature_name=feature_names)
    params = {'num_leaves': 30}
    sasquatch = lgb.train(params, lgb_train, 100)
    sasquatch.save_model("best_lgm_model")

model = BohemianWrapsody(clf, feature_names, "regression")

xai = XAI(model=model, train_x=train_x, train_y=train_y, mode="regression", features=feature_names, n_classes=1, ohe_categories=ohe_categories)

#RUNNING EXPLAINERS
print("Shapley Values:")
shap, expect = xai.shap.shapley_tree(test_x)
xai.shap.summary_plot(shap, test_x, "shapsummary-pdp")
xai.shap.plot_shapley(shap[2], expect, "shap_thingy")
print(expect)

print("Single PDP:")
xai.pdp.single(feature_name="temp", filename="bikes-temp-pdp")

print("\nDouble PDP:")
xai.pdp.double(feature_names=["hum", "temp"], filename="bikes-temp-weather-pdp")

print("ALE plot:")
xai.ale.plot_single("weekday", filename="bikes-weekday-ale")

print("\nLIME explanation:")


exp = xai.lime.tabular_explanation(test_x[0], filename="bikes-zero-lime")

print("\nTree surrogate:")
xai.surrogate.tree_surrogate("bikes-tree-surrogate", max_leaves=15)

print("\nFeature Importance:")
imp = xai.feature_importance(method="model_scoring", plot=True, filename="testtubebabies.png")

print("\nCorrelation Matrix")
hi = xai.correlation(filename="poopoopee.png", plot=True)

print("\nRegression Performance")
xai.reg_performance(train_x, train_y, "reggyMcgee")

print("\nFeature Distribution")
xai.feature_distribution("feature-dist")

print("\nGAM Surrogate")
xai.gam_surrogate()

print("\nMARS Surrogate")
xai.mars_surrogate("bikes")

print("\nSummarise feature")
xai.summarise_feature("temp")