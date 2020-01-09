import numpy as np
import pandas as pd

import sys, pprint, os

from explainme import XAI, BohemianWrapsody

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score
from keras.utils import to_categorical

import xgboost as xgb

from catboost import CatBoostClassifier, Pool

import lightgbm as lgb

from utilities import join_x_with_y_split

def instance_to_df(instance, features):
    return pd.DataFrame(instance.reshape(1,-1), columns=features)
    

raw = load_iris()
data = raw.data
target = raw.target
features = raw.feature_names


train_x, test_x, train_y, test_y = train_test_split(data, target)

clf = RandomForestClassifier(n_estimators=200)
clf.fit(train_x, train_y)
sk_score = clf.score(test_x, test_y)

keras_train_y = to_categorical(train_y)
keras_test_y = to_categorical(test_y)

#train/load keras model
if "best_model_c.h5" in os.listdir():
    model = load_model("best_model_c.h5")
else:
    model = Sequential()
    model.add(Dense(128, input_dim=len(features), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
    es = EarlyStopping(monitor='val_loss', patience=75, verbose=1)
    model.fit(train_x, keras_train_y, epochs=4000, validation_split=0.1, callbacks=[es])
    model.save("best_model_c.h5")
keras_score = model.evaluate(test_x, keras_test_y)

#train/load xgboost model
predictor = xgb.XGBClassifier(max_depth=4, objective="multi:softmax", num_class=3)
predictor.fit(train_x, train_y)

if "best_cat_model" in os.listdir():
    guessbox = CatBoostClassifier()
    guessbox.load_model("best_cat_model")
else:
    cat_train = Pool(train_x, label=train_y)
    
    guessbox = CatBoostClassifier(iterations=100, 
                              depth=4, 
                              learning_rate=1, 
                              loss_function='MultiClass')
                              
    guessbox.fit(train_x, train_y)
    guessbox.save_model("best_cat_model")

#TRAIN AND SCORE THE MODEL (LightGBM)
if "best_lgm_model_c" in os.listdir():
    sasquatch = lgb.Booster(model_file='best_lgm_model_c')
else:
    lgb_train = lgb.Dataset(train_x, label=train_y, feature_name=features)
    params = {'num_leaves': 30, "objective":"multiclass", "num_class": 3}
    sasquatch = lgb.train(params, lgb_train, 100)
    sasquatch.save_model("best_lgm_model_c")


print("The sklearn score is {}\n".format(sk_score))
print("The keras score is {}\n".format(keras_score))

model = BohemianWrapsody(clf, features, "classification")

print(model.predict_proba(test_x[12]))

xai = XAI(model=model, train_x=train_x, train_y=train_y, mode="classification", features=features, n_classes=3)

#RUNNING EXPLAINERS
print("Shapley Values:")
shap, expect = xai.shap.shapley_tree(test_x)
xai.shap.summary_plot(shap, test_x, "shapsummary-pdp")

print("\nCounterfactuals")
xai.counterfactual_explanation(test_x[2].reshape(1,-1))

print("Single PDP:")
xai.pdp.calculate_single_pdp(feature_name="petal width (cm)", filename="pw-iris-pdp")

print("\nDouble PDP:")
xai.pdp.calculate_double_pdp(feature_names=["petal width (cm)", "petal length (cm)"], filename="pw-pl-iris-pdp")

print("ALE plot:")
xai.ale.plot_single("petal width (cm)", filename="pw-iris-ale")

print("\nLIME explanation:")
exp = xai.lime.tabular_explanation(test_x[0], filename="iris-zero-lime")

print("\nTree surrogate:")

xai.surrogate.global_surrogate("iris-tree-surrogate", max_leaves=15)

print("\nFeature Importance:")
imp = xai.feature_importance(method="pred_variance", plot=True, filename="feat-imp-iris.png")

print("\nCorrelation Matrix")
hi = xai.correlation(filename="iris-corr.png", plot=True)

print("\nFeature Distribution")
xai.feature_distribution("feature-dist-iris")

print("\nAnchor Explanation")
xai.anchor.anchor_explanation(test_x[1])

print("\nSummarise feature")
xai.summarise_feature("petal width (cm)")