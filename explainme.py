# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:46:32 2019

@author: smcwilliams1
"""
from utilities import sample_from_large_data, colour_scale_corr, create_importance_graph, rename_df_columns, join_x_with_y, calc_rank_suffix, correlation_format, marsformat_to_function
from keras.utils import to_categorical
import sys, os

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret.blackbox import MorrisSensitivity

import xgboost as xgb

from catboost import Pool

from utilities import preserve
from interpret.perf import RegressionPerf

from pygam import LinearGAM, LogisticGAM

from pyearth import Earth

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pprint import pprint

from alibi.explainers import CounterFactual

from shapley import Shap
from anch import Anchor
from cetpar import CetPar
from lime_xai import Lime
from pdp import PDP
from surrogate import Surrogate
from ale import ALE
from themis import Themis

import time


class BohemianWrapsody():
    """Class that wraps ML models for use with XAI class"""
    def __init__(self, model, feature_names, mode):
        """
        Parameters:
        ----------
        model: A supported model (SKLearn, XGBoost, CatBoost, LightGBM, Keras)
        
        feature_names: array-type containing the names of the features in the model (arranged ordinally)
        
        mode: string describing the type of problem (regression or classification)
        """
        assert mode == "regression" or mode == "classification", "wrong mode, can only be regression or classification"
        if "xgboost.core.Booster" in str(type(model)):
            raise ValueError("Sorry but only xgboost models using the sklearn API are accepted. To avoid this you should initialise a model with XGBClassifier()/XGBRegressor() rather than training with xgb.train().")
        self.original = model
        self.feature_names = feature_names
        self.model_type = str(type(model))
        self.mode = mode
        
        self.original.feature_names=feature_names
        
        
    def predict(self, X, **kwargs):
        """
        Overrides the predict function of the wrapped models
        
        Parameters:
        ----------
        X: 2D array-type object representing feature data. [num_instances x num_features]
        
        Returns:
        -------
        results: 1D array object representing predictions produced by the model. [num_instances]
        """
        if "xgboost" in self.model_type:
            new_x =  xgb.DMatrix(X, feature_names=self.feature_names)
            results = self.original.predict(X, validate_features=False, **kwargs)
        elif "catboost" in self.model_type:
            results = self.original.predict(X)
        else:
            results = self.original.predict(X)
        
        #reshape back to 1 dimensions
        if results.ndim > 2 and "pred_contribs" not in kwargs.keys():
            raise ValueError("Your attitude stinks")
        elif results.ndim == 2:
            #keras models give the output of every node rather than the index (class number) for each instance
            new_res = []
            for output in results:
                new_res.append(np.argmax(output))
            results = np.array(new_res)
            
        return results
        
    def predict_proba(self, X):
        """
        Overrides the predict_proba function of the wrapped models which returns probabilities of each class.
        
        Parameters:
        ----------
        X: 2D array-type object representing feature data
        
        Returns:
        -------
        results: 2D array object representing predictions produced by the model in probabilities. [num_instances x num_classes]
        """
        if X.ndim == 1 and "catboost" not in self.model_type:
            X = X.reshape(1,-1)
        elif X.ndim > 2:
            raise ValueError("Too many dimensions friend :)")
            
        
        if "XGBClassifier" in self.model_type:
            results = self.original.predict_proba(X, validate_features=False)
        elif "XGBRegressor" in self.model_type:
            results = self.original.predict(X, validate_features=False)
        elif "sklearn" in self.model_type:
            results = self.original.predict_proba(X)
        elif "keras.engine" in self.model_type:
            results = self.original.predict(X)
        elif "catboost" in self.model_type:
            results = self.original.predict_proba(X)
        elif "lightgbm" in self.model_type:
            results = self.original.predict(X)
        return results

class XAI():
    """
    Explainable AI Class.
    
    Contains functionality for methods such as LIME, Shapley Values, Global Surrogates
    PDP/ICE plots, Anchors and feature importance.
    
    -------------------------
    
    Parameters:
    
    model: Trained model. Currently only supports sklearn.
    
    train_x: array-type (list, np.array, pd.series) with shape (n, features_length)
    
    train_y: array-type with shape (n,)
    
    mode: string describing whether the problem is a 'classification'/'regression' problem
    
    test_x: array-type (list, np.array, pd.series) with shape (n, features_length)
    
    test_y: array-type with shape (n,)
    
    features: train_x: array-type (list, np.array, pd.series) with shape feature_length
    
    mem_save: boolean indicating whether or not to sacrifice accuracy for memory usage
    
    n_classes: integer describing the number of classes that a classification problem has
    """
    def __init__(self, model, train_x, train_y, mode, test_x=None, test_y=None, features=[], mem_save=False, n_classes=1, class_names=[], ohe_categories={}):
        assert str(type(model)) == "<class 'explainme.BohemianWrapsody'>", "Make sure you're wrapping your model with the provided BohemianWrapsody class."
        self.model = model
        if len(class_names) != n_classes:
            self.model.classes_ = ["Class {}".format(i) for i in range(n_classes)]
        else:
            self.model.classes_ = class_names
        self.train_x, self.train_y = self._set_data(train_x, train_y)
        
        #creates output folder if it doesn't already exist
        if not os.path.isdir("output"):
            os.mkdir("output")
        if test_x != None and test_y != None:
            self.test_x, self.test_y = self._set_data(test_x, test_y)
        else:
            self.test_x = None
            self.test_y = None
        self.mode = mode
        if self.mode == "classification":
            assert n_classes > 1, "You must pass n_classes when loading a classification problem. For large datasets it becomes very expensive to infer"
        self.n_classes = n_classes
        if len(class_names) == 0:
            self.classes = self._set_default_classes()
        else:
            assert len(class_names) == n_classes, "The length of class_names does not equate with the n_classes parameter"
            self.classes = np.array(class_names)
        self.mem_save = mem_save
        self.features = self._set_features(features)
        self.ohe_categories = self._check_ohe(ohe_categories)
        
        assert set(self.ohe_categories.keys()).intersection(set(self.features)) == set(), "Please give the onehotencoding feature names (the dict keys) a different name to the features in the feature name list"
        
        self.shap = Shap(self)
        self.surrogate = Surrogate(self)
        self.pdp = PDP(self)
        self.lime = Lime(self)
        self.anchor = Anchor(self)
        self.ceteris = CetPar(self)
        self.ale = ALE(self)
        self.themis = Themis(self)
        
    def _check_ohe(self, ohe):
        assert isinstance(ohe, dict), "the ohe_categories must be in dictionary form"
        for key, arr in ohe.items():
            assert isinstance(arr, list) and isinstance(key, str), "the key value pairs in the dict must take the value (str:list), at least one entry in the dictionary has a value of type {}".format(str(type(arr)))
            for elem in arr:
                assert isinstance(elem, str), "The list of features that comprise the onehot encoded columns should be the same feature names defined in the XAI object"
        return ohe
    
    def _set_features(self, features):
        if len(features) == 0:
            final_features = ["Feature {}".format(chr(x)) for x in range(65,65+len(self.train_x[0]))]
        elif len(features) != len(self.train_x[0]):
            raise ValueError("Feature name list must be the same length as actual list of features")
        else:
            final_features = features
        final_features = np.array(final_features)
        return final_features
        
    def _set_default_classes(self):
        if self.n_classes == 1:
            classes = np.array(["Target"])
        else:
            classes = np.array(["Class {}".format(x) for x in range(self.n_classes)])
        return classes
        
    def _set_data(self, train_x, train_y):
        if type(train_x) == pd.core.series.Series:
            train_x = list(train_x)
        if type(train_y) == pd.core.series.Series:
            train_y = list(train_y)
        x = np.array(train_x)
        y = np.array(train_y)
        assert len(x.shape) == 2, "Feature values are the wrong shape"
        assert x.shape[0] == y.shape[0], "unequal values for features and targets. {} != {}".format(x.shape[0], y.shape[0])
        return x,y
    
    def gam_surrogate(self, show=True):
        """
        Trains a GAM on the training data in order to give an extra flavour of interpretability into the mix
        """
        
        if self.mode == "classification":
            gam = LogisticGAM().fit(self.train_x, self.train_y)
        elif self.mode == "regression":
            gam = LinearGAM().fit(self.train_x, self.train_y)
        else:
            raise ValueError("the mode has not been set correctly")
        if show:
            gam.summary()
        return gam
        
    def feature_importance(self, show=True, plot=False, filename=None, method="pred_variance"):
        """
        Calculates which features contribute the most towards a change in the predicted output.
        -------------------------
        Args:
            show: boolean indicating whether to print the importances out
            
        Returns:
            importance: a sorted pd.Series of all the features (least to most)
        """
        def morris():
            if self.mode == "classification":
                sensitivity = MorrisSensitivity(predict_fn=self.model.predict_proba, data=self.train_x, feature_names=self.features)
            else:
                sensitivity = MorrisSensitivity(predict_fn=self.model.predict, data=self.train_x, feature_names=self.features)
            sensitivity_global = sensitivity.explain_global(name="Global Sensitivity")
            
            #get importance from morris sensitivity and normalise it so that it sums to 1 (like the other two)
            morris_importance = pd.Series(sensitivity_global.data()["scores"], index=self.features)
            total_morris = sum(morris_importance)
            new_morris = pd.Series()
            for val, name in zip(morris_importance, morris_importance.index):
                new_morris[name] = val/total_morris
            return new_morris
        
        def pred_var():
            interpreter = Interpretation(training_data=self.train_x, training_labels=self.train_y, feature_names=self.features)
            annotated_model = InMemoryModel(classifier_fn, examples=self.train_x, feature_names=self.features)
            pv_importance = interpreter.feature_importance.feature_importance(annotated_model, n_jobs=1, progressbar=False)
            return pv_importance
            
        def model_scoring():
            interpreter = Interpretation(training_data=self.train_x, training_labels=self.train_y, feature_names=self.features)
            annotated_model = InMemoryModel(classifier_fn, examples=self.train_x, feature_names=self.features)
            ms_importance = interpreter.feature_importance.feature_importance(annotated_model, n_jobs=1, method="model-scoring", progressbar=False)
            return ms_importance
        
        methods = ("morris", "model_scoring", "pred_variance")
        assert method in methods, "method argument must be one of {}".format(methods)
        assert isinstance(show, bool), "show argument must be a boolean"
        if self.mode == "regression":
            classifier_fn = self.model.predict
        else:
            classifier_fn = self.model.predict_proba
            
        if self.mem_save or method == "morris":
            if self.mem_save:
                print("Using Morris sensitivity for efficiency as mem_save has been set")
            final_importance = morris()
        elif method == "model_scoring":
            final_importance = model_scoring()
        elif method == "pred_variance":
            final_importance = pred_var()
        else:
            raise ValueError("method parameter must be in {}".format(methods))
        final_importance = final_importance.sort_values()
    
        
        assert 0.98 < sum(final_importance) < 1.02, "Importances must sum to 1. Or at least be close"
        
        if show:
            print(str(final_importance))
        if plot:
            assert filename != None, "If you want to put this on a graph I'm going to need a filename thanks matey"
            create_importance_graph(final_importance, method, filename)
        
        return final_importance
        
    def mars_surrogate(self, filename_prefix, feature=None, surrogate=True, save=True):
        """
        Trains a more interpretable Multivariate Adaptive Regression Spline (MARS) model
        
        Parameters:
        ----------
        filename_prefix: string describing prefix for the filenames that the basis functions will be named after
        
        feature: optional string to print out the only the basis fuinctions relevant for that feature
        
        surrogate: boolean which determines whether the model will train on the data or be trained on what the xai.model predicts
        
        save: boolean determining whther to save the basis functions to a file
        
        Returns:
        -------
        
        model: a pyearth model
        """
        assert self.mode == "regression", "MARS is regression only, sorry that's life"
        model = Earth()
        if surrogate:
            y = self.model.predict(self.train_x)
        else:
            y = self.train_y
        model.fit(self.train_x, y)
        
        convert = {}
        for num, i in enumerate(self.features):
            convert["x{}".format(num)] = i
            
        #UNSURE IF WHAT THIS PRINTS IS WORTHWHILE
        
        #     idx = list(self.features).index(feature)
                
        #     #Plot the model
        #     y_hat = model.predict(self.train_x)
            
        #     plt.figure()
        #     plt.plot(self.train_x[:,idx],self.train_y,'r.')
        #     plt.plot(self.train_x[:,idx],y_hat,'b.')
        #     plt.xlabel('test_x')
        #     plt.ylabel('y')
        #     plt.title('Simple Earth Example')
        #     plt.show()
        #     plt.savefig("mars-{}.png".format(feature))
    
        model_params = marsformat_to_function(model.basis_, model.coef_[0], convert)
        
        params = np.array(model_params)
        
        if save:
            for i, bf in enumerate(params):
                if len(bf[2]) != 0:
                    idx = self.features.tolist().index(bf[2])
                    maxi = float(max(self.train_x[:,idx]))
                    mini = float(min(self.train_x[:,idx]))
                    x = np.linspace(mini, maxi, num=100)
                    string = "[{}*{} for i in x]".format(bf[0].replace(bf[2], "i"), bf[1])
                    y = eval(string)
                    plt.subplots()
                    plt.plot(x,y)
                    plt.title("Basis Function {}".format(i))
                    plt.xlabel(bf[2])
                    plt.ylabel("Basis Function Response")
                    filename="{}-basis-function{}".format(filename_prefix, i)
                    if not os.path.isdir("output/surrogate/mars"):
                        if not os.path.isdir("output/surrogate"):
                                os.mkdir("output/surrogate")
                        os.mkdir("output/surrogate/mars")
                    if ".png" not in filename:
                        filename += ".png"
                    plt.savefig("./output/surrogate/mars/{}".format(filename))
        
        
        if feature is not None:
            results = [(i[1],i[0]) for i in params if i[2] == feature]
            if len(results) == 0:
                print("Selected feature is not used in the model that's been trained.")
            else:
                print("Coefficients", "Basis Functions")
                for i in results:
                    print(i[0], i[1])
        
        return model
        
    def counterfactual_explanation(self, instance, **kwargs):
        cf = CounterFactual(self.model.predict_proba, shape=instance.shape, **kwargs)
        print(instance.shape)
        print(self.model.predict_proba(instance))
        explanation = cf.explain(instance)
        print(explanation)
        
    def summarise_instance(self, instance, folder_name, filename_suffix, categorical_names={}, y=None):
        """
        -------------------------
        Args:
        
            instance: 1D array-type object of length n_features describing an instance in the feature space
            
            folder_name: string describing name of the folder the files with be saved into
            
            filename_suffix: string to be appended to the filename. Will follow a string that describes the file, eg. pdp-[filename_suffix].png
            
            categorical_names: dict with categorical feature index as key and list of possible values the feature can take as the value
            
            y: target value associated with the instance
        
        Returns:
        
            anchor_result: Anchor explanation object
            
            neighbours: nearest neighbours to the inswtance is the training data feature space
            
            lime_exp: LIME explanation object describing local area around instance
            
            [shap_values, expected]: list with first element being the shapley values for the instance, the second element is the expected value (average)
                                            (beware that these elements may be nested arrays and so the shape of this output can be peculiar and inconsistent)
        """
        instance = np.array(instance)
        assert isinstance(instance, np.ndarray), "instance must be convertible to a numpy array"
        assert isinstance(folder_name, str), "folder name must be a string, please think twice next time"
        assert isinstance(filename_suffix, str), "filename suffix has to be in string form, quite obviously actually."
        assert isinstance(categorical_names, dict), "categorical_names must be a dictionary. Have you considered reading the documentation?"
        assert isinstance(y, int) or isinstance(y, float), "y must be a target value and therefore a numer. Consider converting class names to integers ??"
        
        assert len(instance.shape) == 1, "instance must be 1 dimensional"
        assert instance.shape[0] == self.features.shape[0], "instance must have same number of features as is specified in the XAI object"
        
        #Determines whether this folder already exists, and if so whether the user wants to continue and risk overwriting other files if they haven't changed
        #the filename_suffix parameter
        try:
            os.mkdir(folder_name)
        except FileExistsError:
            overwrite = input("You already have a folder named {0}. Continuing may cause overwritten files. \n\nPress n to quit. \nPress x to quit angrily. \nPress anything else to continue happily. \nChoice: ".format(folder_name)).lower()
            if overwrite == "n":
                sys.exit("Bye Bye.")
            elif overwrite == "x":
                sys.exit("\nNext time remember you have files in that folder you complete imbecile. Grow up.")
        
        #gets anchor result and writes it to a text file
        anchor_result, anchor_string = self.anchor.anchor_explanation(instance, categorical_names=categorical_names)
        with open("./{0}/anchor-{1}.txt".format(folder_name, filename_suffix), "w") as f:
            f.write(anchor_string)
        
        #gets the 10 nearest neighbours and writes it, along with the the the original instance, to a text file
        neighbours, root = self.ceteris.neighbours(instance, y=y)
        with open("./{0}/neighbours-{1}.txt".format(folder_name, filename_suffix), "w") as f:
            f.write("Root instance: \n\n")
            f.write(str(root)+ "\n\n\n")
            
            f.write("Nearest neighbours: \n\n")
            f.write(str(neighbours))
        
        lime_exp = self.lime.tabular_explanation(instance, filename="./{0}/lime-{1}".format(folder_name, filename_suffix))
        
        shap_values, expected = self.shap.shapley_kernel(instance.reshape(1, -1))
        if self.mode == "classification":
            prediction_idx = np.argmax(self.model.predict_proba(instance.reshape(1,-1)))
            self.shap.plot_shapley(shap_values[prediction_idx][0], expected[prediction_idx], filename="./{0}/shap-{1}".format(folder_name, filename_suffix))
        else:
            self.shap.plot_shapley(shap_values[0], expected, filename="./{0}/shap-{1}".format(folder_name, filename_suffix))

        return anchor_result, neighbours, lime_exp, [shap_values, expected]
        
    def summarise_feature(self, feature):
        """
        Creates a summary of a feature using a collection of methods and saved all the results into a folder
        -------------------------
        Args:
        
            feature: string describing feature to be explored
            
            folder_name: string describing name of folder that files will be stored into
        """
                
        assert isinstance(feature, str), "The feature should be a string. This string should be stored in the XAI object."
        assert feature in self.features, "I, the sentient being analysing this model, have determined that the feature you have given me is not stored in my database - please try again."
        
        print("Processing correlation matrix...")
        corr = self.correlation(filename="summary-correlation-matrix", plot=True, feature=feature)
        highly_correlated = correlation_format(corr, feature)
        print("Processing PDP...")
        self.pdp.calculate_single_pdp(feature_name=feature, filename="summary-pdp-{}".format(feature))
        print("Processing ALE...")
        self.ale.plot_single(feature, filename="summary-ale-{}".format(feature))
        print("Processing ICE plots...")
        # self.ceteris.plot_instances(self.train_x, self.train_y, selected_variables=[feature])
        print("Processing feature distribution...")
        self.feature_distribution(filename_prefix="summary-feat-dist-{}".format(feature), feature=feature)
        
        print("Processing feature importance...")
        imp = self.feature_importance(plot=True, filename="summary-feature-imp")
        feat_imp = imp[feature]
        idx = list(imp.where(imp == feat_imp).index).index(feature)
        rank = len(self.features)-idx
        
        output = "\nOut of {0} features, '{1}' has been calculated to be {2}% important. \nThis ranks it the {3}{4} most important of all the features".format(len(self.features), feature, int(100*feat_imp), rank, calc_rank_suffix(rank))
        print(output)
        
    
    def correlation(self, filename=None, plot=False, feature=None):
        """
        Calculates the correlation matrix for the training data using the Pearson Coefficient
        -------------------------
        Args:
        
            filename: string to be the name of the file that will be created from this calculation
        """
        if plot:
            assert filename != None, "filename can't be none bro"
        df = pd.DataFrame(self.train_x, columns=self.features)
        corr = df.corr()
        if feature != None:
            feats = list(self.features)
            idx = feats.index(feature)
            arr = np.array(corr)[idx].reshape(1,-1)
            corr = pd.DataFrame(arr, columns=self.features)
            if plot:
                arr = arr[0]
                arr = np.delete(arr, idx)
                color = colour_scale_corr(arr)
                plt.subplots()
                plt.bar(list(range(len(arr))), arr, color=color)
                removed_feat = np.delete(self.features, idx)
                plt.xticks(list(range(len(arr))), removed_feat, rotation=45)
                plt.title("Feature correlation for '{}'".format(feature))
                plt.xlabel("Features")
                plt.ylabel("Correlation coefficient [-1, 1]")
                plt.show()
                plt.tight_layout()
                if not os.path.isdir("output/misc"):
                    os.mkdir("output/misc")
                if ".png" not in filename:
                    filename = filename + ".png"
                plt.savefig("output/misc/{}".format(filename))
        else:
            if plot:
                plt.matshow(corr)
                plt.tight_layout()
                plt.title("Correlation Matrix")
                plt.show()
                plt.yticks(range(len(self.features)), self.features, rotation=60)
                if not os.path.isdir("output/misc"):
                    os.mkdir("output/misc")
                if ".png" not in filename:
                    filename = filename + ".png"
                plt.savefig("output/misc/{}".format(filename))
        return corr
        
    def reg_performance(self, x, y, filename, title="Performance"):
        x = np.array(x)
        y = np.array(y)
        #TESTS
        assert self.mode == "regression", "This performance metric is only available for regression problems."
        assert x.shape[0] == y.shape[0], "There must be the same number of instances as there are target values"
        assert x.shape[1] == len(self.features), "The number of features in the test data must match the number of features stored in XAI object"
        assert len(y.shape)== 1, "The target values must be stored in a 1 dimensional array"
        assert len(x.shape) == 2, "The feature data must be a 2 dimensional array. Ensure each instance is of the same length."
        assert isinstance(filename, str), "Filename must be passed as a string"
        assert isinstance(title, str), "Title argument must be passed as a string"
        
        
        perf = RegressionPerf(self.model.predict, data=self.train_x).explain_perf(x, y, name=title)
        print("Printing nonsense.....")
        if not os.path.isdir("output/misc"):
            os.mkdir("output/misc")
        preserve(perf, file_name="output/misc/{}".format(filename))
        
    def feature_distribution(self, filename_prefix, feature=None):
        """
        Retrains the data on an ExplainableBoostingRegressor from microsoft/interpret
        Calculates a histogram of the values the feature takes, as well as plotting the range of possible model outputs for that given
        
        -------------------------
        Args:
            filename_prefix: string to be appended to the filename. Will follow a string that describes the file, eg. pdp-[filename_suffix].png
            
            feature: string describing feature to be explored. Refers to the feature stored in the XAI object.
    
        """
        
        #############
        #
        # DOESNT WORK FOR CLASSIFICATION :/ :/ 
        # 
        # .... maybe it never will :(
        #
        #############
        
        
        if self.mode == "classification":
           print("Cannot currently do classification as the EBM classifier doesn't know how to do it correctly.")
        else:
            ebm = ExplainableBoostingRegressor(feature_names=self.features)
            ebm.fit(self.train_x, self.train_y)
            ebm_global = ebm.explain_global()
            
            print("Printing nonsense.....")
            if feature is None:
                print("Calculating distribution for all features")
                if not os.path.isdir("output/misc"):
                    os.mkdir("output/misc")
                for i, feature in enumerate(self.features):
                    preserve(ebm_global, selector_key=i, file_name="output/misc/{0}-{1}".format(filename_prefix,feature))
            else:
                assert feature in self.features, "Passed feature must be in stored list of features in XAI object"
                idx = list(self.features).index(feature)
                if not os.path.isdir("output/misc"):
                    os.mkdir("output/misc")
                preserve(ebm_global, selector_key=idx, file_name="output/misc/{0}-{1}".format(filename_prefix,feature))
                
    
    def decision_boundary_exploration(self, sample_size=2000, n_nearest=3, top_n=20):
        """
        -------------------------
        Args:
        
            sample_size: integer describing the number of instances to sample when xai.mem_save has been set
            
            n_nearest: integer describing the number of neighbours to search for when exploring the decision boundary. Will take some fine-tuning, as 5 might be too much, or too little.
            
            top_n: integer describing the top n number of instances that you wish to be returned
        
        Returns:
        
            main_list: a sorted array of instances that satisfy the decision boundary criteria
        """
        if self.mem_save:
            print("Don't bother as you'll be here all day")
            return -1
        if self.mem_save and sample_size < len(self.train_x):
            np.random.seed(42)
            boundary_x = self.train_x[np.random.choice(self.train_x.shape[0], sample_size, replace=False)]
            np.random.seed(42)
            boundary_y = np.random.choice(self.train_y, sample_size, replace=False)
            df = join_x_with_y(boundary_x, boundary_y, self.features)
        else:
            df = join_x_with_y(self.train_x, self.train_y, self.features)

        targ = np.array(df["Target"])
        feat = np.array(df[df.columns.difference(["Target"])])
        
        length = len(targ)
        
        count = 0
        main_list = []
        if self.mode == "regression":
            for x, y in zip(feat, targ):
                t0 = time.time()
                neighbours, _ = self.ceteris.neighbours(x=x, y=y, n=n_nearest, show=False)
                t1 = time.time()
                print("That took {} seconds".format(t1-t0))
                sublist = [x]
                total = 0
                for i in list(neighbours["Target"]):
                    total += abs(i-y) 
                sublist.append(total)
                main_list.append(sublist)
                count += 1
        else:
            for x, y in zip(feat, targ):
                neighbours, _ = self.ceteris.neighbours(x=x, y=y, n=n_nearest, show=False)
                sublist = [x]
                total = 0
                for i in list(neighbours["Target"]):
                    if i!=y:
                        total += 1
                sublist.append(total)
                main_list.append(sublist)
                    
        main_list = sorted(main_list, key=lambda x: x[1])
        main_list = np.flip(np.array(main_list[-top_n:]))
        return main_list