# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:46:32 2019

@author: smcwilliams1
"""
from sklearn.metrics import accuracy_score, r2_score

from utilities import create_permuted, is_shapley_tree, sample_from_large_data
from keras.utils import to_categorical

import lime
import lime.lime_tabular, lime.lime_image, lime.lime_text

import shap

from anchor import utils, anchor_tabular

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pdpbox import pdp, get_dataset, info_plots

class WrapGod():
    """
    For wrapping a model such that the predict function has a consistent name for all supported model types
    """
    def __init__(self, model, predict_fn):
        self.model = model
        self.predict_fn = predict_fn
        
    def predict(self, *args):
        return self.predict_fn(*args)

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
    """
    def __init__(self, model, train_x, train_y, mode, test_x=None, test_y=None, features=[], mem_save=False):
        self.model = model
        self.train_x, self.train_y = self._set_data(train_x, train_y)
        if test_x != None and test_y != None:
            self.test_x, self.test_y = self._set_data(test_x, test_y)
        else:
            self.test_x = None
            self.test_y = None
        self.mode = mode
        self.mem_save = mem_save
        self.features = self._set_features(features)
        self.shap = Shap(self)
        self.surrogate = Surrogate(self)
        self.pdp = PDP(self)
        self.lime = Lime(self)
        self.anchor = Anchor(self)
    
    def _set_features(self, features):
        if len(features) == 0:
            final_features = ["Feature {}".format(chr(x)) for x in range(65,65+len(self.train_x[0]))]
        elif len(features) != len(self.train_x[0]):
            raise ValueError("Feature name list must be the same length as actual list of features")
        else:
            final_features = features
        return final_features
        
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
    
    def feature_importance(self):
        """
        Calculates which features contribute the most towards a change in the predicted output.
        
        -------------------------
        
        Returns:
        
        importance: a sorted pd.Series of all the features (least to most)
        """
        if self.mode == "regression":
            classifier_fn = self.model.predict
        else:
            classifier_fn = self.model.predict_proba
        
        interpreter = Interpretation(training_data=self.train_x, training_labels=self.train_y, feature_names=self.features)
        annotated_model = InMemoryModel(classifier_fn, examples=self.train_x, feature_names=self.features)
        importance = interpreter.feature_importance.feature_importance(annotated_model)
        
        print("\n\n" +str(importance))
        
        return importance
    
class Shap():
    """
    Subclass of XAI
    
    
    Contains functionality for using game theory's Shapley value technique to explain the impact 
    each feature has on the prediction.
    """
    def __init__(self, xai):
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.xai = xai
        self.mode = xai.mode

    def shapley_tree(self, instances, feature_dependence="tree_path_dependent"):
        """
        Args:
        
            instances: 2d array-type object containing feature data to be explained
            
            feature_dependence: string defining method by which to analyse tree. If "independent" then data is required by TreeExplainer module
        
        -------------------------
        Returns:
        
            shap_values: in regression cases will return an array-type object with dimensions (n samples x x features)
                         in classification cases it will return one of these for each class value
        
            expected_value: the expected (Average) value for which all the features interact upon
            
        """
        instances = np.array(instances)
        #TYPE-CHECKING
        assert isinstance(instances, np.ndarray), "instances must be in array form"
        assert isinstance(feature_dependence, str), "feature dependence must be passed as a string"
        #VALUE-CHECKING
        assert len(instances.shape) == 2, "Instances passed must be a 2D array or dataframe"
        assert instances.shape[1] == self.train_x.shape[1], "Passed instances must have same number of features that the model was trained on"
        assert feature_dependence == "tree_path_dependent" or feature_dependence == "independent", "feature dependence must take the form either 'independent' or 'tree_path_dependent'"
        
        background_data = shap.kmeans(self.train_x, 15)
        
        if feature_dependence != "tree_path_dependent":
            dataset = pd.DataFrame(self.train_x)
            dataset["Y"] = pd.Series(self.train_y, index=dataset.index)
            explainer = shap.TreeExplainer(self.model, data=background_data, feature_dependence=feature_dependence)
        else:
            explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(np.array(instances))
        return shap_values, explainer.expected_value

    def shapley_interaction(self, feature_dependence="tree_path_dependent"):
        """
        If feature_dependence argument is 'independent' then data is required by TreeExplainer module
        """
        #TYPE-CHECKING
        assert isinstance(feature_dependence, str), "feature dependence must be passed as a string"
        #VALUE-CHECKING
        assert feature_dependence == "tree_path_dependent" or feature_dependence == "independent", "feature dependence must take the form either 'independent' or 'tree_path_dependent'"
        
        if feature_dependence != "tree_path_dependent":
            dataset = pd.DataFrame(self.train_x)
            dataset["Y"] = pd.Series(self.train_y, index=dataset.index)
            explainer = shap.TreeExplainer(self.model, data=dataset, feature_dependence=feature_dependence)
        else:
            explainer = shap.TreeExplainer(self.model)
        shap_inter = explainer.shap_interaction_values(self.train_x)
        return shap_inter

    def shapley_kernel(self, instances, link="identity"):
        """
        Args:
        
            instances: 2d array-type object containing feature data to be explained
        
            link (str): "identity" or "logit"
        
        -------------------------
        Returns
        
            shap_values: in regression cases will return an array-type object with dimensions (n samples x x features)
                         in classification cases it will return one of these for each class value
        
            expected_value: the expected (Average) value for which all the features interact upon
        
        """
        instances = np.array(instances)
        #TYPE-CHECKING
        assert isinstance(instances, np.ndarray), "instances must be in array form"
        assert isinstance(link, str), "link argument must be passed as a string"
        #VALUE-CHECKING
        assert len(instances.shape) == 2, "Instances passed must be a 2D array or dataframe"
        assert instances.shape[1] == self.train_x.shape[1], "Passed instances must have same number of features that the model was trained on"
        assert link == "identity" or link == "logit", "link argument must take the form either 'identity' or 'logit'"
        
        background_data = shap.kmeans(self.train_x, 15)
        if self.mode == "regression":
            explainer = shap.KernelExplainer(self.model.predict, background_data, link=link)
        elif self.mode == "classification":
            explainer = shap.KernelExplainer(self.model.predict_proba, background_data, link=link)
        
        shap_values = np.array(explainer.shap_values(instances))
        
        if len(shap_values.shape) == 3 and shap_values.shape[0] == 1:
            shap_values = shap_values.reshape(instances.shape)
            
        return shap_values, explainer.expected_value
        
    def plot_shapley(self, shap_values, expected, filename=None, features = None):
        """
        Args:
        
            shap_values: 1D array-type object containing shapley values
        
            expected: float which represents the expected (~average~) value from which the shapley values are referring to
        
            filename: string denoting the desired name of the plot
        
            features (optional): 1D array-type containing feature data
        """
        shap_values = np.array(shap_values)
        expected = float(expected)
        if features is not None:
            features = np.array(features)
            assert isinstance(features, np.ndarray), "feature names must be in array form"
        #TYPE-CHECKING
        assert isinstance(shap_values, np.ndarray), "shapley values must be in array form"
        assert isinstance(expected, float), "expected value must be a float"
        assert isinstance(filename, str), "filename must be a string"
        assert np.array(list(shap_values)).dtype == np.dtype("float64"), "Values in the shap_values array must be in float format"
        
        if features is not None:
            features = np.array(features)
            assert len(shap_values.shape) == 1 and len(features.shape) == 1, "Saving the plot to file is only possible with one sample at a time"
            assert shap_values.shape == features.shape, "features passed must be the same shape as the shapley values"
        else:
            assert len(shap_values.shape) == 1, "Saving the plot to file is only possible with one sample at a time"
        fig = shap.force_plot(expected, shap_values, features=features, show=False, matplotlib=True)
        plt.show(fig)
        if filename == None:
            filename = input("What filename would you like for the shapley plot? (.png): ")
        plt.savefig(filename)
        
class Anchor():
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
        
    def anchor_explanation(self, instance, classes=[], categorical_names={}, validation_split=0.1, threshold = 0.95):
        """
        NOTE: Unsure if it's possible to specify categorical features
        
        Requires the categories to be named and specified if there are any.
        
        -------------------------
        
        Args:
            instance: 1D array-type object used to explore model
            
            classes: 1D array-type object describing names of classes
            
            categorical_names: dict with categorical feature index as key and list of possible values the feature can take as the value
            
            validation_split: float indicating the ratio for splitting training data
            
            threshold: float defining the threshold by which other instances are compared to
        """
        instance = np.array(instance)
        classes = np.array(classes)
        #TYPE-CHECKING
        assert isinstance(instance, np.ndarray), "instances must be in array form"
        assert isinstance(classes, np.ndarray), "class names must be in array form"
        assert isinstance(categorical_names, dict), "categorical names must be in dictionary form"
        assert 0 < validation_split < 1, "validation split ratio must be a value between 0 and 1"
        assert 0 < threshold < 1, "precision threshold must be a value between 0 and 1"
        #VALUE-CHECKING
        assert instance.shape == self.train_x[0].shape, "instance must be appropriate shape"
        assert len(categorical_names.keys()) <= len(self.xai.features), "cannot be more categorical features than there are features in total"
        
        #NOT REALLY NEEDED 
        dataset_folder = ''
        dataset = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder)
        
        if len(classes) == 0:
            classes = np.array(["Class {}".format(x) for x in range(1,self.model.n_classes_+1)])
        
        #CANT USE to_categorical WITH LARGE DATA BECAUSE OF A MEMORY ERROR 
        assert classes.shape[0] == self.model.n_classes_, "Class name list must be the same length as number of classes"
        
        cutoff = int((1-validation_split)*len(self.train_x))
        train_x = self.train_x[:cutoff]
        val_x = self.train_x[cutoff:]
        train_y = self.train_y[:cutoff]
        val_y = self.train_y[cutoff:]
        
        explainer = anchor_tabular.AnchorTabularExplainer(classes, self.xai.features, self.train_x, categorical_names)
        explainer.fit(train_x, train_y, val_x, val_y)

        prediction = self.model.predict(instance.reshape(1,-1))[0]
        if self.mode == "classification":
            prediction = classes[int(prediction)]
        
        result = explainer.explain_instance(instance, self.model.predict, threshold=threshold)
        
        print("IF\t{0}\nTHEN PREDICT {1} WITH {2}% ACCURACY.".format('\nAND\t'.join(result.names()), prediction, 100*result.precision()))
        
        return result
    
class Surrogate():
    """
    Subclass of XAI.
    
    Contains functionality for reinterpreting the model with a tree structure as a surrogate.
    Trees give a more intuitive understanding of how the model makes predictions as opposed with more 
    mathematically involved models.
    """
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode

    def global_surrogate(self, plot=True, filename=None, size=1500):
        """
        Uses the original model and the training data to remodel it as a tree.
        
        -------------------------
        Args:
        
            plot: boolean to indicate whether or not to save an image of the tree model to file
            
            filename: string describing the intended filename
            
            size: integer describing how many instances to sample from original dataset. If XAI.mem_save=False this will be ignored.
    
        """
        #TYPE-CHECKING
        assert isinstance(plot, bool), "plot argument must be a boolean"
        assert isinstance(filename, str), "filename argument must be a string"
        assert isinstance(size, int), "size argument must be a integer"
        #VALUE-CHECKING
        if self.xai.mem_save:
            assert size < len(self.train_x), "size argument must be less than the number of instances in training data"
        if filename != None:
            assert isinstance(filename, str), "filename must be a string"
        
        #determine predictor function based on learning method
        if self.mode == "regression":
            classifier_fn = self.model.predict
        else:
            classifier_fn = self.model.predict_proba
        
        if self.xai.mem_save:
            self.train_x, self.train_y = sample_from_large_data(self.train_x, self.train_y, size=size)

        #initialise the skater module with the relevant model/data information
        interpreter = Interpretation(training_data=self.train_x, training_labels=self.train_y, feature_names=self.xai.features)
        annotated_model = InMemoryModel(classifier_fn, examples=self.train_x, feature_names=self.xai.features)

        #initialise interpretable model and pass it the data
        surrogate_explainer = interpreter.tree_surrogate(oracle=annotated_model, seed=5)
        
        #THIS DOESN'T WORK WITH VERY LARGE DATA SETS
        surrogate_explainer.fit(self.train_x, self.train_y, use_oracle=True, prune='post', scorer_type='default')

        if plot:
            if filename is None:
                filename = input("What filename would you like for the tree depiction? (.png): ")
            surrogate_explainer.plot_global_decisions(enable_node_id=False, file_name='{}.png'.format(filename))
        return surrogate_explainer
        
class PDP():
    """
    Subclass of XAI. 
    
    Contains functionality for computing partial dependence and plotting them.
    """
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
        self.train_df = self.join_x_with_y_split()
        
    def join_x_with_y_split(self):
        """
        Joins the feature data and the targets. pdpbox expects the pd.dataframe targets to be split into one-hot vectors if
        the problem is classification.
        """
        if self.mode == "classification":
            train_y = self.train_y
            if len(train_y.shape) != 2:
                train_y = to_categorical(self.train_y)
            classes = ["Class {}".format(x) for x in range(1,len(train_y[0])+1)]
        elif self.mode == "regression":
            train_y = self.train_y
            if len(train_y.shape) == 1:
                classes = ["Target"]
            else:
                classes = ["Target {}".format(x) for x in range(1,len(train_y[0])+1)]
                
        training_df = pd.DataFrame(self.train_x, columns=self.xai.features)
        targets = pd.DataFrame(train_y, columns=classes)
        
        return training_df.join(targets)
        
    def calculate_single_pdp(self, feature_name=None, filename=None, size=2000):
        """
        Calculates the partial dependence of one of the features w.r.t. the predicted output and plots it.iter
        
        The plot is visualised as an ICE plot with the PDP plot overlaid on top. This gives a fuller picture of how the feature
        affects the output, rather than looking at the global average alone.
        
        -------------------------
        Args:
        
            feature_name: string detailing the feature to plot. If left as None then the option to select is given via the console.
            
            filename: string for saving the plot to file.
            
            size: integer describing the number of instances to sample. Only applied if xai.mem_save=True.
            
        """
        #TYPECHECKING
        assert isinstance(filename, str), "filename must be in string format"
        assert isinstance(size, int), "size must be an integer"
        if feature_name is not None:
            assert isinstance(feature_name, str), "feature_name must be a string"
        if self.xai.mem_save:
            assert size < len(self.train_x), "size parameter must be less than the number of instances"
        if feature_name is None:
            selected = False
            while not selected:
                print("\n{}\t{}".format("Index", "Feature Name"))
                for i, feature in enumerate(self.xai.features):
                    print("{}).\t{}".format(i, feature))
                
                max_index = len(self.xai.features)-1
                selection = input("\nSelect feature to plot: ")
                try:
                    selection = int(selection)
                    if 0 <= selection <= max_index:
                        feature_name = self.xai.features[selection]
                        selected = True
                    else:
                        print("Feature index not in range, try again.")
                except ValueError:
                    if selection in self.xai.features:
                        feature_name = selection
                        selected = True
                    else:
                        print("Please type the feature name in correctly or use the index.")
        elif not isinstance(feature_name, str):
            raise ValueError("Expecting the feature name to be passed as a string")
        
        if self.xai.mem_save:
            isolated = pdp.pdp_isolate(model=self.model, dataset=self.train_df.sample(2000), model_features=self.xai.features, feature=feature_name)
        else:
            isolated = pdp.pdp_isolate(model=self.model, dataset=self.train_df, model_features=self.xai.features, feature=feature_name)
            
        fig, axes = pdp.pdp_plot(isolated, feature_name=feature_name, center=True, x_quantile=True, ncols=6, plot_lines=True, frac_to_plot=1.0)
        plt.show(fig)
        if isinstance(filename, str):
            plt.savefig(filename)
        else:
            filename = input("What filename would you like for the pd plot? (.png): ")
            plt.savefig(filename)
        return fig, axes
    
    def calculate_double_pdp(self, feature_names=[], filename=None, size=2000):
        """
        Calculates the partial dependence of 2 of the features w.r.t. the predicted output, and plots them as a contour map.
        
        -------------------------
        Args:
        
            feature_names: list of strings detailing the features to plot. If left empty then the option to select is given via the console.
            
            filename: string for saving the plot to file.
            
            size: integer describing the number of instances to sample. Only applied if xai.mem_save=True.
            
        """
        feature_names = np.array(feature_names)
        #TYPECHECKING
        assert isinstance(feature_names, np.ndarray), "feature names must be in array format"
        assert isinstance(filename, str), "filename must be in string format"
        assert isinstance(size, int), "size must be an integer"
        #VALUE-CHECKING
        if self.xai.mem_save:
            assert size < len(self.train_x), "size parameter must be less than the number of instances"
        
        if len(feature_names) == 0:
            position = {0: "first", 1: "second"}
            while len(feature_names) < 2:
                print("\n{}\t{}".format("Index", "Feature Name"))
                for i, feature in enumerate(self.xai.features):
                    print("{}).\t{}".format(i, feature))
                
                max_index = len(self.xai.features)-1
    
                selection = input("\nSelect {} feature to plot: ".format(position[len(feature_names)]))
                try:
                    selection = int(selection)
                    if 0 <= selection <= max_index:
                        feature_names.append(self.xai.features[selection])
                    else:
                        print("Feature index not in range, try again.")
                except ValueError:
                    if selection in self.xai.features:
                        feature_names.append(selection)
                    else:
                        print("Please type the feature name in correctly or use the index.")
        elif len(feature_names) != 2:
            raise ValueError("You must pass 2 features")
        
        if self.xai.mem_save:
            isolated = pdp.pdp_interact(model=self.model, dataset=self.train_df.sample(2000), model_features=self.xai.features, features=feature_names)
        else:
            isolated = pdp.pdp_interact(model=self.model, dataset=self.train_df, model_features=self.xai.features, features=feature_names)

        fig, axes = pdp.pdp_interact_plot(isolated, feature_names=feature_names, plot_type='contour', x_quantile=True, ncols=2, plot_pdp=True)
        plt.show(fig)
        if isinstance(filename, str):
            plt.savefig(filename)
        else:
            filename = input("What filename would you like for the pd plot? (.png): ")
            plt.savefig(filename)
        return fig, axes
            
class Lime():
    """
    Subclass of XAI.
    
    Contains functionality for exploring the search space locally to attempt to explain the way that each feature
    contributes to the prediction.
    """
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
        self.features = xai.features
        self.train_df, self.classes = self.join_x_with_y_split()
        
    def join_x_with_y_split(self):
        """
        Joins the feature data and the targets. pdpbox expects the pd.dataframe targets to be split into one-hot vectors if
        the problem is classification.
        """
        if self.mode == "classification":
            train_y = self.train_y
            if len(train_y.shape) != 2:
                train_y = to_categorical(self.train_y)
            classes = ["Class {}".format(x) for x in range(1,len(train_y[0])+1)]
        elif self.mode == "regression":
            train_y = self.train_y
            if len(train_y.shape) == 1:
                classes = ["Target"]
            else:
                classes = ["Target {}".format(x) for x in range(1,len(train_y[0])+1)]
                
        training_df = pd.DataFrame(self.train_x, columns=self.xai.features)
        targets = pd.DataFrame(train_y, columns=classes)
        
        return training_df.join(targets), classes
    
    def as_pyplot_figure(self, expl, label=1, **kwargs):
        exp = expl.as_list(label=label, **kwargs)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        if self.mode == "classification":
            title = 'Local explanation for class %s' % expl.class_names[label]
        else:
            title = 'Local explanation'
        plt.title(title)
        plt.tight_layout()
        return fig
    
    def lime_explanation(self, instance, filename=None):
        """
        Takes as a parameter: instance object.
        The instance can be either an array-type object containing the features required for prediction
        or it can be a number representing the index of an instance in the training data.
        
        Returns explanation object which contains information about the instance.
        
        From this you can call exp.save_to_file(filename) to output the file as html.
        
        Or you can call exp.as_pyplot_figure() which returns a figure object which can
        then be shown with matplotlib.
        
        -------------------------
        Args:
        
            instance: 1D array-type object representing an instance for the model to evaluate
                        or integer referring to 
            
        -------------------------
        Returns:
            
            explanation: Explanation object describing the effect of each feature on the prediction
        """
        instance = np.array(instance)
        #TYPE-CHECKING
        assert isinstance(instance, np.ndarray), "instance passed must be in array form"
        categorical_features = np.argwhere(np.array([len(set(self.train_df.as_matrix()[:,x])) for x in range(self.train_df.shape[1])]) <= 10).flatten()

        explainer = lime.lime_tabular.LimeTabularExplainer(self.train_x, feature_names=self.features, class_names=self.classes, categorical_features=categorical_features, verbose=True, mode=self.mode)
        
        if self.mode == "classification":
            classifier_fn = self.model.predict_proba
        else:
            classifier_fn = self.model.predict

        exp = explainer.explain_instance(instance, classifier_fn)

        fig = self.as_pyplot_figure(exp)

        if filename == None:
            filename = input("What filename would you like for the lime explanation? (.png): ")
        plt.savefig(filename)
        return exp