import numpy as np
import pandas as pd
import lime

from stolen import LimeTabularExplainer

from keras.utils import to_categorical
from utilities import join_x_with_y_split

import matplotlib.pyplot as plt

import os

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
        self.train_df = join_x_with_y_split(self.train_x, self.train_y, feature_names=self.features, n_classes=self.xai.n_classes)
        self.classes = xai.classes
    
    def as_pyplot_figure(self, expl, label=1, **kwargs):
        """
        Function taken straight from the lime package. Adopted because plt.tight_layout() needed to be called to stop the plot being very very ugly.
        -------------------------
        Args:
        
            expl: Lime explanation object to be converted to pyplot
            
            label: integer describing which class to plot.
        -------------------------
        Returns:
            
            fig: pyplot Figure object with lime explanation in it.
        """
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
            title = 'Local explanation for %s' % expl.class_names[label]
        else:
            title = 'Local explanation'
        plt.title(title)
        plt.tight_layout()
        return fig
    
    def tabular_explanation(self, instance, plot_class=-1, filename=None):
        """
        -------------------------
        Args:
        
            instance: 1D array-type object representing an instance for the model to evaluate
            
            plot_class: integer describing which class to explain the instance with, as an index. Defaults to -1 which
                        uses the predict function to choose the class to explore.
            
            filename: string denoting the filename by which to save the .png file to
            
        -------------------------
        Returns:
            
            exp: Lime explanation object describing the effect of each feature on the prediction
        """
        instance = np.array(instance)
        assert len(instance.shape) == 1, "You must only pass 1 instance. By my calculations you have tried to pass me a 2D array??? no thanks"
        assert instance.shape[0] == len(self.features), "You must pass me an instance with the same number of features that the model was trained on, otherwise what's the point"
        #TYPE-CHECKING
        assert isinstance(instance, np.ndarray), "instance passed must be in array form"
        assert isinstance(plot_class, int), "the plot_class variable must be an integer"
        assert -1 <= plot_class < len(self.classes), "the maximum number of classes is {0}. You have passed {1}".format(len(self.classes), plot_class)
        
        categorical_features = np.argwhere(np.array([len(set(self.train_df.as_matrix()[:,x])) for x in range(self.train_df.shape[1])]) <= 10).flatten()

        explainer = LimeTabularExplainer(self.train_x, feature_names=self.features, class_names=self.classes, categorical_features=categorical_features, verbose=True, mode=self.mode)
        
        if self.mode == "classification":
            classifier_fn = self.model.predict_proba
            if plot_class == -1:
                prediction = classifier_fn(instance.reshape(1,-1))
                plot_class = np.argmax(prediction)
        else:
            classifier_fn = self.model.predict
            
        label_list = list(range(len(self.classes)))
        
        print(instance.shape)

        exp = explainer.explain_instance(instance, classifier_fn, labels=label_list)
        
        fig = self.as_pyplot_figure(exp, label=plot_class)

        if filename == None:
            filename = input("What filename would you like for the lime explanation? (.png): ")
        if not os.path.isdir("output/lime"):
            os.mkdir("output/lime")
        plt.savefig("output/lime/{}".format(filename))
        plt.show()
        return exp
        
    def text_explanation(self, instance):
        ## TODO
        pass