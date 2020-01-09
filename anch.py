import numpy as np
import pandas as pd
from anchor import utils, anchor_tabular

from utilities import reg_to_bin
from keras.utils import to_categorical

from pprint import pprint

class Anchor():
    """
    Subclass of XAI. 
    
    Contains functionality for using an instance to compute human-interpretable explanations with confidence.
    """
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
        
    def anchor_explanation(self, instance, classes=[], categorical_names={}, validation_split=0.1, threshold = 0.95, show=True):
        """
        NOTE: Unsure if it's possible to specify categorical features
        
        Requires the categories to be named and specified if there are any.
        -------------------------
        
        Args:
            instance: 1D array-type object used to explore model
            
            classes: 1D array-type object describing names of classes
            
            categorical_names: dict with categorical feature index as key and list of possible values the feature can take as the value
            
            validation_split: float indicating the ratio for splitting training data
            
            threshold: float defining the precision threshold 
            
            show: boolean indicating whether to print the explanation or not
            
        Returns:
            result: anchor explanation object with all necessary details of the explanation
            
            result_str: string describing the explanation in interpretable English
        """
        ###################
        # 
        # THROWS ERROR WHEN FUNCTION IS CALLED TWICE
        # REALLY FUCKIN CONFUSING
        #
        ###################
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
            
        
        cutoff = int((1-validation_split)*len(self.train_x))
        train_x = self.train_x[:cutoff]
        val_x = self.train_x[cutoff:]
        train_y = self.train_y[:cutoff]
        val_y = self.train_y[cutoff:]
        
        if len(classes) == 0:
            classes = np.array(["Class {}".format(x) for x in range(1,self.xai.n_classes+1)])
        assert classes.shape[0] == self.xai.n_classes, "Class name list must be the same length as number of classes"
                
        explainer = anchor_tabular.AnchorTabularExplainer(classes, self.xai.features, self.train_x, categorical_names)
        explainer.fit(train_x, train_y, val_x, val_y)

        prediction = self.model.predict(instance.reshape(1,-1))[0]
        if self.mode == "classification":
            prediction = classes[int(prediction)]
        else:
            print("Using anchors for regression is not wise, as any unique contiguous area of the feature space is very small.")
        
        result = explainer.explain_instance(instance, self.model.predict, threshold=threshold)
        
        result_str = "IF\t{0}\nTHEN PREDICT {1} WITH {2}% CONFIDENCE.\nTHIS EXPLANATION APPLIES TO {3}% OF THE TRAINING DATA.".format('\nAND\t'.join(result.names()), prediction, 100*result.precision(), round(100*result.coverage(), 2))
        
        if show:
            print(result_str)
        
        del explainer
        
        return result, result_str
    