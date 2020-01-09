import numpy as np
import pandas as pd

import os

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

from utilities import sample_from_large_data

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

    def tree_surrogate(self, filename, plot=True, size=300, max_leaves=10):
        
        # DO SOMETHING ABOUT THE SIZE PARAMETER. memory error in the call to the fit function
        
        
        """
        Uses the original model and the training data to remodel it as a tree.
        
        -------------------------
        Args:
        
            plot: boolean to indicate whether or not to save an image of the tree model to file
            
            filename: string describing the intended filename
            
            size: integer describing how many instances to sample from original dataset. If XAI.mem_save=False this will be ignored.
            
            max_leaf_nodes: integer indicating the maximum number of leaf nodes. Gives customisation to the accuracy/interpretability trade-off.
    
        """
        #TYPE-CHECKING
        assert isinstance(plot, bool), "plot argument must be a boolean"
        assert isinstance(filename, str), "filename argument must be a string"
        assert isinstance(size, int), "size argument must be a integer"
        #VALUE-CHECKING
        if self.xai.mem_save:
            assert size < len(self.train_x), "size argument must be less than the number of instances in training data"
        assert isinstance(filename, str), "filename must be a string"
        if filename[-4:] == ".png" or filename[-4:] == ".PNG":
            filename = filename[:-4]
        
        #determine predictor function based on learning method
        if self.mode == "regression":
            classifier_fn = self.model.predict
        else:
            classifier_fn = self.model.predict_proba
        
        if self.xai.mem_save:
            self.train_x, self.train_y = sample_from_large_data(self.train_x, self.train_y, size=size)

        #initialise the skater module with the relevant model/data information
        interpreter = Interpretation(training_data=self.train_x, training_labels=self.train_y, feature_names=self.xai.features)
        annotated_model = InMemoryModel(classifier_fn, examples=self.train_x, feature_names=self.xai.features, target_names=self.xai.classes)

        #initialise interpretable model and pass it the data
        surrogate_explainer = interpreter.tree_surrogate(oracle=annotated_model, seed=5, max_leaf_nodes=max_leaves)
        
        #THIS DOESN'T WORK WITH VERY LARGE DATA SETS
        surrogate_explainer.fit(self.train_x, self.train_y, use_oracle=True, prune='post', scorer_type='default')
        
        
        
        if plot:
            if not os.path.isdir("output/surrogate"):
                os.mkdir("output/surrogate")
            if ".png" not in filename:
                filename += ".png"
            surrogate_explainer.plot_global_decisions(enable_node_id=False, file_name="./output/surrogate/{}".format(filename), show_img=True)
        return surrogate_explainer