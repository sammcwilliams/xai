import pandas as pd
import numpy as np

from alepython import ale_plot

class ALE():
    """
    Subclass of XAI. 
    
    Contains functionality for computing the Accumulated Local Effects and plots them.
    """
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
    
    def plot_single(self, feature, filename, monte_carlo=False, bins=10, **kwargs):
        """
        Args:
        
            feature: string referring to feature stored in xai.features
        
            filename: string describing filename of saved plot
        
            monte_carlo: boolean indicating whether or not to compute monte carlo samples
        
            bins: integer describing how many times to split the feature space
        """
        #can only plot one feature at the moment
        #NO CATEGORICAL FEATURES
        
        assert isinstance(feature, str), "Feature must be passed as a string"
        assert isinstance(filename, str), "Filename must be passed as a string"
        assert isinstance(monte_carlo, bool), "monte_carlo argument must be a true or false boolean"
        assert isinstance(bins, int), "bins argument must be an integer"
        
        
        
        
        df = pd.DataFrame(self.train_x, columns=self.xai.features)
        if self.mode == "regression":
            ale_plot(self.model, df, (feature,), filename=filename, monte_carlo=monte_carlo, bins=bins, **kwargs)
        else:
            ale_plot(self.model, df, (feature,), filename=filename, monte_carlo=monte_carlo, bins=bins, predictor=self.model.predict_proba, **kwargs)
            