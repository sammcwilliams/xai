import numpy as np
import pandas as pd

from ceteris_paribus.explainer import explain
from ceteris_paribus.plots.plots import plot
from ceteris_paribus.profiles import individual_variable_profile
from ceteris_paribus.select_data import select_neighbours

from utilities import rename_df_columns

class CetPar():
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
        self.features = xai.features
    
    def plot_instances(self, x, y, selected_variables=None, size=2000):
        """
        -------------------------
        Args:
        
            x: 2D array-type object containing the instances to plot the ICE plots for
            
            y: target values associated with the x instances
            
            selected variables: 1D array-type object containing strings associated with the feature names stored in the XAI object. If none given then
                                they'll be denoted "Feature A", "Feature B" etc.
                                
            size: integer indicating how many instances to sample for plotting
            
        -------------------------
        """
        #TYPE-CHECKING
        if selected_variables != None:
            assert not isinstance(selected_variables, str), "You must pass the selected variables as an iterable like a list/array/tuple. Passing as a string messes with my zen dude."
            selected_variables = list(selected_variables)
            assert isinstance(selected_variables, list), "selected features could not be converted into list form"
        if len(y.shape) == 1 and y.shape[0] == 1:
            y = y[0]
        x = np.array(x)
        assert isinstance(x, np.ndarray), "x must be in array form"
        #VALUE-CHECKING
        assert 0 < len(x.shape) < 3 and len(y.shape) < 2, "Too many dimensions. x -> {}\t y->{}".format(x.shape, y.shape)
        if len(x.shape) == 2:
            assert x.shape[0] == y.shape[0], "shape of instance array and target must be correlated. expected {} targets, got {}".format(x.shape[0], y.shape[0])
        elif len(x.shape) == 1:
            assert len(y.shape) == 0, "shape of instance array and target must be correlated. expected 1 target"
            
        #gets name of any sklearn model
        model_label = str(self.model)[0:str(self.model).index("(")]
        length = self.train_x.shape[0]
        if self.xai.mem_save and size < self.train_x.shape[0]:
            for i in range(size):
                np.random.seed(42)
                background_x = self.train_x[np.random.choice(self.train_x.shape[0], size, replace=False)]
                np.random.seed(42)
                background_y = np.random.choice(self.train_y, size, replace=False)
        else:
            background_x = self.train_x
            background_y = self.train_y
        explainer = explain(self.model, self.features, background_x, background_y, predict_function=self.model.predict, label=model_label)
        if len(x) > 20:
            print("WARNING: For large number of instances to plot, compute time increases linearly. 10 instances takes roughly 7s for any size of training data with this implementation of ICE")
            
        profile = individual_variable_profile(explainer, x, variables=list(self.features), y=y)
        plot(profile, selected_variables=selected_variables)
        self.train_x = self.xai.train_x
        self.train_y = self.xai.train_y
        
    def neighbours(self, x, y=None, n=10, show=True):
        """
        -------------------------
        Args:
        
            x: 1D array-type object to find the neighbours of
            
            y: target value associated with the instance passed as x. Is ignored if not passed, as is only used for visual reference.
            
            n: integer describing the number of similar neighbours to return
            
            show: boolean to indicate whether to print the neighbours before returning
            
        -------------------------
        Returns:
            
            neighbours: pd.Dataframe showing the n most similar instances in the training data
            
            example: pd.DataFrame of the instance passed as a parameter
        """
        x = np.array(x)
        assert isinstance(x, np.ndarray), "instance passed must be an array_type object"
        assert isinstance(n, int) and n > 0, "n must be a positive integer"
        assert isinstance(show, bool), "show argument must be a boolean mate"
        assert len(x.shape) == 1, "instance must be a 1 dimensional"
        if y is not None:
            assert isinstance(y, int) or isinstance(y, float), "y must be a number as it is supposed to be the target value. Consider converting class strings to class numbers ??"
        neighbours = select_neighbours(self.train_x, x, y=self.train_y, variable_names=self.features, selected_variables=self.features, n=n)
        
        neighb_y = pd.DataFrame(neighbours[1])
        
        neighb_y = neighb_y.rename({0: "Target"}, axis="columns")
        
        neighb_x = rename_df_columns(neighbours[0], self.features)
        neighbours = neighb_x.join(neighb_y)
        
        example = rename_df_columns(pd.DataFrame(x.reshape(1,-1)), self.features)
        if y != None:
            example = example.join(pd.DataFrame(y.reshape(1,-1)).rename({0: "Target"}, axis="columns"))
        
        if show:
            print("Root instance:")
            print(example)
            
            print("\nNearest neighbours:")
            print(neighbours)
            
        return neighbours, example