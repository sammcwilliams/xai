import numpy as np
import pandas as pd

import os

from pdpbox import pdp, get_dataset, info_plots

from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utilities import join_x_with_y_split, correlation_format, pdp_isolate_custom, pdp_interact_custom

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
        self.train_df = join_x_with_y_split(self.train_x, self.train_y, feature_names=self.xai.features, n_classes=self.xai.n_classes)
        self.ohe_categories = xai.ohe_categories
        
    def single(self, feature_name=None, filename=None, size=2000, custom_df=None):
        """
        Calculates the partial dependence of one of the features w.r.t. the predicted output and plots it.
        
        The plot is visualised as an ICE plot with the PDP plot overlaid on top. This gives a fuller picture of how the feature
        affects the output, rather than looking at the global average alone.
        
        -------------------------
        Args:
        
            feature_name: string detailing the feature to plot. If left as None then the option to select is given via the console.
            
            filename: string for saving the plot to file.
            
            size: integer describing the number of instances to sample. Only applied if xai.mem_save=True.
            
            custom_df: pd.DataFrame containing data to plot instead of the training data. Classes must be split into one hot vectors such that the number of columns
                        in the DataFrame = n_features + n_classes. The Dataframe's columns must also be labelled with the feature names that you intend reference with
                        the feature_name parameter.
        
        Returns:
            fig: pyplot Figure object of pd plot
            
            axes: pyplot Axes object of pd plot
        """
        
        #TYPECHECKING
        if filename != None:
            assert isinstance(filename, str), "filename must be in string format"
        assert isinstance(size, int), "size must be an integer"
        if custom_df is not None:
            assert isinstance(custom_df, pd.DataFrame), "custom_df must be a dataframe"
            assert len(custom_df.columns) == len(self.xai.features)+self.xai.n_classes, "Expecting custom_df to have {0}+{1} columns, got {2}".format(len(self.xai.features), len(self.xai.n_classes), len(custom_df.columns))
        if feature_name is not None:
            assert isinstance(feature_name, str), "feature_name must be a string"
        if feature_name not in self.xai.features and feature_name not in self.ohe_categories.keys():
            raise ValueError("feature_name not stored in feature list nor the onehot categories")
        if self.xai.mem_save and size >= len(self.train_x) and custom_df==None:
            size = len(self.train_x)-1
            print("size parameter: {0} exceeded the number instances in the training data. Setting to {1}".format(size, len(self.train_x)))
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
        
        if feature_name in self.ohe_categories.keys():
            feature_name = self.ohe_categories[feature_name]
        else:
            correlation_format(self.xai.correlation(plot=False), feature_name)

        if self.xai.mem_save:
            if custom_df is None:
                isolated = pdp_isolate_custom(model=self.model, dataset=self.train_df.sample(size), model_features=self.xai.features, feature=feature_name)
            else:
                isolated = pdp_isolate_custom(model=self.model, dataset=custom_df, model_features=self.xai.features, feature=feature_name)
        else:
            if custom_df is None:
                isolated = pdp_isolate_custom(model=self.model, dataset=self.train_df, model_features=self.xai.features, feature=feature_name)
            else:
                isolated = pdp_isolate_custom(model=self.model, dataset=custom_df, model_features=self.xai.features, feature=feature_name)
        if self.mode == "regression":
            print("PDPBox has interpreted feature {0} as having type: {1}".format(isolated.feature, isolated.feature_type))
        else:
            print("PDPBox has interpreted feature {0} as having type: {1}".format(isolated[0].feature, isolated[0].feature_type))
        fig, axes = pdp.pdp_plot(isolated, feature_name=feature_name, center=True, x_quantile=True, ncols=6, plot_lines=True, frac_to_plot=1.0)
        plt.show(fig)
        if not os.path.isdir("output/pdp"):
            os.mkdir("output/pdp")
        if isinstance(filename, str):
            plt.savefig("output/pdp/{}".format(filename))
        else:
            filename = input("What filename would you like for the pd plot? (.png): ")
            plt.savefig("output/pdp/{}".format(filename))
        return fig, axes
    
    def double(self, feature_names=[], filename=None, size=2000, custom_df=None):
        """
        Calculates the partial dependence of 2 of the features w.r.t. the predicted output, and plots them as a contour map.
        
        -------------------------
        Args:
        
            feature_names: list of strings detailing the features to plot. If left empty then the option to select is given via the console.
            
            filename: string for saving the plot to file.
            
            size: integer describing the number of instances to sample. Only applied if xai.mem_save=True.
            
            custom_df: pd.DataFrame containing data to plot instead of the training data. Classes must be split into one hot vectors such that the number of columns
                        in the DataFrame = n_features + n_classes. The Dataframe's columns must also be labelled with the feature names that you intend reference with
                        the feature_name parameter.
        
        Returns:
            fig: pyplot Figure object of pd plot
            
            axes: pyplot Axes object of pd plot
        """
        
        feature_names = list(feature_names)
        formatted_feat_names = feature_names
        #TYPECHECKING
        assert isinstance(feature_names, list), "feature names must be in list format"
        if filename != None:
            assert isinstance(filename, str), "filename must be in string format"
        if custom_df is not None:
            assert isinstance(custom_df, pd.DataFrame), "custom_df must be a dataframe"
            assert len(custom_df.columns) == len(self.xai.features)+self.xai.n_classes, "Expecting custom_df to have {0}+{1} columns, got {2}".format(len(self.xai.features), len(self.xai.n_classes), len(custom_df.columns))
        assert isinstance(size, int), "size must be an integer"
        temp_features = []
        for i, feature_name in enumerate(feature_names):
            if (feature_name not in self.xai.features) and (feature_name not in self.ohe_categories.keys()):
                raise ValueError("feature_name not stored in feature list nor the onehot categories")
            if feature_name in self.ohe_categories.keys():
                temp_features.append(self.ohe_categories[feature_name])
            else:
                temp_features.append(feature_name)
                correlation_format(self.xai.correlation(plot=False), feature_name)
        feature_names = temp_features
        print(feature_names)
        
        #VALUE-CHECKING
        if self.xai.mem_save and size >= len(self.train_x) and custom_df==None:
            size = len(self.train_x)-1
            print("size parameter: {0} exceeded the number instances in the training data. Setting to {1}".format(size, len(self.train_x)))
        
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
            if custom_df is None:
                isolated = pdp_interact_custom(model=self.model, dataset=self.train_df.sample(size), model_features=self.xai.features, features=feature_names)
            else:
                isolated = pdp_interact_custom(model=self.model, dataset=custom_df, model_features=self.xai.features, features=feature_names)
        else:
            if custom_df is None:
                isolated = pdp_interact_custom(model=self.model, dataset=self.train_df, model_features=self.xai.features, features=feature_names)
            else:
                isolated = pdp_interact_custom(model=self.model, dataset=custom_df, model_features=self.xai.features, features=feature_names)
        if self.mode == "regression":
            print("PDPBox has interpreted feature {0} as having type: {1}".format(isolated.features[0], isolated.feature_types[0]))
            print("PDPBox has interpreted feature {0} as having type: {1}".format(isolated.features[1], isolated.feature_types[1]))
        else:
            print("PDPBox has interpreted feature {0} as having type: {1}".format(isolated[0].features[0], isolated[0].feature_types[0]))
            print("PDPBox has interpreted feature {0} as having type: {1}".format(isolated[0].features[1], isolated[0].feature_types[1]))
        fig, axes = pdp.pdp_interact_plot(isolated, feature_names=formatted_feat_names, plot_type='contour', x_quantile=True, ncols=2, plot_pdp=False)
        plt.tight_layout()
        plt.show(fig)
        if not os.path.isdir("output/pdp"):
            os.mkdir("output/pdp")
        if isinstance(filename, str):
            plt.savefig("output/pdp/{}".format(filename))
        else:
            filename = input("What filename would you like for the pd plot? (.png): ")
            plt.savefig("output/pdp/{}".format(filename))
        return fig, axes