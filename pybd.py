import numpy as np
import pandas as pd

from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PYBD():
    def __init__(self, xai):
        self.xai = xai
        self.model = xai.model
        self.train_x = xai.train_x
        self.train_y = xai.train_y
        self.mode = xai.mode
        
    def explain(self, instance):
        exp = Explainer(clf=self.model, data=self.train_x, colnames=self.xai.features)
        explanation = exp.explain(observation=instance, direction="up")
        self.visualize(explanation, filename="pybd_plot")
        
    def visualize(self, exp, figsize=(7,6), filename=None, dpi=90,fontsize=14):
        """
        Get user friendly visualization of explanation
        Parameters
        ----------
        figsize : tuple int
            Pyplot figure size
        filename : string
            Name of file to save the visualization. 
            If not specified, standard pyplot.show() will be performed.
        dpi : int
            Digits per inch for saving to the file
        """
        
        ##
        ## FUNCTION HAS BEEN BROUGHT IN BECAUSE I NEED TO SET plt.tight_layout() OTHERWISE THE PLOT LOOKS UGLY
        ##

        if not exp._has_intercept or not exp._has_final_prognosis:
            return

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        positions = list(range(len(exp._attributes)+2))
        previous_value = exp._baseline
        for (attr_info, position) in zip(exp._attributes, positions[1:]):
            cumulative = attr_info.cumulative+exp._baseline
            height=1
            left = previous_value if attr_info.contribution > 0 else cumulative
            width = abs(attr_info.contribution)
            color = "blue" if attr_info.contribution > 0 else "orange"
            rect = patches.Rectangle(
                xy=(left, position-0.5),width=width,height=height,alpha=0.8,color=color)
            ax.add_patch(rect)
            plt.errorbar(x=left, y=position, yerr=0.5, color="black")
            plt.errorbar(x=left+width, y=position, yerr=0.5, color="black")
            plt.text(left+width+0.15, y=position-0.2, size=fontsize,
                     s = exp._get_prefix(attr_info.contribution) + str(round(attr_info.contribution,2)))
            previous_value = cumulative
        
        #add final prediction bar
        rectf = patches.Rectangle(
            xy=(exp._baseline,positions[len(positions)-1]-0.5), 
            width=exp._final_prediction, 
            height=1, color="grey", alpha=0.8
        )
        ax.add_patch(rectf)
        ax.axvline(x=exp._baseline,mew=3,color="black",alpha=1)
        plt.errorbar(x=exp._baseline, y=len(positions)-1, yerr=0.5, color="black")
        plt.errorbar(x=exp._baseline+exp._final_prediction, y=len(positions)-1, yerr=0.5, color="black")
        plt.text(
            x=exp._baseline+exp._final_prediction+0.15,
            y=positions[len(positions)-1]-0.2,
            s=str(round(exp._final_prediction+exp._baseline,2)),size=fontsize,weight="bold")

        ax.set_yticks(positions[1:])
        ax.grid(color="gray",alpha=0.5)
        sign = "+" if exp._direction==ExplainerDirection.Up else "-"
        labels=[sign + "=".join([attr.name,str(attr.value)]) for attr in exp._attributes]+["Final Prognosis"]
        ax.set_yticklabels(labels,size=fontsize)
        
        all_cumulative = [attr.cumulative for attr in exp._attributes]
        leftbound = min([min(all_cumulative), 0]) + exp._baseline
        rightbound= max(max(all_cumulative)+exp._baseline,exp._baseline)
        plt.text(x=exp._baseline+0.15, y=positions[0]-0.2, s="Baseline = "+str(round(exp._baseline,2)),
                size=fontsize,color="red")

        ax.set_xlim(leftbound-1, rightbound+1)
        ax.set_ylim(-1,len(exp._attributes)+2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        approach = "\"up\"" if exp._direction==ExplainerDirection.Up else "\"down\""
        plt.title("Prediction explanation for "+approach+" approach")
        
        plt.tight_layout()
        #fig.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        #fig.subplots_adjust(hspace=0, wspace=0.1)
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename,dpi=dpi)