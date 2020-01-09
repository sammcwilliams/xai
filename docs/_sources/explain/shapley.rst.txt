==============
Shapley Values
==============

Usage
-----

The Shapley value is a technique brought in from game theory that treats each instance as a player, and treats the prediction as the payoff.

In essence what it does is it looks at how each feature may contribute to the prediction in contrast to the average, by iterating through and permuting
features. After analysing the training data in this way, the instances you pass to it can be explained as features causing deviations from the average.

For some semblance of detail, see this `page <https://christophm.github.io/interpretable-ml-book/shapley.html>`_

Calculation
^^^^^^^^^^^

.. code-block::

    shapley_values, expected = xai.shap.shapley_kernel(test_x[:50], n_samples=100)


By passing the first 50 instances in the test set, the function will return both the shapley values for these 50 instances and the 
expected value(s) (the average). 

The n_samples parameter is only used in conjuction with the kernel Shapley explainer as in reality, calculating the
Shapley values is incredibly expensive. Therefore instead of iterating over every instance in the data and doing an exponential calculation, simply sampling
the data using kmeans, with n_samples as the k. The exponential calculation is still happening but just less times.


.. code-block::

    shapley_values, expected = xai.shap.shapley_tree(test_x[:50], feature_dependence="tree_path_dependent")
    
The feature_dependence parameter in the tree explainer determines the way in which the Shapley algorithm considers correlated features. By following the 
path down the tree and seeing the distribution of instances that follow each path, you can find an approximation for the background data. This is indicated using the string
above, "tree_path_dependent". The alternative in the shap subpackage is "independent" where the dependencies between features are broken and the algorithm uses the background data
to explain non-linear transofrms of the model's output.

If you're working with a classification problem, the number of classes will indicate the shape of the returned variables. If there are 3 classes
then the shapley values will take the shape (3, num_instances) and the expected values (3,). Intuitively this means that for each class,
you are supposing the features of an instance take the given values, and the results show how much each feature value would contribute if 
the model had predicted that class.

For regression the shapley values will take the shape (num_instances,) and the expected value will be a float.

Plotting
^^^^^^^^

Plotting the shapley values of an instance requires the values themselves, the expected value (of that class if necessary) and a filename. Optionally
you can also pass the feature names too, which will serve as annotations for the plot.

.. code-block::

    xai.shap.plot_shapley(shapley_values[0], expected, "test_data_shapley_plot.png", feature_names)


Since the shap subpackage doesn't have functionality for plotting multiple figures in matplotlib, you can only plot one instance at a time.
This is easily mitigated with a loop.

.. code-block::

    for i in range(len(shapley_values)):
        xai.shap.plot_shapley(shapley_values[i], expected, "test_data_shapley_plot{0}.png".format(i), feature_names)


Looping is also relevant if you wish to look at every class prediction.

.. code-block::

    for i in range(len(shapley_values)):
        for j in range(xai.n_classes):
            xai.shap.plot_shapley(shapley_values[i][j], expected[j], "test_data_shapley_plot{0}_class{1}.png".format(i,j), feature_names)
            

.. figure:: ../../assets/doc_shapley.png
    :align: center
    
    Figure 1 : Box plot of the Shapley values associated with each feature.

The image above shows that for a prediction of Class 2, the likelihood that the model predicts this class given the features
increases a little bit from the average prediction because of the sepal width, and decreases the chances of predicting Class 2 due to every other feature - resulting in zero chance of the instance 
being in Class 2.

Requisites
----------

The model, the training data and the instances to be inspected are all required to calcluate Shapley values.

Accepted Data/Models
--------------------

Data
^^^^

blah blah blah

Models
^^^^^^

The shap subpackage deals with different models in different ways. There are 3 seperate explainers for each of the supported types.

TreeExplainer
"""""""""""""

The TreeExplainer module exploits the fact that calculating Shapley values from a tree is made easier. This is due to the conditional expectations that are implicit in each path to the leaf nodes. In 
this case, the accepted models are all sklearn tree models, XGBoost models, LightGBM models and CatBoost models.

DeepExplainer
"""""""""""""

The DeepExplainer module handles neural networks as mdoels and is compatible with TensorFlow, Keras and Pytorch based networks.

This module is not actually implemented yet but is due to be at some point.

KernelExplainer
"""""""""""""""

This explainer is for all other models. For this you merely need to pass the predict function and so theoretically any model can be used.

Dimensionality
--------------

blah blah blah

Caveats
-------

Is very computationally expensive.

Output interpretation
---------------------

Just look bro
