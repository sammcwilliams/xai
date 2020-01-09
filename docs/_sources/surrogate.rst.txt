================
Global Surrogate
================

Usage
-----

Tree-like models have a lot more transparency in terms of interpretability, and as such this module's purpose is to approximate the function of the given model
and remodel it into tree form. The resulting image is the surrogate model in tree form, with the decision rules on each node.

.. code-block::

    xai.surrogate.global_surrogate(filename="tree_surrogate.png", max_leaves=10)
    
The filename argument here is self-explanatory, and the max_leaves argument constrains the explainer to have a maximum of 10 leaf nodes in this example.
    
For the Iris dataset, the Random Forest Classifier defined in the Initialisation section is reinterpreted as the following decision tree. While I do understand that 
reinterpreting a tree as another surrogate tree is pointless, this works for all sklearn models so the princible is there.

.. figure:: ../assets/doc_surrogate.png
    :align: center
    
    Figure 1: Diagram of the model interpreted in tree form.


Requisites
----------

This explainer requires the model to be modelled, and the training data used to train the model.

Accepted Data/Models
--------------------

Data
^^^^

Because the explainer is using the original model to interpret the data, as long as the data is in the same format that you trained the model with, then the explainer should be able to understand it.

Models
^^^^^^

The subpackage Skater takes the model's predict function as a parameter, meaning that theoretically this technique can work with any model.

Dimensionality
--------------

blah blah blah

Caveats
-------

For large models/datasets, the memory requirements increase rapidly. However this is a technical issue and not theoretical, meaning it's very fixable.

The new surrogate model will have been trained on the model's response, and not the ground truth presented in the data. While this is clearly the aim of a surrogate model (to see inside the black box)
it is an important distinction.

It should also be made clear that complex, high-dimensional models like neural networks, get reduced in accuracy when reinterpreted as decision trees. This is exacerbated when the number of leaf nodes are limited
so that it's more human-readable.

Output interpretation
---------------------

As you can see in Figure 1, at each node there are various pieces of information.

* The criteria by which to navigate the tree - This is what defines the mechanics and traversal of the tree. Only present on non-leaf nodes.
* Gini - This is a metric used for splitting decision trees. If you are familar with the CART algorithm for building decision trees you will be familiar with this. Essentially the lower the gini index, the less the node would misclassify a random instance passing through it.
* MSE - In the regression case, the mean square error is the metric used for determining split points. If you don't know what MSE is then put this tool down.
* Samples - This is the proportion of instances in the dataset which would travel to/through that node.
* Value - Indicates the model's response at that node.
* Class - Gives the classification the model would output, induced by the value.