=========================
Exploratory Data Analysis
=========================

Decision Boundary Analysis
--------------------------

This method uses the Ceteris Paribus, Nearest Neighbours algorithm to find the nearest n neighbours in the feature space of each instance in the training data. By comparing the target values
of the nearest neighbours and the original instance in question, you can use the difference in the predictions as a measurement of how well defined the decision boundary is.
This can then inform the data collection process.

For classification problems, this difference in predictions is measured as the number of neighbours that do not share the same class as the original. For regression
this measurement is done by cumulatively adding the absolute difference between original instance and the neighbours. 

The following snippet of code returns the top 20 instances with the highest metric described above. This top 20 is defined by ``top_n=20``, and the number of neighbours to explore for each
instance is given by ``n_nearest=10``.

.. code-block::

    boundaries = xai.decision_boundary_exploration(n_nearest=10, top_n=20)
    
    
Caveats
^^^^^^^

Don't bother with this for large datasets because it takes way too long.

    
