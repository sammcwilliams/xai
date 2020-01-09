===============
Ceteris Paribus
===============

Nearest Neighbour
-----------------

Usage
^^^^^

Using the training data supplied to the XAI object, an instance (existing in the training data or not) can be supplied to the nearest neighbour function
to find the most similar instances that the model was trained on. This can be used to probe where the decision boundaries are, as the algorithm does
not use the target values in the distance calculation.

.. code-block::

    neighbours = xai.ceteris.neighbours(test_x[0], y=test_y[0], n=5)


The line above returns a pandas DataFrame with 5 rows containing the n nearest neighbours. The target y-value is optional however its
inclusion can be very informative.

.. code-block::

    Root instance:
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Class
    0                6.1               2.8                4.0               1.3      1
    
    Nearest neighbours:
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Class
    0                6.4               2.9                4.3               1.3      1
    1                5.7               2.8                4.1               1.3      1
    2                5.8               2.7                3.9               1.2      1
    3                5.7               2.9                4.2               1.3      1
    4                5.7               2.8                4.5               1.3      1
    
Requisites
^^^^^^^^^^

blah blah blah

Accepted Data
^^^^^^^^^^^^^

blah blah blah

Accepted Models
^^^^^^^^^^^^^^^

blah blah blah

Dimensionality
^^^^^^^^^^^^^^

blah blah blah

Caveats
^^^^^^^

1.  ICE plots, much like Partial Dependence plots, can produce invalid datapoints when the feature in question is correlated with others.

2.  Typical ICE plots use the prediction likelihood on the y-axis rather than the predicted class, and also seperate the classes into seperate plots. For visualisation
    purposes. For this type of ICE plot please see the PDP section where you can plot features per class, with the addition of an average running through it.

Output interpretation
^^^^^^^^^^^^^^^^^^^^^

Just look matey


Plot instances per feature (ICE)
--------------------------------

Usage
^^^^^

This submodule has the capability to plot multiple predicted feature distributions for each instance, given that all the other features do not change. 
These are called Individual Conditional Expectation (ICE) plots. Ceteris Paribus translates to "all other things being equal".

.. code-block::

    xai.ceteris.plot_instances(test_x[:20], test_y[:20])


The preceding line plots the first 20 instances of the test set, looking only at the sepal features of the iris dataset. At the moment this will save in a folder, an HTML file and accompanying 
javascript files, which when run in a browser will bring up an interactive visualisation of the instances.

.. figure:: ../assets/doc_ceteris.png
    :align: center
    
    Figure 1: Diagram showing the browser window containing the interactive ICE plots for the selected features.

Hovering your mouse over any of the instances in the bottom portion of the window will single out that instance in each of the ICE plots. Similarly, when you hover over the line
in each of the plots, a summary for that instance appears describing the prediction for that feature.
    
Requisites
^^^^^^^^^^

blah blah blah

Accepted Data
^^^^^^^^^^^^^

blah blah blah

Accepted Models
^^^^^^^^^^^^^^^

blah blah blah

Dimensionality
^^^^^^^^^^^^^^

blah blah blah

Caveats
^^^^^^^

1.  ICE plots, much like Partial Dependence plots, can produce invalid datapoints when the feature in question is correlated with others.

2.  Typical ICE plots use the prediction likelihood on the y-axis rather than the predicted class, and also seperate the classes into seperate plots. For visualisation
    purposes. For this type of ICE plot please see the PDP section where you can plot features per class, with the addition of an average running through it.

Output interpretation
^^^^^^^^^^^^^^^^^^^^^

Just look mate
