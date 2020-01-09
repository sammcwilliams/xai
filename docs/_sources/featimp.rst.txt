==================
Feature Importance
==================

Usage
-----

To calculate the importance of the features simply call the feature importance function attached to the XAI object. As a parameter optionally you can choose between 4 different
methods for calculating the feature importance.

* "morris"
* "model_scoring"
* "pred_variance"
* "all" (default)

The "all" option will average the three options and is used by default.

For an explanation of the three options please see :ref:`featimp-output`

.. code-block:: python
    
    importance = xai.feature_importance(method="all")

This returns a labelled pandas Series object, sorted in ascending order.

.. code-block:: bash

    sepal width (cm)     0.021330
    sepal length (cm)    0.072867
    petal length (cm)    0.424218
    petal width (cm)     0.481585
    dtype: float64
    
Requisites
----------

blah blah blah

Accepted Data
-------------

blah blah blah

Accepted Models
---------------

blah blah blah

Dimensionality
--------------

blah blah blah

Caveats
-------

You will find that in regression problems, the explainer will often spit out explanations for which it is 0% confident. This is due to the continuous nature of regression, however
this can be worked around by first transforming the targets into classes before training your model. Increasing the number of you classes you choose here will give you a greater 
prediction accuracy but will inevitably give you a less useful Anchor explanation, either because the confidence rating will drop, or more conditions will be used.

Also don't call this function more than once unless you **really** like ValueErrors.


.. _featimp-output:

Output interpretation
---------------------

Just look mate