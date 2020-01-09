===============
Getting Started
===============

Installation
------------

Just pop the explainme/ folder in your working directory. No pip or setup.py yet so you will need to install all the required packages listed in requirements.txt

This is done by navigating to the directory with the requirements.txt file in it and running the following

.. code-block:: bash
    
    pip install -r requirements
    
If you encounter any dependency errors, ignore them. Some of the subpackages used are minimally maintained to say the least so a couple of them are adamant that you cannot use
the latest *matplotlib* with with *lime* for example - when you definitely can.

Initialisation
--------------

The following is a short example of the setup needed to use this tool. 

Firstly it's important to import the tool itself. Alongside this I am importing a simple model, a commonly used dataset and a prebuilt 
function that sklearn offers, to split the data.

Throughout this documentation there will be references to one of two datasets - Iris and one pertaining to bike rentals. A description of both are in :ref:`explain-resources`.

.. code-block:: python

    from explainme import XAI
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
The following snippet simply loads the data and assigns the various components of it to variables in memory. 

The data component is in the form of a numpy array with the shape ``(num_instances, num_features)``, however as long as it is formatted in this shape, you may also use python lists when 
initialising the XAI object if you really want. To be clear from here onwards, the usage of the word features will describe all columns of the data *not including* the target variable.

The datatype constraints of the target is the same as with the data, except it takes the shape ``(num_instances,)``

The feature_names describes each column of the data in a numpy array. This again can be in list/array/pd.Series format if you like.

.. code-block::

    raw = load_iris()
    
    data = raw.data
    target = raw.target
    feature_names = raw.feature_names
    
Here we split the data into training and test sets in the ratio 3:1, initialise the model and train it on the training data. 
    
.. code-block::
    
    train_x, test_x, train_y, test_y = train_test_split(data, target, train_size=0.75)
    
    clf = RandomForestClassifier(n_estimators=150)
    
    clf.fit(train_x, train_y)
    
To initialise the XAI object, the model is required, as is both the X and Y components of the **training** data and then whether this is a classification or regression problem. Ideally you would supply 
the feature names too as it makes the tool much easier to use later when you are trying to refer to features. If your time is too precious you can settle for "Feature A", "Feature B" etc.

.. code-block::
    
    xai = XAI(model=clf, train_x=train_x, train_y=train_y, mode="classification", features=feature_names, n_classes=3)

The parameter n_classes is only required if examining a classification problem, in the regression case this parameter can be dropped thusly (defaults to 1):

.. code-block::

    xai = XAI(model=clf, train_x=train_x, train_y=train_y, mode="regression", features=feature_names, mem_save=True)


In cases where the model/data is too large for your machine's RAM, set the parameter ``mem_save=True``, as seen above. (This is not exclusive to regression problems).