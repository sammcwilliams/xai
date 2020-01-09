=========
Summaries
=========

Summarising an instance
-----------------------

Since a lot of the methods used in this toolbox require an instance as an input, it seemed obvious to collate them all into one function. Calling the  ``summarise_instances`` function will return the following 
results:

* An **Anchor** explanation that the instance satisfies
* The **Nearest neighbours** to the instance in the feature space
* A **LIME** explanation of the instance
* A plot of how the **Shapley values** could describe the instance

The function below takes the results from these explanations and puts them in the folder "iris-tests". The top two will be saved as a .txt file and the bottom two will be saved as images.
Each file saved in this folder will be prefixed with the method used, eg. ``lime-instance-zero.png`` would contain the lime explaination.

.. code-block:: 

    xai.summarise_instance(test_x[0], folder_name="iris-tests", filename_suffix="instance-zero", y=test_y[0])
    
    
Summarising a feature
---------------------

Calling the ``summarise_features`` function will return the following results:

* An **ALE plot** of the feature
* A **Partial Dependence Plot (PDP)** of the feature
* An interactive **ICE plot** of the feature
* An insight into the **importance** of the feature
* The **distribution** of the feature

Much like the ``summarise_instance`` function described above, this function takes a folder_name as a parameter, however the filename suffix will not be needed, as the feature serves as an ample enough 
unique identifier. The ALE and PD plots will be exported as images, the interactive ICE plots and feature distributions will be exported as html files, and the feature importance will be saved into a txt file.

.. code-block:: 

    xai.summarise_feature(feature="petal length (cm)", folder_name="iris-tests")
