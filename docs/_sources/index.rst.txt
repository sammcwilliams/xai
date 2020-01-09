.. XAI documentation master file, created by
   sphinx-quickstart on Wed Jun  5 12:03:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


XAI - Ethical and Responsible AI Framework
==========================================

The XAI package is a multi-purpose library designed to facilitate the practice of ethical and responsible data science.
A machine learning model which impacts an individual, even indirectly, should adhere to principles of ethical AI including explainability, fairness, privacy, accountability, accuracy. 
The application of these principles occurs throughout the machine learning pipeline from data through to model deployment and this package serves as a convenience wrapper to apply established methods at each stage.


|

.. |clearer|  raw:: html

    <div style="clear:both"></div>
    
    
.. image:: ../assets/open-delivered-box.png
   :align: left
   :width: 100px
   :height: 100px
   
Explainable
---------

Intepretable models and explainers
""""""""""""""""""""""""""""""""""

|clearer|

The Explainable module provides information about interpretable machine learning models as substitutes for their black-box equivalents. Alternatively, the module provides a series of explainer methods to perform post-hoc analysis on models to explain their actions at local and global scales.

|

.. image:: ../assets/courthouse.png
   :align: left
   :width: 100px
   :height: 100px
   
Fairness
---------

Detection and mitigation of bias
""""""""""""""""""""""""""""""""

|clearer|


The Fairness module identifies bias in training data, models and predictions across various definitions of fairness and suggest remediations to remove or mitigate the effects of bias.

|
   

.. image:: ../assets/security-shield.png
   :align: left
   :width: 100px
   :height: 100px
   

Security
--------

Security and privacy measures
"""""""""""""""""""""""""""""

|clearer|


The Security module detects where models are susceptible to adverserial attacks that aim to leak data and deceive or reverse-engineer the model. The module offers extensions to preserve the privacy of individuals whose data the model may train or predict on.


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Contents:
   
   Getting Started <start>
   Explainability and Interpretability <explain>
   Fairness and Bias <fair>
   Security and Privacy <secure>




