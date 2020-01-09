# XAI - A toolbox for explainability in AI

This is a python tool that aims to collate a variety of statistical techniques in order to gain clearer insights into the decision making process that plague the explainability of black box models.

### Installation

Just pop the explainme/ folder in your working directory. No pip or setup.py yet so don't go there.

## Usage

### Initialisation

```python
from explainme import XAI

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

raw = load_iris()

data = raw.data
target = raw.target
feature_names = raw.feature_names

train_x, test_x, train_y, test_y = train_test_split(data, target)

clf = RandomForestClassifier(n_estimators=150)

clf.fit(train_x, train_y)

xai = XAI(model=clf, train_x=train_x, train_y=train_y, mode="classification", features=feature_names, n_classes=3)
```
The parameter n_classes is only required if examining a classification problem, in the regression case this parameter can be dropped thusly:

```python
xai = XAI(model=clf, train_x=train_x, train_y=train_y, mode="regression", features=feature_names, mem_save=True)
```

In cases where the model/data is too large for your machine's RAM, set the parameter mem_save=True, as seen above. (This is not required for all regression problems).

### Foreword

As this package was developed in the AWS sandbox, all generated plots are automatically saved to file. As I expect this to be the standard usage,
functionality for windowed plots was not added.

### Feature Importance

To calculate the importance of the features simply call the feature importance function attached to the XAI object.

```python
importance = xai.feature_importance()
```

This returns a labelled pandas.Series object, sorted in ascending order.

### Shapley Values

The Shapley value is a technique brought in from game theory that treats each instance as a player, and treats the prediction as the payoff.

In essence what it does is it looks at how each feature may contribute to the prediction in relation to the average, by iterating through and permuting
features. After analysing the training data in this way, the instances you pass to it can be explained as features causing deviations from the average.

For ~~more~~ detail, see this [page](https://christophm.github.io/interpretable-ml-book/shapley.html).

#### Calculation

```python
shapley_values, expected = xai.shap.shapley_kernel(test_x[:50])
```

By passing the first 50 instances in the test set, the function will return both the shapley values for these 50 instances and the 
expected value(s) (the average).

If you're working with a classification problem, the number of classes will indicate the shape of the returned variables. If there are 3 classes
then the shapley values will take the shape (3, num_instances) and the expected values (3,). Intuitively this means that for each class,
if the features of an instance took the given values, how much did each feature contribute to predicting that class.

For regression the shapley values will take the shape (num_instances,) and the expected value will be a float.

#### Plotting

Plotting the shapley values of an instance requires the values themselves, the expected value (of that class if necessary) and a filename. Optionally
you can also pass the feature names too, which will serve as annotations for the plot.

```python
xai.shap.plot_shapley(shapley_values[0], expected, "test_data_shapley_plot.png", feature_names)
```

Since the shap subpackage doesn't have functionality for plotting multiple figures in matplotlib, you can only plot one instance at a time.
This is easily mitigated with a loop.

```python
for i in range(len(shapley_values)):
    xai.shap.plot_shapley(shapley_values[i], expected, "test_data_shapley_plot{0}.png".format(i), feature_names)
```

Looping is also relevant if you wish to look at every class prediction.

```python
for i in range(len(shapley_values)):
    for j in range(xai.n_classes):
        xai.shap.plot_shapley(shapley_values[i][j], expected[j], "test_data_shapley_plot{0}_class{1}.png".format(i,j), feature_names)
```

### Global Surrogate

Tree-like models have a lot more transparency in terms of interpretability, and as such this module's purpose is to approximate the predict function of the given model
and remodel it into tree form. The resulting image is the surrogate model in tree form, with the decision rules on each node.

```python
xai.surrogate.global_surrogate(filename="tree_surrogate.png")
```


### Ceteris Paribus

#### Nearest Neighbour

Using the training data supplied to the XAI object, an instance (existing in the training data or not) can be supplied to the nearest neighbour function
to find the most similar instances that the model was trained on. This can be used to probe where the decision boundaries are, as the algorithm does
not use the target values in the distance calculation.

```python
neighbours = xai.ceteris.neighbours(test_x[0], y=test_y[0], n=5)
```

The line above returns a pandas DataFrame with 5 rows containing the n nearest neighbours. The target y-value is optional however its
inclusion can be very informative.

#### Plot instances per feature (ICE)

This submodule has the capability to plot multiple predicted feature distributions for each instance, given that all the other features do not change. 
These are called Individual Conditional Expectation (ICE) plots. Ceteris Paribus translates to "all other things being equal".

```python
xai.ceteris.plot_instances(test_x[:20], test_y[:20], selected variables=["sepal length (cm)", "sepal width (cm)"])
```

The preceding line plots the first 20 instances of the test set, looking only at the sepal features of the iris dataset. At the moment this will save in a folder, an HTML file and accompanying 
javascript files, which when run in a browser will bring up an interactive front-end visualisation of the instances.


### Local Interpretable Model Explanation (LIME)

This approach in essence takes the instance passed to it, perturbs each feature such that it explores the local feature space
and learns a linear model around this area. It is important to note that the explanations it gives for values that a feature takes, can only be valid when all the other values stay somewhat similar.

Simply pass a 1D array to the following function, along with an optional filename to export the explanation and a class to associate the explanation with. By default, the plot_class variable
takes the value -1 and thus will use the predict function of the model to choose a class to explain the instance with.

```python
xai.lime.tabular_explanation(test_x[16], filename="lime_explanation.png")
```

### Partial Dependence Plots (PDP)

PDP's are a plot of a feature's effect on the prediction globally across all instances. In fact, they are an average of all the ICE plots, which in this implementation, have been overlaid on
top of the PDP. The PDP on its own could be misleading, for example, half of the instances could show an increase in the prediction and the other half a decrease.
This would result in the PDP showing a horizontal line, when in fact the effect is more nuanced.

Additionally, PDP's do not take into consideration the interactivity between features and assume independence. To help mitigate this, you can plot two features
against each other with respect to the prediction. This is capped at two due to a difficulty representing graphs in higher dimensions.

For a single feature, you need to pass the function the feature name as a string and also a filename. For two features, they need to be passed as a list or array type object.

```python
xai.pdp.calculate_single_pdp(feature_name="sepal length (cm)", filename="sepal_length_pdp.png")
xai.pdp.calculate_double_pdp(feature_names=["sepal length (cm)","sepal width (cm)"], filename="sepal_features_pdp.png")
```

In classification problems, a plot is produced for each class.

### Anchors

Anchors are an approach that trains an explainer on your dataset, which then can be given an instance to interpret. The interpretation 
you are given is a series of if-then statements that describe the conditions for the prediction. In contrast to other explainability methods that take
individual instances for explanations - anchors have the capability to predict globally on unseen data.

For example in the Iris dataset, when the anchor algorithms encounters the following instance:

```python
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                6.3               3.3                6.0               2.5
```

It outputs the following explanation:

```python
IF      petal length (cm) > 5.23
AND     sepal length (cm) > 5.80
THEN PREDICT Class 3 WITH 100.0% CONFIDENCE.
```

To run the function it requires only one instance to be passed, with the categorical features if necessary.

```python
xai.anchor.anchor_explanation(test_x[42], categorical_names={2:["Male","Female"]})
```

The key of the categorical_names parameter is the index of the feature, with the value being a list of all the possible values it can take.

### Aequitas

### alibi

### Add something

