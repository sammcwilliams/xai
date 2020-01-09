# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:40:57 2019

@author: smcwilliams1
"""
import numpy as np
import pandas as pd

import sys, os
import matplotlib.pyplot as plt

from keras.utils import to_categorical

from contextlib import contextmanager

from interpret.visual.interactive import _preserve_output
import logging

from pdpbox.utils import (_check_model, _check_dataset, _check_percentile_range, _check_feature,
                    _check_grid_type, _check_memory_limit, _check_frac_to_plot, _make_list, _expand_default,
                    _plot_title, _calc_memory_usage, _get_grids, _get_grid_combos, _check_classes, _calc_figsize,
                    _get_string)
                    
                
from joblib import Parallel, delayed
from pdpbox.pdp_calc_utils import _calc_ice_lines, _calc_ice_lines_inter, _prepare_pdp_count_data
from pdpbox.pdp import PDPIsolate, PDPInteract

import lime

log = logging.getLogger(__name__)

def instance_to_df(instance, features):
    instance = instance.reshape(1,-1)
    instance = pd.DataFrame(instance, columns=features)
    return instance

def rename_df_columns(df, features):
    assert len(df.columns.values) == len(features), "don't be silly"
    mapper = {}
    for i, j in zip(df.columns.values, features):
        mapper[i] = j
    changed = df.rename(mapper, axis="columns")
    return changed

def sample_from_large_data(orig_x, orig_y, size=3000):
    max_idx = len(orig_x) -1
    for i in range(size):
        idx = np.random.randint(max_idx)
        
        x = orig_x[idx]
        if i != 0:
            x = x.reshape(1,-1)
            new_x = np.append(new_x, x, axis=0)
        else:
            new_x = x.reshape(1,-1)
        orig_x = np.delete(orig_x, idx, 0)
        
        y = orig_y[idx]
        if i != 0:
            y = y.reshape(1,-1)
            new_y = np.append(new_y, y, axis=0)
        else:
            new_y = y.reshape(1,-1)
        orig_y = np.delete(orig_y, idx, 0)
        
        max_idx -= 1
    return new_x, new_y
    
def join_x_with_y_split(x, y, feature_names, n_classes):
    """
    Joins the feature data and the targets. pdpbox expects the pd.dataframe targets to be split into one-hot vectors if
    the problem is classification.
    
        x: 2d array-type
        
        y: 1d array type
        
        feature_names: 1d array type with feature labels
    
        n_classes: integer referring to number of classes. 1 in regression cases.
    """
    if n_classes != 1:
        y = to_categorical(y)
        classes = ["Class {}".format(x) for x in range(n_classes)]
    else:
        classes = ["Target"]
            
    df = pd.DataFrame(x, columns=feature_names)
    targets = pd.DataFrame(y, columns=classes)
    
    return df.join(targets)
    
def graph_formula(formula):
    return eval(formula)
    
def join_x_with_y(x, y, feature_names):
    """
    Joins the feature data and the targets. pdpbox expects the pd.dataframe targets to be split into one-hot vectors if
    the problem is classification.
    
        x: 2d array-type
        
        y: 1d array type
        
        feature_names: 1d array type with feature labels
    """
    classes = ["Target"]
            
    training_df = pd.DataFrame(x, columns=feature_names)
    targets = pd.DataFrame(y, columns=classes)
    
    return training_df.join(targets)
    
def calc_rank_suffix(rank):
    rank = str(rank)
    if rank[len(rank)-2] == "1" and len(rank) > 1:
        suffix = "th"
    elif rank[len(rank)-1] == "1":
        suffix = "st"
    elif rank[len(rank)-1] == "2":
        suffix = "nd"
    elif rank[len(rank)-1] == "3":
        suffix = "rd"
    else:
        suffix = "th"
    return suffix
    
def correlation_format(corr, feature, threshold=0.5):
    highly_correlated = []
    for i, score in enumerate(corr[feature]):
        if abs(score) > threshold and score != 1.0:
            highly_correlated.append([corr.columns[i],score])
    if len(highly_correlated) != 0:
        print("WARNING: the feature you have selected ('{0}') is correlated highly with other features in the training data.".format(feature))
    for feat in highly_correlated:
        print("\tThe correlation between features '{0}' and '{1}' has been given a score of: {2}".format(feature, feat[0], feat[1]))
    print()
    return highly_correlated



def colour_scale_fi(arr):
    #higher value is green, lowest is red
    
    #being used by create_importance_graph()
    
    top = arr[len(arr)-1]
    scale = 100 / top
    
    scaled = [x*scale for x in arr]
    colours = []
    for i in scaled:
        if i == 50:
            colours.append((200,200,0))
        elif i < 50:
            num = i*4
            colours.append((200,i*4,0))
        else:
            num = 200 - (i-50)*4
            colours.append((num,200,0))
    
    main_c = []
    for i in colours:
        temp = []
        for j in i:
            temp.append(max(j/255,0))
        main_c.append(tuple(temp))
        
    
    return main_c
    
def colour_scale_corr(arr):
    #x -> 100  means red    255,0,0
    #x -> 75   means        255,178,0
    #x -> 50   means middle 255,255,0
    #x-> 0     means green  0,255,0
    color = []
    scale = 250/255
    for x in arr:
        new_x = abs(x)
        if new_x >= 0.5:
            temp = (new_x - 0.5)*2
            color.append((scale, scale*(1-temp), 0))
        else:
            temp = (0.5 - new_x)*2
            color.append((scale*(1-temp), scale, 0))
    return color
    

def marsformat_to_function(bf, co, convert):
    #first remove pruned basis functions
    co = co.tolist()
    data = []
    for b in bf:
        if not b.is_pruned():
            data.append([str(b), co.pop(0)])
    assert len(co) == 0, "if this isn't empty then somethings gone wronnnggg"
    
    #second turn feature indexes to names
    side = [""]
    for num, i in enumerate(data):
        for key in sorted(convert.keys(), key=len, reverse=True): #orders the keys such that it replaces "x10" before it replaces "x1"
            if key in i[0]:
                data[num][0] = i[0].replace(key, convert[key])
                side.append(convert[key])
    
    #third change function to be understandable and callable
    for num, i in enumerate(data):
        if "h" in i[0]:
            data[num][0] = i[0].replace("h(", "max(0,")
    
    #finally add feature_name column for easier string manipulation later
    temp = []
    for num, i in enumerate(data):
        i.append(side[num])
        temp.append(i)
    data = temp
    return data
        
        

def create_importance_graph(imps, mode, filename):
    converter = {"pred_variance": "Prediction Variance", "model_scoring": "Model Scoring", "morris": "Morris Sensitivity"}
    names = list(imps.index)
    vals = list(imps)
    vals = [x*100 for x in vals]
    fig, ax = plt.subplots()
    
    colors = colour_scale_fi(vals)
    
    print(colors)
    
    plt.barh(names,vals, align="center", color=colors)
    plt.xlabel("Percentage importance %")
    plt.ylabel("Features")
    plt.title("Feature importance ({})".format(converter[mode]))
    plt.tight_layout()
    if not os.path.isdir("output/feature_importance"):
        os.mkdir("output/feature_importance")
    if ".png" not in filename:
        filename += ".png"
    plt.savefig("./output/feature_importance/{}".format(filename))
            
def reg_to_bin(target, bin_array):
    new_class = -1
    for i, interval in enumerate(bin_array):
        if interval < target <= bin_array[i+1]:
            #print("The value {0} is between {1} and {2}.\n This means it is in class {3}".format(target, interval, bin_arr[i+1], i))
            new_class = i
    return new_class
    
def preserve(explanation, selector_key=None, file_name=None, **kwargs):
    """ Preserves an explanation's visualization for Jupyter cell, or file.

    If file_name is not None the following occurs:
    - For Plotly figures, saves to HTML using `plot`.
    - For dataframes, saves to HTML using `to_html`.
    - For strings (html), saves to HTML.
    - For Dash components, fails with exception. This is currently not supported.

    Args:
        explanation: An explanation.
        selector_key: If integer, treat as index for explanation. Otherwise, looks up value in first column, gets index.
        file_name: If assigned, will save the visualization to this filename.
        **kwargs: Kwargs which are passed to the underlying render/export call.

    Returns:
        None.
    """

    try:
        # Get explanation key
        if selector_key is None:
            key = None
        elif isinstance(selector_key, int):
            key = selector_key
        else:
            series = explanation.selector[explanation.selector.columns[0]]
            key = series[series == selector_key].index[0]

        # Get visual object
        visual = explanation.visualize(key=key)

        # Output to front-end/file
        _preserve_output(
            explanation.name,
            visual,
            selector_key=selector_key,
            file_name=file_name,
            **kwargs
        )
        return None
    except Exception as e:  # pragma: no cover
        log.error(e, exc_info=True)
        raise e
        
def explain_instance_lime(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                    `classifier.predict_proba()`. For ScikitRegressors, this
                    is `regressor.predict()`.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        data, inverse = lime.lime_tabular.LimeTabularExplainer.__data_inverse(data_row, num_samples)
        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()

        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction proabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            yss = predict_fn(inverse).reshape(len(inverse))
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        values = self.convert_and_round(data_row)

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)

        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp
        

def pdp_isolate_custom(model, dataset, model_features, feature, num_grid_points=10, grid_type='percentile',
                percentile_range=None, grid_range=None, cust_grid_points=None,
                memory_limit=0.5, n_jobs=1, predict_kwds=None, data_transformer=None):
    """Calculate PDP isolation plot

    Parameters
    ----------
    model: a fitted sklearn model
    dataset: pandas DataFrame
        data set on which the model is trained
    model_features: list or 1-d array
        list of model features
    feature: string or list
        feature or feature list to investigate,
        for one-hot encoding features, feature list is required
    num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    grid_type: string, optional, default='percentile'
        'percentile' or 'equal',
        type of grid points for numeric feature
    percentile_range: tuple or None, optional, default=None
        percentile range to investigate,
        for numeric feature when grid_type='percentile'
    grid_range: tuple or None, optional, default=None
        value range to investigate,
        for numeric feature when grid_type='equal'
    cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points for numeric feature
    memory_limit: float, (0, 1)
        fraction of memory to use
    n_jobs: integer, default=1
        number of jobs to run in parallel.
        make sure n_jobs=1 when you are using XGBoost model.
        check:
        1. https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries
        2. https://github.com/scikit-learn/scikit-learn/issues/6627
    predict_kwds: dict or None, optional, default=None
        keywords to be passed to the model's predict function
    data_transformer: function or None, optional, default=None
        function to transform the data set as some features changing values

    Returns
    -------
    pdp_isolate_out: instance of PDPIsolate

    """

    # check function inputs
    n_classes = len(model.classes_)
    if n_classes == 1:
        n_classes = 0
    if model.mode == "regression":
        predict = model.predict
    elif model.mode == "classification":
        predict = model.predict_proba
    else:
        raise ValueError("Vital error when wrapping the class")
    # avoid polluting the original dataset
    # copy training data set and get the model features
    # it's extremely important to keep the original feature order
    _check_dataset(df=dataset)
    _dataset = dataset.copy()

    feature_type = _check_feature(feature=feature, df=_dataset)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)
    _check_memory_limit(memory_limit=memory_limit)

    if predict_kwds is None:
        predict_kwds = dict()

    # feature_grids: grid points to calculate on
    # display_columns: xticklabels for grid points
    percentile_info = []
    if feature_type == 'binary':
        feature_grids = np.array([0, 1])
        display_columns = ['%s_0' % feature, '%s_1' % feature]
    elif feature_type == 'onehot':
        feature_grids = np.array(feature)
        display_columns = feature
    else:
        # calculate grid points for numeric features
        if cust_grid_points is None:
            feature_grids, percentile_info = _get_grids(
                feature_values=_dataset[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
                percentile_range=percentile_range, grid_range=grid_range)
        else:
            # make sure grid points are unique and in ascending order
            feature_grids = np.array(sorted(np.unique(cust_grid_points)))
        display_columns = [_get_string(v) for v in feature_grids]

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset, total_units=len(feature_grids), n_jobs=n_jobs, memory_limit=memory_limit)
    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(_calc_ice_lines)(
            feature_grid, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
            feature=feature, feature_type=feature_type, predict_kwds=predict_kwds, data_transformer=data_transformer)
        for feature_grid in feature_grids)

    if n_classes > 2:
        ice_lines = []
        for n_class in range(n_classes):
            ice_line_n_class = pd.concat([grid_result[n_class] for grid_result in grid_results], axis=1)
            ice_lines.append(ice_line_n_class)
    else:
        ice_lines = pd.concat(grid_results, axis=1)

    # calculate the counts
    count_data = _prepare_pdp_count_data(
        feature=feature, feature_type=feature_type, data=_dataset[_make_list(feature)], feature_grids=feature_grids)

    # prepare histogram information for numeric feature
    hist_data = None
    if feature_type == 'numeric':
        hist_data = _dataset[feature].values

    # combine the final results
    pdp_params = {'n_classes': n_classes, 'feature': feature, 'feature_type': feature_type,
                  'feature_grids': feature_grids, 'percentile_info': percentile_info,
                  'display_columns': display_columns, 'count_data': count_data, 'hist_data': hist_data}
    if n_classes > 2:
        pdp_isolate_out = []
        for n_class in range(n_classes):
            pdp = ice_lines[n_class][feature_grids].mean().values
            pdp_isolate_out.append(
                PDPIsolate(which_class=n_class, ice_lines=ice_lines[n_class], pdp=pdp, **pdp_params))
    else:
        pdp = ice_lines[feature_grids].mean().values
        pdp_isolate_out = PDPIsolate(which_class=None, ice_lines=ice_lines, pdp=pdp, **pdp_params)

    return pdp_isolate_out
    
def pdp_interact_custom(model, dataset, model_features, features, num_grid_points=None, grid_types=None,
                 percentile_ranges=None, grid_ranges=None, cust_grid_points=None, memory_limit=0.5,
                 n_jobs=1, predict_kwds=None, data_transformer=None):
    """Calculate PDP interaction plot

    Parameters
    ----------
    model: a fitted sklearn model
    dataset: pandas DataFrame
        data set on which the model is trained
    model_features: list or 1-d array
        list of model features
    features: list
        [feature1, feature2]
    num_grid_points: list, default=None
        [feature1 num_grid_points, feature2 num_grid_points]
    grid_types: list, default=None
        [feature1 grid_type, feature2 grid_type]
    percentile_ranges: list, default=None
        [feature1 percentile_range, feature2 percentile_range]
    grid_ranges: list, default=None
        [feature1 grid_range, feature2 grid_range]
    cust_grid_points: list, default=None
        [feature1 cust_grid_points, feature2 cust_grid_points]
    memory_limit: float, (0, 1)
        fraction of memory to use
    n_jobs: integer, default=1
        number of jobs to run in parallel.
        make sure n_jobs=1 when you are using XGBoost model.
        check:
        1. https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries
        2. https://github.com/scikit-learn/scikit-learn/issues/6627
    predict_kwds: dict or None, optional, default=None
        keywords to be passed to the model's predict function
    data_transformer: function or None, optional, default=None
        function to transform the data set as some features changing values

    Returns
    -------
    pdp_interact_out: instance of PDPInteract
    """
    if predict_kwds is None:
        predict_kwds = dict()

    # check function inputs
    n_classes = len(model.classes_)
    
    if n_classes == 1:
        n_classes = 0
    if model.mode == "regression":
        predict = model.predict
    elif model.mode == "classification":
        predict = model.predict_proba
    else:
        raise ValueError("Vital error when wrapping the class")
    _check_dataset(df=dataset)
    _dataset = dataset.copy()

    num_grid_points = _expand_default(x=num_grid_points, default=10)
    grid_types = _expand_default(x=grid_types, default='percentile')
    _check_grid_type(grid_type=grid_types[0])
    _check_grid_type(grid_type=grid_types[1])

    percentile_ranges = _expand_default(x=percentile_ranges, default=None)
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    grid_ranges = _expand_default(x=grid_ranges, default=None)
    cust_grid_points = _expand_default(x=cust_grid_points, default=None)

    _check_memory_limit(memory_limit=memory_limit)

    # calculate pdp_isolate for each feature
    pdp_isolate_outs = []
    for idx in range(2):
        pdp_isolate_out = pdp_isolate_custom(
            model=model, dataset=_dataset, model_features=model_features, feature=features[idx],
            num_grid_points=num_grid_points[idx], grid_type=grid_types[idx], percentile_range=percentile_ranges[idx],
            grid_range=grid_ranges[idx], cust_grid_points=cust_grid_points[idx], memory_limit=memory_limit,
            n_jobs=n_jobs, predict_kwds=predict_kwds, data_transformer=data_transformer)
        pdp_isolate_outs.append(pdp_isolate_out)

    if n_classes > 2:
        feature_grids = [pdp_isolate_outs[0][0].feature_grids, pdp_isolate_outs[1][0].feature_grids]
        feature_types = [pdp_isolate_outs[0][0].feature_type, pdp_isolate_outs[1][0].feature_type]
    else:
        feature_grids = [pdp_isolate_outs[0].feature_grids, pdp_isolate_outs[1].feature_grids]
        feature_types = [pdp_isolate_outs[0].feature_type, pdp_isolate_outs[1].feature_type]

    # make features into list
    feature_list = _make_list(features[0]) + _make_list(features[1])

    # create grid combination
    grid_combos = _get_grid_combos(feature_grids, feature_types)

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset, total_units=len(grid_combos), n_jobs=n_jobs, memory_limit=memory_limit)

    grid_results = Parallel(n_jobs=true_n_jobs)(delayed(_calc_ice_lines_inter)(
        grid_combo, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
        feature_list=feature_list, predict_kwds=predict_kwds, data_transformer=data_transformer)
                                                for grid_combo in grid_combos)

    ice_lines = pd.concat(grid_results, axis=0).reset_index(drop=True)
    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    # combine the final results
    pdp_interact_params = {'n_classes': n_classes, 'features': features, 'feature_types': feature_types,
                           'feature_grids': feature_grids}
    if n_classes > 2:
        pdp_interact_out = []
        for n_class in range(n_classes):
            _pdp = pdp[feature_list + ['class_%d_preds' % n_class]].rename(
                columns={'class_%d_preds' % n_class: 'preds'})
            pdp_interact_out.append(
                PDPInteract(which_class=n_class,
                            pdp_isolate_outs=[pdp_isolate_outs[0][n_class], pdp_isolate_outs[1][n_class]],
                            pdp=_pdp, **pdp_interact_params))
    else:
        pdp_interact_out = PDPInteract(
            which_class=None, pdp_isolate_outs=pdp_isolate_outs, pdp=pdp, **pdp_interact_params)

    return pdp_interact_out