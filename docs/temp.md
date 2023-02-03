### Step 7: Train Machine Model
This step is used for training new machine models for behavioral classifications in SimBA. There are a range of machine learning settings (called hyper-parameters) data sampling method, and evaluation settings, that influence the performance of the classifiers and how it is interpreted. We have currated a list of parameters and sampling methods that can be tweaked and validated. For more technical explanations, see the [scikit-learn RandomRandomForestClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) or join the discussion on our [Gitter page](https://gitter.im/SimBA-Resource/community). 

To facilitate use and make this step as painless as possible, users can import hyper-parameters, sampling and evaluations setting from pre-packaged CSV files (SEE BELOW). This may be a lot to look at, so please read this whole section before starting to build your own classifiers.

> **Note**: SimBA allows you to generate predictive classifiers in two different *modes*. You can either **(i)** specify a single set of hyperparameters, and train a single predictive classifiers using that defined set of hyperparameters. Alternatively, you can **(ii)** specify many different Hyperparameter settings, and batch train multiple different models, each using a different set of Hyperparameters. The second option is relevant for the current Scenario. For example, here we may want to generate five different classifiers that all predict the behavior BtWGaNP. We want to evaluate each one, and proceed to the Experimental data with the classifier that best captures behavior BtWGaNP in the pilot data and validation video. Thus, the first section of this part of the tutorial describes the different Hyperparameter settings, and what you can do to avoid setting them manually (*HINT*: you can load them all from a *metadata* file at the top of the window), while the second part of the tutorial section describes how to proceed with either of the two *modes* for generating classifiers.       

#### Train predictive classifier(s): settings

Click on `SETTINGS` in the [TRAIN MACHINE MODEL] tab and the following, slightly indimidating (but I promise: easy to use!) window will pop up. We will go over the meaning of each of the settings in turn. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/machine_model_settings_2023.png" />
</p>


**1. LOAD META DATA**. If you have a CSV file containing hyperparameter metadata, you can import this file by clicking on `Browse File` and then click on `Load`. This will autofill all the Hyperparameter entry boxes and model evaluation settings. For the Scenario 1, we [provide](https://github.com/sgoldenlab/simba/blob/master/misc/BtWGaNP_meta.csv) a Metadata file that will populate the Hyperparameter entry boxes and evaluation settings with some default values. Please save this file to disk and load it. If you downloaded SimBA through our github page, the Metadata file should be in the *simba/misc* directory. Alternatively, there is an up-to-date meta file[HERE](https://github.com/sgoldenlab/simba/blob/master/misc/meta_data_2023.csv) which covers additional newer settings recently included in SimBA (including class weights ans shapley values).


**2. MACHINE MODEL ALGORITHM**. Choose a machine model algorithm to use from the drop down menu: `RF` ,`GBC`,`Xboost`. For this scenario, we will choose RF (*Note:*: GBC and XGBoost options are still under development). 

**3. BEHAVIOR**. Use the drop-down menu to select the behavioral classifier you wish to define the hyper-parameters for. In this Scenario, only one *BEHAVIOR* should be seen in the drop-down menu (BtWGaNP). If you are generating multiple classifiers, they should all be seen in the drop-down menu. 

**4. RANDOM FOREST ESTIMATORS**. The number of decision trees we should use in our classifier. Typically a value between 100-2000. In most instances, a higher value won't hurt you. If you have a large dataset and are seeing `MemoryError` while building the classifier, then try to decrease the number of estimators. 

**5. MAX FEATURES**. Number of features to consider when looking for the best split. Select `sqrt` from the dropdown to use the square root of the total number of features in your dataset when evaluating a split. Select `log` from the dropdown to use the log of the total number of features in your dataset when evaluating a split. Select None from the dropdown to use all of the features in your dataset when evaluating a split. 

**6. CRITERION**. Select the metric used to measure the quality of each split (*gini* or *entropy*). 

**7. TEST SIZE**. Select the ratio of your data that should be used to test your model. For example, selecting `0.2` from the drop-down will results in the model beeing trained on 80% of your data, and tested on 20% of your data.  

**8. MINIMUM SAMPLE LEAF**. The minimum number of samples required to be a leaf node (e.g., *1*, or increase this value to [prevent over-fitting](https://elitedatascience.com/overfitting-in-machine-learning)).

**9. UNDER-SAMPLE SETTING**. `random undersample` or `None`. If `random undersample`, a random sample of the majority class annotations (usually behavior absent) will be used in the training set. The size of this sample will be taken as a ratio of the minority class (usually behavior present) and the ratio is specified in the `under-sample ratio` box (see below). For more information, click [HERE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html). This setting address issues that arise from "imbalanced" data sets, where the behavior that is predicted is sparse. **Note:** *Class* means the classification that a video frame belongs to. In this Scenario it is either (i) Not BtWGaNP, or (ii) BtWGaNP. The majority class is the class of frames that contains the most examples, which - most certaily - in any use case of SimBA, will be *Not BtWGaNP*. Conversely, the minority class is the class of frames that contains the least examples, which will be *BtWGaNP*. 

> Note: If your behavior is sparse (say you have annotated the behavior to be present in 1-10% of the total number of frames), see if you can improve your classifier by setting the `UNDER-SAMPLE SETTING` to `random undersample` and the `under-sample ratio` to `1`. 

**10. UNDER-SAMPLE RATIO**. The ratio of behavior-absent annotation to behavior-present annotation to use in the training set. For example, if set to `1.0`, an equal number of behavior-present and behavior-absent annotated frames will be used for training. If set to `2.0`, twice as many behavior-absent annotatations than behavior-present annotations will be used for training. If set to `0.5`, twice as many behavior-present annotatations than behavior-absent annotations will be used for training. 

**11. OVER SAMPLE SETTING**. `SMOTE`, `SMOTEEN` or `None`. If "SMOTE" or "SMOTEEN", synthetic data will be generated in the minority class based on k-mean distances in order to balance the two classes. See the imblearn API for details on [SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html) and [SMOTEEN](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html). 

**12. OVER SAMPLE RATIO**.  The desired ratio of the number of samples in the minority class over the number of samples in the majority class after over synthetic sampling. For example, if `1.0`, N number of synthetic behavior-present annotations will be created so the total count of behavior-present (synthetic and real) equals the total count of behavior-absent annotated frames. 

**13. CLASS WEIGHT SETTINGS**. Although `random undersampling` discussed in point `9` above can produce accurate classifiers, it has a major drawback. When performing undersampling we basically chuck out a bunch of data from the minority class. Instead, we may want to keep **all** the data, but value accurate observations of the minority class equally to the observations in the majority class by assigning weights to the different classes. To do this, select [`balanced`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) or `balanced_subsample`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) in the `class weight settings` dropdown. 

You can also assign your own weights to the two different classes of observations (behavior-present versus behavior-absent) by selecting the `custom` option in the `class weight settings` dropdown. For example, setting behavior PRESENT to `2`, and behavior ABSENT to `1` will lead the classifier to attribute twice the importance to behavior present annotatons over behavior absent annotations. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/class_weights.png" />
</p>


**14. CREATE MODEL META DATA FILE**. Creates a CSV file listing the hyper-parameter settings used when creating the classifier. The generated meta file can be used to create further models by importing it in the `LOAD SETTINGS` menu (see **Step 1** above).

**15. CREATE Example Decision Tree - graphviz**. Saves a visualization of a random decision tree in PDF and .DOT formats. Requires [graphviz](https://graphviz.gitlab.io/). For more information on this visualization, click [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html). For information on how to install on Windows, click [here](https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software). For an example of a graphviz decision tree visualization generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_tree.pdf). *Note:*  The trees can be very large depending on the Hyperparameter settings. Rather than using a dedicated PDF viewer, try opening the generated PDF by dragging it into a web browser to get a better view. 

**16. Fancy Example Decision Tree - dtreeviz**. Saves a nice looking visualization of a random decision tree in SVG format. Requires [dtreeviz](https://github.com/parrt/dtreeviz). For an example of a dtreeviz decision tree visualization generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/dtreeviz_SimBA.png). *Note:* These SVG example decision trees are very large. To be able to view them on standard computers, SimBA limits the depth of the example tree to 3 levels.

**17. Create Classification Report**. Saves a classification report truth table in PNG format displaying precision, recall, f1, and support values. For more information, click [here](http://www.scikit-yb.org/zh/latest/api/classifier/classification_report.html). For an example of a classification report generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_classificationReport.png).  

**18. Create Features Importance Bar Graph**. Creates a CSV file that lists the importance's [(gini importances)](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html). For an example of a feature importance list generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_feature_importance_log.csv). Also creates a bar chart depecting the top N features features in the classifier. For an example of a bar chart depicting the top *N* features generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_feature_bars.png) *Note:* Although [gini importances](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) gives an indication of the most important features for predicting the behavior, there are known [flaws when assessing well correlated features](https://explained.ai/rf-importance/). As SimBA uses a very large set of well correlated features to enable flexible usage, consider evaluating features through permutation importance calculatations instead or shapley values (see below). 

**19. # Features**. The number of top N features to plot in the `Create Features Importance Bar Graph`.

**20. Compute Feature Permutation Importances**: Creates a CSV file listing the importance's (permutation importance's) of all features for the classifier. For more details, please click [here](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)). Note that calculating permutation importance's is computationally expensive and takes a long time. For an example CSV file that list featue permutation importances, click [here] (https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_prediction_permutations_importances.csv). 

**21. Create Learning Curves**: Creates a CSV file listing the f1 score at different test data sizes. For more details on learning curves, please click [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)). For more information on the f1 performance score, click [here](https://en.wikipedia.org/wiki/F1_score). The learning curve is useful for estimating the benefit of annotating further data. For an example CSV file of the learning curve generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_prediction_learning_curve.csv).  
**22. Learning Curve Shuffle K Splits**: Number of cross validations applied at each test data size in the learning curve. 

**23. Learning Curve shuffle Data Splits**: Number of test data sizes in the learning curve.

**24. Create Precision Recall Curves**: Creates a CSV file listing precision at different recall values. This is useful for titration of the false positive vs. false negative classifications of the models by manipulating the `Discrimination threshold` (see below). For an example CSV file of of a precision-recall curve generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/misc/BtWGaNP_prediction_precision_recall.csv). This may be informative if your aiming to titrate your classifier to (i) predict all occurances of behavior BtWGaNP, and can live with a few false-positive classifications, or (ii) conversely, you want all predictions of behavior BtWGaNP to be accurate, and can live with a few false-negative classifications, or (iii) you are aiming for a balance between false-negative and false-positive classifications. 

**25. Calculate SHAP scores**: Creates a CSV file listing the contribution of each individual feature to the classification probability of each frame. For more information, see the [SimBA SHAP tutorial](https://github.com/sgoldenlab/simba/edit/master/docs/SHAP.md) and the [SHAP GitHub repository](https://github.com/slundberg/shap). SHAP calculations are an computationally expensive process, so we most likely need to take a smaller random subset of our video frames, and calculate SHAP scores for this random subset. 

**26 # target present**: The number of frames (integer - e.g., `100`) with the behavioral target **present** (with the behavioral target being the behavior selected in the **Behavior** drop-down menu in Step 3 above) to calculate SHAP values for. 

**27 # target absent**: The number of frames (integer - e.g., `100`) with the behavioral target **absent** (with the behavioral target being the behavior selected in the **Behavior** drop-down menu in Step 3 above) to calculate SHAP values for.
