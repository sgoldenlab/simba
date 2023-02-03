### Step 7: Train Machine Model
This step is used for training new machine models for behavioral classifications. There are a range of machine learning parameters (called Hyperparameters) and sampling data sampling method, that influence Random Forest models. We have currated a list of Hyperparameters and made it easy to tweak, and validate, their values. For more in-depth technical explanations, please see [sklearn.ensemble.RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) or join the discussion on our [Gitter page](https://gitter.im/SimBA-Resource/community). We have also made it possible to import these settings, in order to make the setting of these Hyperparameters as painless as possible (see below). This is a lot to look at, please read this whole section before starting anything.

**Note**: SimBA allows you to generate predictive classifiers in two different *modes*. You can either (i) specify a single set of Hyperparameters, and train a single predictive classifiers using that defined set of hyperparameters, or alternatively, you can (ii) specify many different Hyperparameter settings, and batch train multiple different models, each using a different set of Hyperparameters. The second option is relevant for the current Scenario. For example, here we may want to generate five different classifiers that predict the behavior BtWGaNP, evaluate each one, and proceed to the Experimental data with the classifier that best captures behavior BtWGaNP in the pilot data and validation video. Thus, the first section of this part of the tutorial describes the different Hyperparameter settings, and what you can do to avoid setting them manually (*HINT*: you can load them all from a *metadata* file at the top of the window), while the second part of the tutorial section describes how to proceed with either of the two *modes* for generating classifiers.       

#### Train predictive classifier(s): settings

1. Click on `Settings` and the following, slightly indimidating (but I promise: easy to use!) window will pop up. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Settings1.png" />
</p>

>*Note I:* If you have a CSV file containing hyperparameter metadata, you can import this file by clicking on `Browse File` and then click on `Load`. This will autofill all the Hyperparameter entry boxes and model evaluation settings. For the Scenario 1, we [provide](https://github.com/sgoldenlab/simba/blob/master/misc/BtWGaNP_meta.csv) a Metadata file that will populate the Hyperparameter entry boxes and evaluation settings with some default values. Please save this file to disk and load it. If you downloaded SimBA through our github page, the Metadata file should be in the *simba/misc* directory. 

>*Note II:* If you open the *BtWGaNP_meta.csv* file, you'll see that the left-most column - with the heading *Classifier_name* - contains one entry: *BtWGaNP*. This *Classifier_name* entry is only for reference, and does not affect the classfier you are generating. If you are generating a classifier using a different classifier name, let's say *attack*, you can leave the entries in the *BtWGaNP_meta.csv* file as is without causing any errors. 

2. Under **Machine Model**, choose a machine model from the drop down menu: `RF` ,`GBC`,`Xboost`. For this Scenario, choose RF (*Note:*: GBC and Xgboost options are still under development). 

- `RF`: Random forest

- `GBC`: Gradient boost classifier

- `Xgboost`: eXtreme Gradient boost

3. Under the **Model** heading, use the dropdown menu to select the behavioral classifier you wish to define the hyper-parameters for. In this Scenario, only one *Model Name* should be seen in the drop-down menu (BtWGaNP). If you are generating multiple classifiers, they should all be seen in the drop-down menu. 

4. Under the **Hyperparameters** heading, select the Hyperparameters settings for your model. For more details, please click [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Alternatively, import settings from a meta data file (see above). 

Here is a brief description of the different Hyperparameter settings, together with some further links to where you can read more about them. 

- `RF N estimators`: Number of decision trees in the decision ensemble (e.g., 2000).

- `RF Max features`: Number of features to consider when looking for the best split (e.g., enter *sqrt* to take the square root of the total number of features in your dataset when evaluating a split).

- `RF Criterion`: The metric used to measure the quality of each split (e.g., *gini* or *entropy*).

- `Train Test Size`: The ratio of the dataset withheld for testing the model (e.g., 0.20).

- `RF Min sample leaf`: The minimum number of samples required to be at a leaf node (e.g., *1*, or increase to [prevent over-fitting](https://elitedatascience.com/overfitting-in-machine-learning)). 

- `Under sample setting`: "Random undersample" or "None". If "Random undersample", a random sample of the majority class will be used in the train set. The size of this sample will be taken as a ratio of the minority class and should be specified in the "under sample ratio" box below. For more information, click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html). This setting address issues that arise from "imbalanced" data sets, where the behavior that is predicted is very sparse. **Note:** *Class* means the classification that a video frame belongs to. In this Scenario it is either (i) Not BtWGaNP, or (ii) BtWGaNP. The majority class is the class that contains the most examples, which - most certaily - in any use case of SimBA,will be *Not BtWGaNP*. Conversely, the minority class is the class that contains the least examples, which will be *BtWGaNP*.    

- `Under sample ratio`: The ratio of samples of the majority class to the minority class in the training data set. Applied only if `Under sample ratio` is set to "Random undersample". Ignored if "Under sample setting" is set to "None" or NaN.

- `Over sample setting`: "SMOTE", "SMOTEEN" or "None". If "SMOTE" or "SMOTEEN", synthetic data will be generated in the minority class based on k-mean distances to balance the two classes. For more details on SMOTE, click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html). Alternatively, import recommended settings from a meta data file (see **Step 1**). For more information on *minority/majority classes* see the documentation for `Under sample setting` above. 

- `Over sample ratio`: The desired ratio of the number of samples in the minority class over the number of samples in the majority class after over sampling.Applied only if `Over sample setting` is set to "SMOTE" or "SMOTEEN". Ignored if "Under sample setting" is set to "None" or NaN.

5. Under **Model Evaluation Settings**.

Here is a brief description of the different Model evaluation settings available in SimBA, together with links to where you can read more about them. These model evaluation tools generate graphs, images, and CSV files that contain different metrics on how well your classifiers performs, how the classifiers reaches its decisions, and how you could improve the predictive classifier.   

- `Generate RF model meta data file`: Generates a CSV file listing the hyper-parameter settings used when creating the model. The generated meta file can be used to create further models by importing it in the `Load Settings` menu (see above, **Step 1**).

- `Generate Example Decision Tree - graphviz`: Saves a visualization of a random decision tree in PDF and .DOT formats. Requires [graphviz](https://graphviz.gitlab.io/). For more information on this visualization, click [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html). For information on how to install on Windows, click [here](https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software). For an example of a graphviz decision tree visualization generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_tree.pdf). *Note:*  The trees can be very large depending on the Hyperparameter settings. Rather than using a dedicated PDF viewer, try opening the generated PDF by dragging it into a web browser to get a better view. 

- `Generate Fancy Example Decision Tree - dtreeviz`: Saves a nice looking visualization of a random decision tree in SVG format. Requires [dtreeviz](https://github.com/parrt/dtreeviz). For an example of a dtreeviz decision tree visualization generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/dtreeviz_SimBA.png). *Note:* These SVG example decision trees are very large. To be able to view them on standard computers, SimBA limits the depth of the example tree to 3 levels.

- `Generate Classification Report`: Saves a classification report truth table in PNG format displaying precision, recall, f1, and support values. For more information, click [here](http://www.scikit-yb.org/zh/latest/api/classifier/classification_report.html). For an example of a classification report generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_classificationReport.png).  

- `Generate Features Importance Log`: Creates a CSV file that lists the importance's [(gini importances)](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) of all features used to generate the classifier. The CSV file is saved in the `project_folder\logs` directory, with the file name 'BtWGaNP_feature_importance_log.csv'. For an example of a feature importance list generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_feature_importance_log.csv). *Note:* Although [gini importances](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) gives an indication of the most important features for predicting the behavior, there are known [flaws when assessing well correlated features](https://explained.ai/rf-importance/). As SimBA uses a very large set of well correlated features to enable flexible usage, consider evaluating features through permutation importance calculatations instead (see below). 

- `Generate Features Importance Bar Graph`: Creates a bar chart of the top *N* features based on gini importances (see above). Specify *N* in the `N feature importance bars` entry box below (see blow). **The creation of this bar graph requires that the `Generate Features Importance Log` box is ticked**. For an example of a bar chart depicting the top *N* features generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_feature_bars.png).   

- `N feature importance bars`: Integer defining the number of top features to be included in the bar graph (e.g., 15). 

- `Compute Feature Permutation Importance's`: Creates a CSV file listing the importance's (permutation importance's) of all features for the classifier. For more details, please click [here](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)). Note that calculating permutation importance's is computationally expensive and takes a long time. For an example CSV file that list featue permutation importances, click [here] (https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_prediction_permutations_importances.csv). 

- `Generate Sklearn Learning Curve`: Creates a CSV file listing the f1 score at different test data sizes. For more details on learning curves, please click [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)). For more information on the f1 performance score, click [here](https://en.wikipedia.org/wiki/F1_score). The learning curve is useful for estimating the benefit of annotating further data. For an example CSV file of the learning curve generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_prediction_learning_curve.csv). 

- `LearningCurve shuffle K splits`: Number of cross validations applied at each test data size in the learning curve. 

- `LearningCurve shuffle Data splits`: Number of test data sizes in the learning curve.  

- `Generate Precision Recall Curves`: Creates a CSV file listing precision at different recall values. This is useful for titration of the false positive vs. false negative classifications of the models by manipulating the `Discrimination threshold` (see below). For an example CSV file of of a precision-recall curve generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/misc/BtWGaNP_prediction_precision_recall.csv). This may be informative if your aiming to titrate your classifier to (i) predict all occurances of behavior BtWGaNP, and can live with a few false-positive classifications, or (ii) conversely, you want all predictions of behavior BtWGaNP to be accurate, and can live with a few false-negative classifications, or (iii) you are aiming for a balance between false-negative and false-positive classifications. 

- `Calculate SHAP scores`: Creates a CSV file listing the contribution of each individual feature to the classification probability of each frame. For more information, see the [SimBA SHAP tutorial](https://github.com/sgoldenlab/simba/edit/master/docs/SHAP.md) and the [SHAP GitHub repository](https://github.com/slundberg/shap). SHAP calculations are an computationally expensive process, so we most likely need to take a smeller random subset of our video frames, and calculate SHAP scores for this random subset:

  - `# target present`: The number of frames (integer - e.g., `100`) with the behavioral target **present** (with the behavioral target being the behavior selected in the **Model** drop-down menu in Step 3 above) to calculate SHAP values for. 
 
   - `# target absent`: The number of frames (integer - e.g., `100`) with the behavioral target **absent** (with the behavioral target being the behavior selected in the **Model** drop-down menu in Step 3 above) to calculate SHAP values for.
