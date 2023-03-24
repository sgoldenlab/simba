# <p align="center"> Unsupervised ML in SimBA </p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised.png" />
</p>




## TUTORIAL 

After opening your SimBA project, head to the `[Add-ons]` tab and click on the `Unsupervised` button to bring up the SimBA Unsupervised graphical interface. The code for unsupervised learning in SimBA lives [HERE](https://github.com/sgoldenlab/simba/tree/master/simba/unsupervised).

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_0.png" />
</p>

### STEP 1

The first tab in this pop-up is named `Create Dataset`. We will use this tab to create a single-file dataset with all our supervised classifications. 
This file will then be used throughout the rest of the pipeline to build and analyze dimensionality reduction and clustering models. There are some user-defined options for how this file should be created:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_1.png" />
</p>

In the `FEATURE SLICE` dropdown, select which fields you want SimBA to use when creating the dataset, the options are `ALL FEATURES (EXCLUDING POSE)`, `ALL FEATURES (INCLUDING POSE)`, `USER-DEFINED FEATURE SET`. If selecting `USER-DEFINED FEATURE SET`, you need to select the path to a CSV file listing the features you want to use: more information below. 

In the `CLASSIFIER SLICE` dropdown, select classified events you want to include in your dataset. The drop-down will include as many otions as there are classifiers (plus one: ALL CLASSIFIERS). 

The `Feature file` file selection box will become available if `FEATURE SLICE` is set to `USER-DEFINED FEATURE SET`. Here, select the path to a CSV file listing the features you want to include in your unsupervised dataset. For an example of such a file, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/unsupervised_example_x.csv). 

In the `MINIMUM BOUT LENGTH` entry-box, enter the **shortest** behavior bout that should be included in your dataset. Any bout of behaviour shorter than the length choosen in `MINIMUM BOUT LENGTH` entry-box will be omitted from the dataset. 

Finally, click `CREATE DATASET`. A new file (in `.pickle` format) will be saved inside the project_folder/logs directory of your SimBA project. 

### STEP 2

Next, we want to use the dataset create (fit) in Step 1 to create a dimensionality reduction model. Click the [DIMENSIONALITY REDUCTION] tab and the `DIMENSIONALITY REDUCTION MODELS: FIT` button to bring up the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_2.png" />
</p>

In the `DATASET (PICKLE):` selection box, select the path to the file created in Step 1. 

In the `SAVE DIRECTORY` selection box, select the path to a folder where the models should be saved. *NOTE THAT THIS FOLDER **HAS TO BE** AN EMPTY DIRECTORY*

In the `ALGORITHM` dropdown, select which algorithm to use to perform dimensionality reduction. In this tutorial, we will select `UMAP`. 

In the `VARIANCE THRESHOLD` drop-down, select the minimum required variance of a feature for it to be included in the model. If `NONE`, then all features regardless of variance will be considered in the model.  

In the `SCALING` dropdown, select the method to normalize your feature values. 

In the `GRID SEARCH HYPERPARAMETERS` sub-menu there are three list-boxes (`N NEIGHBOURS`, `MIN DISTANCE` and `SPREAD`). Similar to [model settings when creating supervised ML models](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-7-train-machine-model), unsupervised models also require a set of hyper-parameters, and we don't necessarily know which set of hyper-parameters creates the most informative model. We may therefore have to **grid-search** the different hyper-parameters, i.e., we have to create a bunch of different models all at once. The values you see inserted into these list-boxes when opening the `DIMENSIONALITY REDUCTION MODELS: FIT` window are the [default values](https://umap-learn.readthedocs.io/en/latest/parameters.html) of the algorithm python package. To include further parameters, or to delete existing parameters, use the `VALUE` box and the `ADD` and `REMOVE` buttons:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_4.gif" />
</p>

SimBA will create as many models as there are products of the values in the three listboxes `N NEIGHBOURS`, `MIN DISTANCE` and `SPREAD`. When filled in, click the RUN button. You can follow the progress in the terminal and the main window of SimBA. Once complete, one pickled file for each model will be saved in the folder selected in the SAVE DIRECTORY selection box. 

### STEP 3

Next, we want to perform clusring on the models created in Step 2. Click the [CLUSTERING] tab and the `CLUSTERING MODELS: FIT` button to bring up the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_3.png" />
</p>

In the `EMBEDDING DIRECTORY:` selection box, select the to the directory where you saved your dimensionality reduction models in Step 2. 

In the `SAVE DIRECTORY` selection box, select the path to a folder where the clusring models should be saved. *NOTE THAT THIS FOLDER **HAS TO BE** AN EMPTY DIRECTORY*

In the `ALGORITHM` dropdown, select which algorithm to use to perform clustring. In this tutorial, we will select `HDBSCAN`. 

In the `GRID SEARCH HYPERPARAMETERS` sub-menu there are three list-boxes (`ALPHA`, `MIN CLUSTER SIZE`, `EPSILON` and `MIN SAMPLES`). Just as for dimensionality reduction models (and [supervised ML models](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-7-train-machine-model)), clustering models accepts a set of hyper-parameters. The values you see inserted into these list-boxes when opening the `DIMENSIONALITY REDUCTION MODELS: FIT` window are the [default values]([https://umap-learn.readthedocs.io/en/latest/parameters.html](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)) of the algorithm python package. SimBA will create as many models as there are products of the values in the four listboxes `ALPHA`, `MIN CLUSTER SIZE`, `EPSILON` and `MIN SAMPLES`. When filled in, click the RUN button. You can follow the progress in the terminal and the main window of SimBA. Once complete, one pickled file for each model will be saved in the folder selected in the SAVE DIRECTORY selection box. 

### STEP 4 

Next, we may want to visualize the results of our dimensionality reduction and clustring algorithms. For this, click on the [VISUALIZATION] tab and the `GRID SERACH VISUALIZATION` button to bring up the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_5.png" />
</p>

In the `CLUSTERERS DIRECTORY:` selection box, select the directory where you saved your clustering models in Step 3.

In the `IMAGE SAVE DIRECTORY:` selection box, select the directory where want your images to ba saved. *NOTE THAT THIS FOLDER **HAS TO BE** AN EMPTY DIRECTORY*

#TODO....

### STEP 4 

Next, we may want to do some statistics to compute (i) how well our models clusters the data, and (ii) what our clusters represents. For this, click on the the `[METRICS]` tab. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_5.png" />
</p>

#### DENSITY BASED CLUSTER VALIDATION (DBCV)

[Density-based cluster validation (DBCV)](https://github.com/christopherjenness/DBCV) is a method to compare the distances between observations with and between clusters. It returns a value between -1 and 1, ranging from *really* bad clusters and *really* good clusters. To perform DBCV, click on `DENSITY BASED CLUSTER VALIDATION` and select a path to a folder with cluster models  and click `RUN`. 

The results are saved in a xlxs/xls file with a datetime stamp within the `project_folder/logs` directory of your SImBA project, the file may be named something like `DBCV_20230321161230.xlsx`.

#### CLUSTER FREQUENTIST STATISTICS 

We may want to compute some descriptive and frequentist statistics of the feature values in the observations in each cluster. For this, click on the `CLUSTER FREQUENTIST STATISTICS` button.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_7.png" />
</p>

When ticking the `CLUSTER DESCRIPTIVE STATISTICS` check-box, SimBA will calculate the mean, standard deviation, and standard error of all feature values grouped by CLUSTER. 

When ticking the `CLUSTER FEATURE ONE-WAY ANOVA` check-box, SimBA will perform one one-way ANOVA per feature in the dataset where the cluster assignment is the independent variable and the feature is the dependent variable. 

When ticking the `CLUSTER FEATURE POST-HOC (TUKEY)` check-box, SimBA will perform post-hoc comparisons between each cluster assignment within every feature. 

When ticking the `USE SCALED FEATURE VALUES` check-box, SimBA will calculate the statistics and descriptive statistics using the scaled feature values going into the dimensionality reduction as choosen in Step 1. If un-ticked, SimBA will use the raw feature values. 

Click `RUN` to perform the analysis. The results will be saved in a multi-sheet xlxs file within the `project_folder/logs` directory named something like `cluster_descriptive_statistics_20230321110700.xlsx`. For an example of expected output, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/cluster_descriptive_statistics_20230321110700.xlsx). 

#### CLUSTER XAI STATISTICS

We may want to use ML algorithms to better understand the differences and similarities between observations in each cluster. For this, click on the `CLUSTER XAI STATISTICS` button.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_8.png" />
</p>

When ticking the `CLUSTER RF GINI IMPORTANCE STATISTICS` check-box, SimBA will fit a random forest model to your data where the cluster assignment is the target. SimBA will then save the GINI feature importance scores for the model, where features with HIGH GINI feature importance scores are more important to discriminate observations in one cluster from observations in the other clusters. 

When ticking the `CLUSTER RF PERMUTATION IMPORTANCE` check-box, SimBA will fit a random forest model to your data where the cluster assignment is the target. SimBA will calculate the [permutation importance](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings) scores for each feature in the model, just as during superved learning in SimBA. High permutation importance scores are indicative that the feature is more important to discriminate observations in one cluster from observations in the other clusters. 

When ticking the `CLUSTER RF SHAPLEY VALUES` check-box, SimBA will fit a random forest model to your data where the cluster assignment is the target. SimBA will then calculate Shapley values for each feature. 


#### EMBEDDING CORRELATIONS

We may want to compute how the dimensionally reduced variables (X and Y) correlate with each of the feature values. To do this, click on the `EMBEDDING CORRELATIONS` button.  

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_9.png" />
</p>

First, choose which type(s) of correlations you want to compute (SPEARMAN, PEARSON, KENDALL).

If you want to create plots of the correlations, check the `PLOTS` checkbox. If checked, the `PLOTS CORRELATION` and `PLOTS PALETTE` become available. Here, choose which type of correlation you want to display in the plots, and which palette you want to use. 

Click `RUN`. The results are saved within the `project_folder/logs` directory of your SimBA project. 

















