# <p align="center"> Unsupervised ML in SimBA </p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised.png" />
</p>




## TUTORIAL 

After opening your SimBA project, head to the `[Add-ons]` tab and click on the `Unsupervised` button to bring up the SimBA Unsupervised graphical interface.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_0.png" />
</p>

## STEP 1

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

## STEP 2

Next, we want to use the dataset create (fit) in Step 1 to create a dimensionality reduction model. Click the [DIMENSIONALITY REDUCTION] tab and the `DIMENSIONALITY REDUCTION MODELS: FIT` button to bring up the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_2.png" />
</p>

In the `DATASET (PICKLE):` selection box, select the path to the file created in Step 1. 

In the `SAVE DIRECTORY` selection box, select the path to a folder where the models should be saved. *NOTE THAT THIS FOLDER **HAS TO BE** AN EMTY DIRECTORY*

In the `ALGORITHM` dropdown, select which algorithm to use to perform dimensionality reduction. In this tutorial, we will select `UMAP`. 

In the `VARIANCE THRESHOLD` drop-down, select the minimum required variance of a feature for it to be included in the model. If `NONE`, then all features regardless of variance will be considered in the model.  

In the `SCALING` dropdown, select the method to normalize your feature values. 

In the `GRID SEARCH HYPERPARAMETERS` sub-menu there are three list-boxes (`N NEIGHBOURS`, `MIN DISTANCE` and `SPREAD`). Similar to [model settings when creating supervised ML models](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-7-train-machine-model), unsupervised models also require a set of hyper-parameters, and we don't necesserily know which set of hyper-parameters creates the most informative model. We may therefore have to **grid-search** the different hyper-parameters, i.e., we have to create a bunch of different models all at once. The values you see inserted into these list-boxes when opening the `DIMENSIONALITY REDUCTION MODELS: FIT` window are the [default values](https://umap-learn.readthedocs.io/en/latest/parameters.html) of the algorithm python package. To include further parameters, or to delete existing parameters, use the `VALUE` box and the `ADD` and `REMOVE` buttons:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_4.gif" />
</p>

SimBA will create as many models as there are products of the values in the three listboxes `N NEIGHBOURS`, `MIN DISTANCE` and `SPREAD`. When filled in, click the RUN button. You can follow the progress in the terminal and the main window of SimBA. Once, complete, one pickled file for each model will be saved in the folder selected in the SAVE DIRECTORY selection box. 

## STEP 3

Next, we want to perform clusring on the models created in Step 2. Click the [CLUSTERING] tab and the `CLUSTERING MODELS: FIT` button to bring up the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/unsupervised_3.png" />
</p>

In the `EMBEDDING DIRECTORY:` selection box, select the to the directory where you saved your dimensionality reduction models in Step 2. 

In the `SAVE DIRECTORY` selection box, select the path to a folder where the clusring models should be saved. *NOTE THAT THIS FOLDER **HAS TO BE** AN EMTY DIRECTORY*

In the `ALGORITHM` dropdown, select which algorithm to use to perform clustring. In this tutorial, we will select `HDBSCAN`. 

In the `GRID SEARCH HYPERPARAMETERS` sub-menu there are three list-boxes (`ALPHA`, `MIN CLUSTER SIZE`, `EPSILON` and `MIN SAMPLES`). Just as for dimensionality reduction models (and [supervised ML models](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-7-train-machine-model)), clustering models accepts a set of hyper-parameters. The values you see inserted into these list-boxes when opening the `DIMENSIONALITY REDUCTION MODELS: FIT` window are the [default values]([https://umap-learn.readthedocs.io/en/latest/parameters.html](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)) of the algorithm python package. SimBA will create as many models as there are products of the values in the four listboxes `ALPHA`, `MIN CLUSTER SIZE`, `EPSILON` and `MIN SAMPLES`. When filled in, click the RUN button. You can follow the progress in the terminal and the main window of SimBA. Once, complete, one pickled file for each model will be saved in the folder selected in the SAVE DIRECTORY selection box. 
