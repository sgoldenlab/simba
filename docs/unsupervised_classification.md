# Unsupervised Machine Learning in SimBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/dim_1.png" />
</p>

## BACKGROUND

SimBA primarily focus on *supervised* machine learning methods. In supervised machine learning (such as [SimBA-generated classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model), or [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)/[SLEAP](https://github.com/murthylab/sleap)-generated convolutional neural networks), we train computer models to recognize behavioral events or body-parts **from human-supplied labels or annotations**. Thus, in supervised machine learning, the goal is to create a model that recognizes the same stuff as the human observer.

Such supervised approaches come with a major drawback. As we are training the computer to recognize behavioral events from **from human-supplied labels or annotations**, the models will at best recognize the same behavioural events as the human annotator. In *unsupervised* machine learning, however, we are agnostic to the behaviors present in the video recording and we are instead creating the algorithm which best clusters (discriminates) the different behaviors present in the videos for us. This means that  *unsupervised* machine learning, but not *supervised machine learning*, can detect novel behaviors that are not immediately observable by the human annotator.

There are several popular and proven python packages focused on unsupervised machine learning from pose-estimation and we list these excellent packages at the [bottom of this page](https://github.com/sgoldenlab/simba/blob/master/docs/unsupervised_classification.md#other-unsupervised-machine-learning-packages-for-pose-estimation-data). The SimBA unsupervised machine learning tool works differently from these tools, and we explain the process step-by-step below. If the SimBA unsupervised machine learning tools do not work for your specific use case, we suggest that you either reach out to us on [Gitter](https://gitter.im/SimBA-Resource/community), or try the [alternative unsupervised packages](https://github.com/sgoldenlab/simba/blob/master/docs/unsupervised_classification.md#other-unsupervised-machine-learning-packages-for-pose-estimation-data).


## TUTORIAL OVERVIEW

The SimBA Unsupervised Classification module allows users to classify their animal behavior data without pre-defining behavioral standards and insteads clusters the data based on behavioral similarities. Our pipeline is designed to build off our supervised machine learning pipeline, where the clusters can help uncover new behavior present in different forms by dividing up behavior that has already been successfully classified and clustering the new, unlabeled behaviors. To find these new behaviors, we cannot rely on performance targets, but instead use dimensionality reduction and fine-tuning the hyperparameters through a [pseudo-grid search approach](https://github.com/sgoldenlab/simba/blob/master/docs/unsupervised_classification.md#step-3-perform-dimensionality-reduction) to find the best clusters. 

In order to perform the unsupervised classification analysis, we must start with our folder of machine results as was generated in the supervised pipeline and located within your project folder as described in [this scenario](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model).  Overall, unsupervised classification provides the potential for increased accuracy and explainability while reducing human variability errors by eliminating the need for frame by frame annotation. The SimBA Unsupervised Classification pipeline and GUI were created by [Simon Nilsson](https://github.com/sronilsson) and [Aasiya Islam](https://github.com/aasiya-islam).

## Pipeline

Training an unsupervised classifier involves the creation of an algorithm that clusters animal activity based on behavioral similarities, typically with three main steps: data pre-processing, dimensionality reduction, and cluster assignment.
The SimBA Unsupervised Classification pipeline consists of six main steps: 

**1) Save Project Folder**- allows user to create an unsupervised project folder and saves outputs to future steps in encompassed folder        
**2) Create Dataset**- cleans and pre-processes the machine results and localizes relevant data into a single dataset      
**3) Perform Dimensionality Reduction**- allows user to select a dimensionality reduction algorithm to apply and visualize data with     
**4) Perform Clustering**- assigns and visualizes clusters with HDBSCAN from the dimensionality reduction results       
**5) Train Model**- trains supervised classifier on the clusters generated previously and saves permutational importance and feature correlation metrics      
**6) Visualize Clusters**- visualize behavioral bouts corresponding to clusters as original video clips or simulated skeleton movements

In addition to the folder of supervised machine results, to complete this analysis we must also have on hand the following files and folders:

- a features to remove file to disregard in classification [(step 2)](https://github.com/sgoldenlab/simba/blob/master/docs/unsupervised_classification.md#step-2-create-dataset)
- the folder of animal behavior videos from which the analysis was performed on [(step 6)](https://github.com/sgoldenlab/simba/blob/master/docs/unsupervised_classification.md#step-6-visualize-clusters)
- the animal headers file to list body parts [(step 6)](https://github.com/sgoldenlab/simba/blob/master/docs/unsupervised_classification.md#step-6-visualize-clusters)

If you do not have these files, please complete the steps in the prior tutorial as described before moving on to the unsupervised learning module.

Please note that Steps 5 and 6, the model training and cluster visualization, are optional and more for explainability of the clusters. Training the model builds supervised learning methods with the use of random forest classifiers on top of the unsupervised classification with the generation of behavioral clusters, so this step is no longer part of the actual unsupervised pipeline. 

The outputs generated from each step can be saved into their respective folders encompassed within the main unsupervised project folder and taken as inputs for subsequent steps throughout the pipeline.

## Step 1: Save Project Folder 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/save_project2.PNG" />
</p>

The first step allows the user to create and save a project folder for their unsupervised analysis results. 
It creates a file folder structure that creates a new folder within the project folder for each step. For each subsequent step in the pipeline, the individual outputs will save into that step's respective folder that is automatically created upon the use of each step. 

To save a project folder, specify the folder path where you would like the project folder to be saved into by clicking the ```Browse Folder``` button and selecting a folder, as the folder path will replace the ```No folder selected``` box, and designate a name for the project folder with the ```Project Name``` entry box. 
Note that the folder name will save as "unsupervised_projectname" with "projectname" being the name you filled out the entry box with. After you have specified the folder path and designated a project name, click the ```Create folder``` button to save the folder.

[insert GIF tutorial]

## Step 2: Create Dataset

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/dataset2.PNG" />
</p>

The second tab will walk you through pre-processing and cleaning the data from prior machine results, and saving the data relevant for unsupervised analysis into a single dataset. 
This step serves to extract behavioral bouts of interest as designated by the classifier and find mean feature values while dropping all features that are not relevant.

To begin, first import a folder of the machine results saved as .csv datasets by clicking the ```Browse Folder``` button and selecting the folder of datasets being the machine results. An example of the machine results folder and .csv format and can be shown below, and is generated as directed by this documentation (hyperlink relevant documentation, discuss kleinberg or link documentation]



<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/dataset_folder2.PNG" />
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/machine_results.PNG" />
</p>


Next, import the file that lists the features that you would like to remove or disregard in the classification. 
The purpose of this is to perform the classification without the irrelevant features (double check this) that we don't want to use in the clusters, and drop them before saving our condensed dataset for future analysis. (body part coordinates are unlikely to be relevant to the behavior clustering, therefore want to remove them (x, y, p)
You can similarly select the ```Browse File``` button to search for and select this file, and the file should be formatted similar to as shown below. (update image to include body part coordinates to make it clear, and there might be others. (remove all features in animals, remove all features custom, based on supervised 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/features2remove.PNG" />
</p>


Finally, input the name of the classifier you would like to focus your analysis on, such as "Attack" in the ```Classifier name``` entry box. Note the formatting of this name should match that depicted in the machine results or prior analysis.

Once everything has been imported, you can begin the pre-processing by clicking the ```Generate/save dataset``` button. 

>**Note:** This step may take a few minutes to process all of the machine results depending on the length and number of datasets in the folder. Once everything has been processed you will see the .pkl file 

Once everything has been processed, you will observe that a new folder labeled 'create_dataset' has been saved in your project folder, and inside the 'create_dataset' folder, there will be a single .pkl file saved under the name of the classifier you inputted, such as "Attack.pkl". The .pkl file is a serialized object file that can be read in and deserialized in future steps to use in our classification, and is mainly used for storage efficiency. It cannot be opened on its own like you would a .csv file. 

[insert pic of pkl file saved w final tutorial]


[GIF of saving dataset then showing pkl file saved]


## Step 3: Perform Dimensionality Reduction

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/DR_2.PNG" />
</p>

The third tab will walk you through selecting a dimensionality reduction algorithm and inputting hyperparameters for the respective algorithm. Dimensionality reduction is used in unsupervised learning to transform the data from high-dimension to low-dimension, simplifying a data point to a single x,y coordinate. It is especially useful as it reduces the number of variables/features while still maintains data integrity and retains meaningful properties of the intrinsic dimension. Here, it is valuable for our visualization of the data as pre-processed and generated in the previous step and gives us our first glimpse of data relationships prior to clustering in the next step. 

We have provided 3 options for dimensionality reduction algorithms to choose from and use in your analysis, being [UMAP](https://umap-learn.readthedocs.io/en/latest/), [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Each has their own set of hyperparameters to perform the dimensionality reduction with, as outlined below, but can be inputted in the same manner. 

>**Note:** To read more about the supporting documentation for each of the hyperparameters, please click on the algorithm name and follow the hyperlink. 

First, import the pre-processed feature dataset as was saved as a .pkl file from the previous step by clicking the ```Browse File``` button and navigating to the 'create_dataset' folder. 


Then, you can select between your dimensionality reduction algorithms via the dropdown menu on the right of the `Select dimensionality reduction algorithm` label, and by clicking the small box inside of the box that says `UMAP` (which is the default algorithm), the other options should be available. As each algorithm is selected, a new set of hyperparameter entry boxes will appear respective to the algorithm. 

For each entry box of hyperparameters, you can list several options for each hyperparameter that you would like to test in a pseudo grid search approach. A grid search is a method for hyperparameter optimization where test several hyperparameters at once as a grid of values then evaluate every position on the grid as a different combination of hyperparameters. Our approach goes through each hyperparameter value and assesses the combinations individually. You may then evaluate each combination of hyperparameters once the dimensionality reduction visualization saves in the folder and assess the effectiveness of each hyperparameter and test out other combinations to fine-tune your approach. 

Below, you will find an explanation of each algorithm and their associated hyperparameters as well as suggested entry values with further documentation. Essentially this step will require a trial-and-error approach with finding the combination of hyperparameters where the resulting visualization best fits the data according to your understanding. 


<p>
<img src="https://github.com/sgoldenlab/simba/blob/master/images/UMAP.PNG" />
</p>

>**Note:** For each hyperparameter, pay attention to the type of number required for input as designated in bold, being either integer (whole number) or float (decimal value)

[UMAP](https://umap-learn.readthedocs.io/en/latest/api.html): UMAP, or Uniform Manifold Approximation & Projection, is a manifold learning and dimensionality reduction algorithm that is designed to be compatible with scikit-learn. It is founded with three assumptions about the data: the data is uniformly distributed on Riemannain manifold, the Riemannian metric is locally constant, and the manifold is locally connected. Its design works at high speeds, scales well with dataset and dimension size, and better preserves the data's global structure as compared to other dimensionality reduction algorithms. For our UMAP pipeline, there are 4 hyperparameter values to input, being `Distance`, `Neighbors`, `Spread`, and `Dimensions`.     
- `Distance` or min_dist controls how tightly the points are packed together by referring to the minimum distance apart that the points can be represented in the low-dimensional space. The typical range of Distance values are [insert range] with a default of 0.1 **(float)**.      
- `Neighbors` or 'n_neighbors' controls the balance between local and global data structure, where low values will concentrate more on the data's local structure and high values will concentrate on the larger neighborhoods of each data point while losing finer detail structure **(integer)**.        
- `Spread` refers to the effecctive scale of the embedded data points, and combined with min_distance, it determines how clustered the embedded points are to one another. A typical range would be [insert range] and the default value is 1.0 **(float)**.       
- `Dimensions` **(integer)**

<p>
<img src="https://github.com/sgoldenlab/simba/blob/master/images/tsne.PNG" />
</p>


[t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html): t-SNE or t-distributed Stochastic Neighbor Embedding works to convert similarities between data points to joint probabilities while minimizing the Kullback-Leibler divergence between high and low dimensional data joint probabilities.  For our t-SNE pipeline, there are 3 hyperparameters of `Perplexity`, `Iterations`, and `Dimensions`.      
- `Perplexity` is similar to UMAP's nearest "n_neighbors" as it again balances local and global data structure. The suggested range of values is between 5 and 50, with a default of 30 **(float)**.       
- `Iterations` or n_iter looks at the maximum number of iterations for the optimization. The suggested minimum value is 250, and the default is 300 **(integer)**.     - `Dimensions` **(integer)**

<p>
<img src="https://github.com/sgoldenlab/simba/blob/master/images/PCA.PNG" />
</p>


[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html): PCA or Principal Component Analysis, is a linear dimensionality reduction technique that uses Singular Value Decomposition (SVD) of data to reduce it to a lower dimensional space. The input data is not scaled but centered per feature before SVD is applied. For the PCA pipeline, there is one main hyperparameter: `n-components`. 
- `n-components` [come back]


>**Note:** If you are inputting multiple values for the hyperparameter entry box, you must do so without commas. For example, you can list '10, 20, 30' as `10 20 30` instead of `10, 20, 30`.

Once you have entered the values for each hyperparameter box, you can save the resulting .npy array by clicking the `Save dimensionality reduction .npy `, which will automatically save in a new folder within the project folder labeled 'dimensionality_reduction'. After selecting the hyperparameter combination based on visualizations that you would like to cluster, this .npy array can then be used as the input for the following clustering step.

To save the visualizations associated with each algorithm, click the `Save visualizations` button, and each resulting combination of the hyperparameters will save as its own scatter plot visualization in the same folder. Note that the combination can be distinguished given the file name that will read `algorithm_reduced_features_hyperparameter_value_.npy`. For example, a UMAP visualization/.npy that features the hyperparameters of spread = 1.0, neighbors = 3, distance = 0.0, and dimensions = 2 will be named `UMAP_reduced_features_spread_1.0_neighbors_3_dist_0.0_dimensions_2`.

Below, you will find examples of different dimensionality reduction visualizations and their corresponding algorithm. Note that visually, the plots look similar, with respect to the differences found between hyperparameters.

>**Note:** Something about not being able to guarantee results or what it will look like with pseudo grid search approach, no metrics to assess performance of one hyperparameter combination over another.

[insert gif]

## Step 4: Perform Clustering

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clustering2.PNG" />
</p>

The fourth tab will guide you through performing clustering using HDBSCAN with the previous dimensionality reduction results. Clustering in unsupervised analysis is greatly useful in grouping data in clusters based on behavioral similarities and gives insight into underlying patterns that distinguish clustered groups. 

Our primary algorithm of choice is [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html), or Hierarchical Density-Based Spatial Clustering of Applications with Noise, which is a hierarchical clustering algorithm that assesses the proximity of the clusters relative to one another based on the degree of differences found between them. It will automatically cluster the data without needing to provide a set number of clusters beforehand, and filters out noise and inconsequential data points with a set minimum cluster size. 

HDBSCAN is also useful as it provides us with different types of visualizations to assess how the data was clustered. The first way is through a standard scatter plot visualization, similar to the one represented earlier via the dimensionality reduction visualization. Here, it color codes the data points based on the cluster assignment, as shown below. [x and y axis labels]. 

First, import the dimensionality reduction results from the previous step, by clicking the `Browse File` button and selecting the .npy array that was saved in the 'dimensionality reduction folder' and represents the best-fitting combination of hyperparameters. 
To save the HDBSCAN scatter plot visualization, click the `Visualize/save HDBSCAN scatter plot` button, and the visualization will save to a folder named 'clustering' within the project folder. An example of this clustered scatter plot can be shown below.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/HDBSCAN_scatter.PNG" />
</p>

The second form of visualization is via a hierarchical tree plot, which represents the cluster tree as a dendrogram. At each of the nodes the data is split off into their respective clusters, and the width of each branch represents the number of data points in the cluster at that level. The clusters are also color-coded with a circle around the cluster branch. The y-axis is labeled by the lambda value, otherwise known as 1/distance of the data points [double check], and the legend designates a color gradient representing the number of data points being split per node. An example of the tree plot can be shown below. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/HDBSCAN_tree.PNG" />
</p>

To save the HDBSCAN tree plot visualization, click the `Visualize/save HDBSCAN tree plot` button, and the visualization will also save in the 'clustering' folder.

Finally, we can save the resulting dataset that assigns clusters to each behavioral bout as was represented by the original dataset we created. As shown below, the .csv file saves a dataset with 6 columns: the x and y-dimensions of the behavioral bout data point (based on the dimensionality reduction results), the video from which the bout was taken from, the bout frame start and end, and the cluster the bout was assigned to. Note that values of -1 result in the bout not being assigned to a cluster, as it was filtered out as noise or insignificant. A cluster number of 0 still represents a considerable cluster. An example of what the clusters .csv may look like can be found below.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clusters_csv.PNG" />
</p>

This file is saved as both a .csv and .pkl file, where the .csv file can be opened to evaluate the cluster assignment, and the .pkl file can be used to train the model in the following step. To save the clusters CSV and .pkl, click the `Save clusters .csv & .pkl` button and the files will save to the same 'clustering' folder. 

[insert clusters CSV and file saving both .csv and .pkl]

[insert gif]

## Step 5: Train Model

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/train_classifier.PNG" />
</p>

The next step involves training our model based on the input data and the cluster groups generated in the previous step. We can then find patterns within the data mainly focusing on feature correlation and how well each feature correlates to or predicts the clusters amongst the data. Note that here lies the main difference between supervised and unsupervised learning: unlike supervised learning, we do not have any outputs or target variables such as metrics in which we can use to assess to assess the performance of the model. Rather, we must interpret the classification results ourselves and make sense of what is generated in relation to the starting data. Our model utilizes random forest classifiers [discuss more].

To start, import the condensed dataset .pkl we generated in the second step by browsing for the file in the 'create_dataset' folder, then import the cluster .pkl file generated in the previous step under the 'clustering' folder. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/feature_correlation.PNG" />
</p>


[insert gif]


## Step 6: Visualize Clusters 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/visualize_clusters2.PNG" />
</p>


The final step allows you to visualize the cluster assignments for each behavioral bout in the video of your choice and make sense of the type of behavior associated with each cluster. There are two forms of visualizations, being the original video clip of the animal behavioral bout saved with the cluster assignment, as well as a dynamic "skeleton" showing the animal movement and associated cluster behavior type. 


To visualize the clusters, we must begin by importing the clusters CSV file as was saved in a previous step that associates each bout with a cluster. Select this file from the respective folder by clicking `Browse File` next to the `Import clusters .csv file`. Then, choose a particular video from the collection of videos in which the data analysis was performed on that you would like to see clips from specifically, and input the exact video name in the `Video name` entry box. An example would be: `Box2-20201003T151955-152154_shorten_frame_no`. 

Next, import both the folders of videos and initial datasets (those which were used in [Step 2](https://github.com/sgoldenlab/simba/blob/master/docs/unsupervised_classification.md#step-2-create-dataset) by clicking the `Browse Folder` button. Finally, import the CSV file that names the headers for the animal body parts, as can be found and saved from [detail folder/SimBA step where its saved]. 

To save the original video clips of the animal behavior, click the `Save original video clips` button. Similarly, you can click the `Save skeleton clips` button to save the skeletal movement representation of the behavioral bout. In a new folder within the project folder labeled 'visualize_clusters', you will find each bout saved with the cluster assignment, the type of video clip, the clip number depicting the bout number within a single video, and the video name. An example would be `Cluster_2_OriginalClip_#4_Box2-20201003T151955-152154_shorten_frame_no.avi"` [change to .mp4 later]. 

Below we have demonstrated examples of what each of the clip types look like. Notice how the skeleton movements mimic those of the original video clip, only that it is annotated to show when a cluster behavior is coming up and when it is happening in real time. Each clip will represent a specific bout but there can be many clips associated with a single cluster, labeled by the associated video name and clip within that video. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/original_clip0.PNG" />
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/skeleton0.PNG" />
</p>

## OTHER UNSUPERVISED MACHINE LEARNING PACKAGES FOR POSE-ESTIMATION DATA

VAME - [GitHub](https://github.com/LINCellularNeuroscience/VAME) - [Paper](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v1.full.pdf)

B-SOID - [GitHub](https://github.com/YttriLab/B-SOID) - [Paper](https://www.biorxiv.org/content/10.1101/770271v2)

PS-VAE - [Paper](https://www.biorxiv.org/content/10.1101/2021.02.22.432309v1.full)

VAE-SNE - [GitHub](https://github.com/jgraving/vaesne/issues) - [Paper](https://www.biorxiv.org/content/10.1101/2020.07.17.207993v1)





[insert gif]
