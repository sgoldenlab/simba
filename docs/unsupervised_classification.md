# SimBA Unsupervised Classification

## Overview

[highlight features of unsupervised vs. supervised, explain tutorial scenario]. The SimBA Unsupervised Classification pipeline and GUI were created by [Simon Nilsson](https://github.com/sronilsson) and [Aasiya Islam](https://github.com/aasiya-islam).

## Pipeline

Training an unsupervised classifier involves the creation of an algorithm that clusters animal activity based on behavioral similarities, typically with three main steps: data pre-processing, dimensionality reduction, and cluster assignment. 
The SimBA Unsupervised Classification pipeline consists of six main steps: 

**1) Save Project Folder**- allows user to create an unsupervised project folder and saves outputs to future steps in encompassed folder     
**2) Create Dataset**- cleans and pre-processes the machine results and localizes relevant data into a single dataset      
**3) Perform Dimensionality Reduction**- allows user to select a dimensionality reduction algorithm to apply and visualize data with     
**4) Perform Clustering**- assigns and visualizes clusters with HDBSCAN from the dimensionality reduction results       
**5) Train Model**- trains model on the clusters generated previously and saves permutational importance and feature correlation metrics     
**6) Visualize Clusters**- visualize behavioral bouts corresponding to clusters as original video clips or simulated skeleton movements

The outputs generated from each step can be saved into their respective folders encompassed within the main unsupervised project folder and taken as inputs for subsequent steps throughout the pipeline.

### Step 1: Save Project Folder 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/create_folder.PNG" />
</p>

The first step allows the user to create and save a project folder for their unsupervised analysis results. 
For each subsequent step in the pipeline, the individual outputs will save into that step's respective folder that is automatically created upon the use of each step. 

To save a project folder, specify the folder path where you would like the project folder to be saved into by clicking the ```Browse Folder``` button and selecting a folder, and designate a name for the project folder with the ```Project Name``` entry box. 
Note that the folder name will save as "unsupervised_projectname" with "projectname" being the name you filled out the entry box with. After you have specified the folder path and designated a project name, click the ```Create folder``` button to save the folder.

[insert GIF tutorial]

### Step 2: Create Dataset

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/create_dataset.PNG" />
</p>

The second step pre-processes and cleans the data from prior machine results and saves the data relevant for unsupervised analysis into a single dataset. 
This step serves to extract behavioral bouts of interest as designated by the classifier and find mean feature values while dropping all features that are not relevant.

To begin, first import a folder of the machine results saved as CSV datasets by clicking the ```Browse Folder``` button and selecting the folder. 

[note that it takes a while, GIF of saving dataset then showing pkl file saved]


### Step 3: Perform Dimensionality Reduction

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/perform_DR.PNG" />
</p>

https://umap-learn.readthedocs.io/en/latest/
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

### Step 4: Perform Clustering

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/perform_clustering.PNG" />
</p>

### Step 5: Train Model

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/train_model.PNG" />
</p>

### Step 6: Visualize Clusters 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/visualize_clusters.PNG" />
</p>
