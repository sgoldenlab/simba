# SimBA Unsupervised Classification

## Overview

[highlight features of unsupervised vs. supervised, explain tutorial scenario and how this continues from prior analysis and Kleinberg, how to navigate from existing GUI]. The SimBA Unsupervised Classification pipeline and GUI were created by [Simon Nilsson](https://github.com/sronilsson) and [Aasiya Islam](https://github.com/aasiya-islam).

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

## Step 1: Save Project Folder 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/create_folder.PNG" />
</p>

The first step allows the user to create and save a project folder for their unsupervised analysis results. 
For each subsequent step in the pipeline, the individual outputs will save into that step's respective folder that is automatically created upon the use of each step. 

To save a project folder, specify the folder path where you would like the project folder to be saved into by clicking the ```Browse Folder``` button and selecting a folder, as the folder path will replace the ```No folder selected``` box, and designate a name for the project folder with the ```Project Name``` entry box. 
Note that the folder name will save as "unsupervised_projectname" with "projectname" being the name you filled out the entry box with. After you have specified the folder path and designated a project name, click the ```Create folder``` button to save the folder.

[insert GIF tutorial]

## Step 2: Create Dataset

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/create_dataset.PNG" />
</p>

The second tab will walk you through pre-processing and cleaning the data from prior machine results, and saving the data relevant for unsupervised analysis into a single dataset. 
This step serves to extract behavioral bouts of interest as designated by the classifier and find mean feature values while dropping all features that are not relevant.

To begin, first import a folder of the machine results saved as CSV datasets by clicking the ```Browse Folder``` button and selecting the folder of datasets. An example of the machine results folder and CSV format and can be shown below, and is generated as directed by this documentation (hyperlink relevant documentation, discuss kleinberg or link documentation]



<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/dataset_folder2.PNG" />
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/machine_results.PNG" />
</p>


Next, import the file that lists the features that you would like to remove or disregard in the classification. 
The purpose of this is to perform the classification without the irrelevant features (double check this) and drop them before saving our condensed dataset for future analysis. 
You can similarly select the ```Browse File``` button to search for and select this file, and the file should be formatted similar to as shown below.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/features2remove.PNG" />
</p>


Finally, input the name of the classifier you would like to focus your analysis on, such as "Attack" in the ```Classifier name``` entry box. Note the formatting of this name should match that depicted in the machine results or prior analysis.

Once everything has been imported, you can begin the pre-processing by clicking the ```Generate/save dataset``` button. 

>**Note:** This step may take a few minutes to process all of the machine results depending on the length and number of datasets in the folder. Once everything has been processed you will see the .pkl file 

Once everything has been processed, you will observe that a new folder labeled 'create_dataset' has been saved in your project folder, and inside the 'create_dataset' folder, there will be a single .pkl file saved under the name of the classifier you inputted, such as "Attack.pkl". The .pkl file is a serialized object file that can be read in and deserialized in future steps to use in our classification, and is mainly used for storage efficiency. It cannot be opened on its own like you would a CSV file. 

[insert pic of pkl file saved]


[GIF of saving dataset then showing pkl file saved]


## Step 3: Perform Dimensionality Reduction

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/perform_DR.PNG" />
</p>

The third tab will walk you through selecting a dimensionality reduction algorithm and inputting hyperparameters for the respective algorithm. Dimensionality reduction is used in unsupervised learning to transform the data from high-dimension to low-dimension, simplifying a data point to a single x,y coordinate. It is especially useful as it reduces the number of variables/features while still maintains data integrity and retains meaningful properties of the intrinsic dimension. Here, it is valuable for our visualization of the data as pre-processed and generated in the previous step and gives us our first glimpse of data relationships prior to clustering in the next step. 

We have provided 3 options for dimensionality reduction algorithms to choose from and use in your analysis, being [UMAP](https://umap-learn.readthedocs.io/en/latest/), [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Each has their own set of hyperparameters to perform the dimensionality reduction with, as outlined below, but can be inputted in the same manner. 

>**Note:** To read more about the supporting documentation for each of the hyperparameters, please click on the algorithm name and follow the hyperlink. 

First, import the pre-processed feature dataset as was saved as a .pkl file from the previous step by clicking the ```Browse File``` button and navigating to the 'create_dataset' folder. 

Next, specify the target group you would like to focus on for the analysis in the ```Target group``` entry box, such as "RI Male". [more info on target group].

Then, you can select between your dimensionality reduction algorithms via the dropdown menu on the right of the `Select dimensionality reduction algorithm` label, and by clicking the small box inside of the box that says `UMAP` (which is the default algorithm), the other options should be available. As each algorithm is selected, a new set of hyperparameter entry boxes will appear respective to the algorithm. 

For each entry box of hyperparameters, you can list several options for each hyperparameter that you would like to test in a pseudo grid search approach. A grid search is a method for hyperparameter optimization where test several hyperparameters at once as a grid of values then evaluate every position on the grid as a different combination of hyperparameters. Our approach goes through each hyperparameter value and assesses the combinations individually. You may then evaluate each combination of hyperparameters once the dimensionality reduction visualization saves in the folder and assess the effectiveness of each hyperparameter and test out other combinations to fine-tune your approach. 

Below, you will find an explanation of each algorithm and their associated hyperparameters as well as suggested entry values with further documentation. Essentially this step will require a trial-and-error approach with finding the combination of hyperparameters where the resulting visualization best fits the data according to your understanding. 

[explain what the hyperparameters do, provide range of values, and link documentation]

<p>
<img src="https://github.com/sgoldenlab/simba/blob/master/images/UMAP.PNG" />
</p>


- UMAP: For UMAP, there are 4 hyperparameters to input, being `Distance`, `Neighbors`, `Spread`, and `Dimensions`. 

<p>
<img src="https://github.com/sgoldenlab/simba/blob/master/images/tsne.PNG" />
</p>


- t-SNE: For t-SNE, there are 3 hyperparameters of `Perplexity`, `Iterations`, and `Dimensions`. 

<p>
<img src="https://github.com/sgoldenlab/simba/blob/master/images/PCA.PNG" />
</p>


- PCA:


>**Note:** If you are inputting multiple values for the hyperparameter entry box, you must do so without commas. For example, you can list '10, 20, 30' as `10 20 30` instead of `10, 20, 30`.

Once you have entered the values for each hyperparamter box, you can save the resulting .npy array by clicking the `Save dimensionality reduction .npy `, which will automatically save in a new folder within the project folder labeled 'dimensionality_reduction'. After selecting the hyperparameter combination based on visualizations that you would like to cluster, this .npy array can then be used as the input for the following clustering step.

To save the visualizations associated with each algorithm, click the `Save visualizations` button, and each resulting combination of the hyperparameters will save as its own scatter plot visualization in the same folder. Note that the combination can be distinguished given the file name that will read [insert file name template]. For example, a visualization that features the hyperparameters of [insert hyperparameter combo] will be named `visualization name`.

Below, you will find examples of different dimensionality reduction visualizations and their corresponding algorithm. Note that visually, the plots look similar, with respect to the differences found between hyperparameters.

>**Note:** Something about not being able to guarantee results or what it will look like with pseudo grid search approach, no metrics to assess performance of one hyperparameter combination over another. 

## Step 4: Perform Clustering

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/perform_clustering.PNG" />
</p>

The fourth tab will guide you through performing clustering using HDBSCAN with the previous dimensionality reduction results. Clustering in unsupervised analysis is greatly useful in grouping data in clusters based on behavioral similarities and gives insight into underlying patterns that distinguish clustered groups. 

Our primary algorithm of choice is HDBSCAN [link documentation], which is a hierarchical clustering algorithm that assesses the proximity of the clusters relative to one another based on the degree of differences found between them. It will automatically cluster the data without needing to provide a set number of clusters beforehand, and filters out noise and inconsequential data points with a set minimum cluster size. 

HDBSCAN is also useful as it provides us with different types of visualizations to assess how the data was clustered. The first way is through a standard scatter plot visualization, similar to the one represented earlier via the dimensionality reduction visualization. Here, it color codes the data points based on the cluster assignment, as shown below. [x and y axis labels]. 

[insert scatter plot]

First, import the dimensionality reduction results from the previous step, by clicking the `Browse File` button and selecting the .npy array that was saved in the 'dimensionality reduction folder' and represents the best-fitting combination of hyperparameters. 
To save the HDBSCAN scatter plot visualization, click the `Visualize/save HDBSCAN scatter plot` button, and the visualization will save to a folder named 'clustering' [double check] within the project folder.

The second form of visualization is via a hierarchical tree plot, which represents the cluster tree as a dendrogram. At each of the nodes the data is split off into their respective clusters, and the width of each branch represents the number of data points in the cluster at that level. The clusters are also color-coded with a circle around the cluster branch. The y-axis is labeled by the lambda value, otherwise known as 1/distance of the data points [double check], and the legend designates a color gradient representing the number of data points being split per node. An example of the tree plot can be shown below. 

[insert tree plot graphic]

To save the HDBSCAN tree plot visualization, click the `Visualize/save HDBSCAN tree plot` button, and the visualization will also save in the 'clustering' folder.

Finally, we can save the resulting dataset that assigns clusters to each behavioral bout as was represented by the original dataset we created. As shown below, the CSV file saves [insert what it contains]. This file is saved as both a CSV and .pkl file, where the CSV file can be opened to evaluate the cluster assignment, and the .pkl file can be used to train the model in the following step.

[insert clusters CSV and file saving both CSV and .pkl]

To save the clusters CSV and .pkl, click the `Save clusters CSV` button [update later] and the files will save to the same 'clustering' folder. 

## Step 5: Train Model

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/train_model.PNG" />
</p>

## Step 6: Visualize Clusters 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/visualize_clusters.PNG" />
</p>


The final step allows you to visualize the cluster assignments for each behavioral bout in t
