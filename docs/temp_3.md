
### REFACTOR CLASSIFICATIONS ACCORDING TO MUTUAL EXCLUSIVITY RULES

When using multiple classifiers, it may happen that we get classification results indicating that the animal are doing several, mutually exclusive, behaviors in any one frame. An example would be that the animal is performing `slow running` and `fast running` within the same frame. SimBA has several methods for implementing user-defined heurstic rules that corrects for this. 

Here we will go through a few examples of different mutual exclusivity rules. If you find that your specific use-case is missing, let us know and we will get it into the SimBA GUI. 

In the `RUN MACHINE MODEL` frame in the `Run machine model` tab, click on `MUTUAL EXCLUSIVITY` and you should see this pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_new_1.png" />
</p>

At the top there is a frame titled `EXCLUSIVITY RULES #`, use the drop-down manu to select the number of rules you which to apply. Once a value is selected, you should see the number of rows change in the bottom `RULE DEFINITIONS` window. 

>Note: The rules will be applied sequentially on each file inside the `project_folder/csv/machine_results` directory.

##### Scenario 1: When three classifications are occuring in a given frame, set the classifier with the highest classification probability.

Leave the 

##### Scenario 2: When three classifications are occuring in a given frame, set a defined classifier to present and the other two to absent (regardless of classification probabilities). 















