
#### REFACTOR CLASSIFICATIONS ACCORDING TO MUTUAL EXCLUSIVITY RULES

When using multiple classifiers, it may happen that we get classification results indicating that the animal are doing several, mutually exclusive, behaviors in any one frame. An example would be that the animal is performing `slow running` and `fast running` within the same frame. SimBA has several methods for implementing user-defined heurstic rules that corrects for this. 

Here we will go through a few examples of different mutual exclusivity rules. If you find that your specific use-case is missing, let us know and we will insert it in the SimBA GUI. 

In the `RUN MACHINE MODEL` frame in the `Run machine model` tab, click on `MUTUAL EXCLUSIVITY` and you should see the following pop-up:














