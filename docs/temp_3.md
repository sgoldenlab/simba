
### REFACTOR CLASSIFICATIONS ACCORDING TO MUTUAL EXCLUSIVITY RULES

When using multiple classifiers, it may happen that we get classification results indicating that the animal are doing several, mutually exclusive, behaviors in any one frame. An example would be that the animal is performing `slow running` and `fast running` within the same frame. SimBA has several methods for implementing user-defined heurstic rules that corrects the data according to mutual exclusivity rules. 

We will go through a few examples of different mutual exclusivity rules and how to apply them. If you find that your specific use-case is missing, then let us know and we will get it into the SimBA GUI. 

In the `RUN MACHINE MODEL` frame in the `Run machine model` tab, click on `MUTUAL EXCLUSIVITY` and you should see this pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_new_1.png" />
</p>

At the top there is a frame titled `EXCLUSIVITY RULES #`, use the drop-down manu to select the number of rules you which to apply. Once a value is selected, you should see the number of rows change in the bottom `RULE DEFINITIONS` window. 

>Note: The rules will be applied **sequentially** on each file inside within the `project_folder/csv/machine_results` directory.

### Scenario 1: When several mutually exclusive classifications are occuring in a given frame, set the classifier with the highest classification probability to present and the remaining classifiers to absent.

Leave the `HIGHEST PROBABILITY` checkbox ticked, and tick the checkboxes for the classifiers that are mutually exclusive. For example,
if you want select the highest probability classifier between `Attack` and `Sniffing` (when both `Attack` and `Sniffing` is classified as present within any given single frame), tick the checkboxes under the `Attack` and `Sniffing` headers.

Next, we need to tell SimBA how to deal with occations when `Attack` and `Sniffing` classification probabilities are equal. In the `TIE BREAK` dropdown, select the classifier that should "win" when classification probabilities of `Attack` and `Rear` are equal.
Alternatively, if we want SimBA to **not choose** a winner when classification probabilities of `Attack` and `Rear` are equal, and instead skip applying the rule to the file all together, then tick the `SKIP ON EQUAL` checkbox (you should see the TIE BREAK drop-down greyed out when the `SKIP ON EQUAL` checkbox is checked)..

Once complete, click `RUN`. SimBA will copy the files prior to applying to rules into the `project_folder/csv/machine_results/Prior_to_mutual_exclusivity_datetime_stamp` sub-directory. The new files, with the corrected classifications, are then saved in the  `project_folder/csv/machine_results/` directory.

### Scenario 2: When several mutually exclusive classifications are occuring in a given frame, set a defined classifier to present and the others to absent (regardless of classification probabilities).

Begin by **un-ticking** the `HIGHEST PROBABILITY` checkbox (this will make the `DETERMINATOR` and `THRESHOLD` available). Next, tick the checkboxes for the classifiers which are mutually exclusive. Next, use the dropdown under the `DETERMINATOR` header to select the classifier that
should **WIN** when the chosen classifiers are occuring at the same time. **Leave the threshold value set to 0.00** (see more info below on this setting).

For example, if I want to set `Attack` to present and `Rear` to absent when both `Attack` and `Rear` is classified as present, (i)) 
I tick the checkboxes for `Attack` and `Rear, (ii) select `Attack` in the `DETERMINATOR` dropdown.

Once complete, click `RUN`. SimBA will copy the files prior to applying to rules into the `project_folder/csv/machine_results/Prior_to_mutual_exclusivity_datetime_stamp` sub-directory. The new files, with the corrected classifications, are saved in the  `project_folder/csv/machine_results/` directory.

### Scenario 3: When several mutually exclusive classifications are occuring in a given frame, set a defined classifier to present and the others to  absent only when the defined classifier is above a certain threshold.

Begin by **un-ticking** the `HIGHEST PROBABILITY` checkbox (this will make the `DETERMINATOR` and `THRESHOLD` available options available). Next, tick the checkboxes for the classifiers that are mutually exclusive. Next, use the dropdown under the `DETERMINATOR` header to select the classifier that
should **WIN** when the chosen classifiers are occuring at the same time. 

Lastly, set the threshold for the `DETERMINATOR` classifier. For example, if I want to set `Attack` to present and `Rear` to absent when both `Attack` and `Rear` is classified as present and the `Attack` classification probability is equal or above 0.6, (i))
I tick the checkboxes for `Attack` and `Rear, (ii) select `Attack` in the `DETERMINATOR` dropdown, (iii) set the `THRESHOLD` to `0.6` and click `Run`. 
    
In frames when both `Attack` and `Rear` is classified as present and the `Attack` classification probability is equal or above 0.6, `Rear` classifications will be set to absent. 
    
> Note: In frames when both `Attack` and `Rear` is classified as present and the `Attack` classification probability is below 0.6, `Rear` classifications will remain as present.









