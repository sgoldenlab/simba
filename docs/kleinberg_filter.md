# Kleinberg behavior classification smoothing in SimBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Kleinberg_1.png" />
</p>

Classifiers generated in SimBA provide a classification for the presence vs. absence of a behavior of interest in every frame of the analysed videos. This mean that, according to the classifier, a behavioural bout can be a short as one frame of the video (e.g., 20ms in a 50 fps video). Similarly, according to the classifier, two behavioral bouts can also be separated by a single frame where the classifier judges the behavioral event to be absent. Often we want to smooth these classified events, so that the (i) behaviours expressed for very short periods are removed, and (ii) behavioral bouts seperated by very short time are concatenated. We can do this by setting [heuristic rules in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data) when we run the classifiers (i.e., a behavioural bout cannot be shorter than N milliseconds). 

However, a more sophisticated (albeit less interpretable) approach is to use infinite hidden Markov models through the [Kleinberg burst detection method](https://link.springer.com/article/10.1023/A:1024940629314) to unify perdiods when the behavior is more and less likely. The Kleinberg burst detection method for smoothing behavioral data was championed by the Curley lab in the excellent [Lee et al. 2019](https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0220596) paper. SimBA performs Kleinberg burst smoothing by using a modified version of the no-longer maintained [pyburst](https://pypi.org/project/pybursts/) package. 

After applying the Kleinberg smoother, you can expect the output to look like below - with the original video on the left and the kleinberg smoothened video on the right. Click [HERE](https://youtu.be/HRzQ64nupM0) to see a longer higher resolution video of expected results. 


<p align="center">
  <img src="https://github.com/sgoldenlab/simba/blob/master/images/klenberg.gif" width="850"/>
</p>

>Note: If you are smoothing your data using the Kleinberg smoother, I recommend refraining from setting [heuristic rules in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data) when we run the classifiers (i.e., a behavioural bout cannot be shorter than N milliseconds) and instead let the Kleinberg algorithm take care of **all** of the smoothing. Thus, when you first run the classifier, set the **min bout duration** for the classifiers you want to smooth (as descibed [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#critical-validation-step-before-running-machine-model-on-new-data)) to `0`. 

 ## How to use Kleinberg Filter in SimBA
 
1. After running the machine model (i.e., after clicking on `Run RF Model` as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model)), go ahead and click on the `Kleinberg Smoothing` button.

 ![](/images/kleinberg1.PNG)
 
2. You should see the below `settings menu` pop up. Check the classifier(s) that you wished to apply the Kleinberg smoothing for and set the "hyperparameters". See the section below for more information about the hyperparameters and the default settings. Click on the `Apply Kleinberg Smoother` button to apply the filter.

![](/images/kleinberg2.PNG)

3. The files in the `project_folder/csv/machine_results` will be overwritten with new files, with the same file names as the original files, but now containing the kleinberg smoothened classifications for the selected classifiers. To compare the results, either (i) back-up the original files and open it up alongside the newly generated file and compare the `0` and `1` in the relevant classification columns in the right-most part of the file. Or, alternative, generate new classifier visualizations as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions) and compare the classifications pre- and post smoothing.

4. For troubleshooting purposes - for each classifier you smooth - SimBA will save a log file in the `project_folder/csv/logs` directory that is named after the classifier name and the current date and time, for example *Kleinberg_log_Attack_20210204202223.csv*. This file contains information on **all** the classified bouts found by the Kleinberg smoother, their hierarchy, and start and end frame. It may look like this:

![](/images/Kleinberg_10.png)

Here, every row is a classified behavioral bout. The *Hierarchy* columns denotes which level in the hierarchy the bout belongs to. The *Start* columns denotes the frame when the behavioral bout started within this part of the hierarchy. The *End* columns denotes the frame when the behavioral bout ended within this part of the hierarchy. If you look carefully you see that the lower hierarchies encompass the same boural events further up the hierarchy.  

If you are unhappy with the results, I suggest you tinker with the hierarchy value based on the result in this logfile - have a look with the majority of your classifies bouts are located and set your hierarchy hyperparameter accordingly. 

If satisfied with the results, go ahead and compute new descriptive stastistics as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-9-analyze-machine-results).


### Hyperparameters

Through the SimBA menues the user can access three hyperparameters. For more information of what the hyperparameters mean, check out the [R 'bursts' package documentation](https://cran.r-project.org/web/packages/bursts/bursts.pdf) the [Lee et al. 2019](https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0220596) paper, or the [original paper](https://link.springer.com/article/10.1023/A:1024940629314). 

In short,

*(i) Sigma* - Higher values and the discrepancy **in time** between no-behavior events and behavior events has to be larger for the event to be recognised. **Higher sigma values and fewer, longer, behavioural bursts will be recognised. Lower sigma values and a greater number of shorter behavioral bursts will be recognized. (Default: 2)** 

*(ii) Gamma* - Higher gamma values and behaviors needs to be sustained for a longer time for them to be recognised as burts. I.e., gamma represents the cost associated with entering a time-period of behevior expression. **Higher gamma values and fewer behavioural bursts will be recognised. Lower gamma values and a greate number of behavioural bursts will be recognised (Default: 0.3)**

 *(ii) Hierarchy* - Which order or depth or the markov chain should be considered when evaluating the bursts. **Higher hierarchy values and fewer behavioural bursts will to be recognised (Default: 1)**
 
 For more information on the hyperparameters [HERE](https://nikkimarinsek.com/blog/kleinberg-burst-detection-algorithm) is an excellent blog post walk-through.
