# Kleinberg behavior classification smoothing in SimBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Kleinberg_1.png" />
</p>

Classifiers generated in SimBA provide a classification for the presence vs. absence of a behavior of interest in every frame of the analysed videos. This mean that, according to the classifier, a behavioural bout can be a short as one frame of the video (e.g., 20ms in a 50 fps video). Similarly, according to the classifier, two behavioral bouts can also be separated by a single frame where the classifier judges the behavioral event to be absent. Often we want to smooth these classified events, so that the (i) behaviours expressed for very short periods are removed, and (ii) behavioral bouts seperated by very short time are concatenated. We can do this by setting [heuristic rules in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data) when we run the classifiers (i.e., a behavioural bout cannot be shorter than N milliseconds). 

However, a more sophisticated (albeit less interpretable) approach is to use infinite hidden Markov models through the [Kleinberg burst detection method](https://link.springer.com/article/10.1023/A:1024940629314) to unify perdiods when the behavior is more and less likely. The Kleinberg burst detection method for smoothing behavioral data was championed by the Curley lab in the excellent [Lee et al. 2019](https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0220596) paper. SimBA performs Kleinberg burst smoothing by using a modified version of the no-longer maintained [pyburst](https://pypi.org/project/pybursts/) package. 

 ## How to use Kleinberg Filter in SimBA
 
1. After running the machine model (i.e., after clicking on `Run RF Model` as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model)), go ahead and click on the `Kleinberg Smoothing` button.
 
 ![](/images/kleinberg1.PNG)
 
2. You should see the below `settings menu` pop up. Check the classifier(s) that you wished to apply the Kleinberg smoothing for and set the "hyperparameters". See the section below for more information about the hyperparameters and the default settings. Click on the `Apply Kleinberg Smoother` button to apply the filter.

![](/images/kleinberg2.PNG)

3. The files in the `project_folder/csv/machine_results` will be overwritten with new files, with the same file names as the original files, but now containing the kleinberg smoothened classifications for the selected classifiers. To compare the results, either (i) back-up the original files and open it up alongside the newly generated file and compare the `0` and `1` in the relevant classification columns in the right-most part of the file. Or, alternative, generate new classifier visualizations as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions) and compare the classifications pre- and post smoothing. 

If satisfied with the results, go ahead and compute new descriptive stastistics as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-9-analyze-machine-results).


### Hyperparameters

Through the SimBA menues the user can access three hyperparameters. For more information of what the hyperparameters mean, check out the [R 'bursts' package documentation](https://cran.r-project.org/web/packages/bursts/bursts.pdf) the [Lee et al. 2019](https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0220596) paper, or the [original paper](https://link.springer.com/article/10.1023/A:1024940629314). 

In short,

*(i) Sigma* - Higher values and the discrepancy between no-behavior events and behavior events has to be larger for the event to be recognised. **Higher sigma values and fewer behavioural bursts will be recognised (Default: 2)** 

*(ii) Gamma* - Higher gamma values and behaviors needs to be sustained for a longer time for them to be recognised as burts. **Higher gamma values and fewer behavioural bursts will be recognised (Default: 0.3)**

 *(ii) Hierarchy* - Which order or depth or the markov chain should be considered when evaluating the bursts. **Higher hierarchy values values and fewer behavioural bursts are likely to be recognised (Default: 2)**
