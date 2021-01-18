# Kleinberg behavior burst smoothing in SimBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Kleinberg_1.png" />
</p>

Classifiers generated in SimBA provide a classification for the presence vs. absence of a behavior of interest in every frame of the analysed videos. This mean that, according to the classifier, a behavioural bout can be a short as one frame of the video (e.g., 20ms in a 50 fps video). Similarly, according to the classifier, two behavioral bouts can also be separated by a single frame where the classifier judges the behavioral event to be absent. Often we want to smooth these classified events, so that the (i) behaviours expressed for very short periods are removed, and (ii) behavioral bouts seperated by very short time are concatenated. We can do this by setting [heuristic rules in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data) when we run the classifiers (i.e., a behavioural bout cannot be shorter than N milliseconds). 

However, a more sophisticated (albeit less interpretable) approach is to use infinite hidden Markov models through the [Kleinberg burst detection method](https://link.springer.com/article/10.1023/A:1024940629314) to unify perdiods when the behavior is more and less likely. The Kleinberg burst detection method for smoothing behavioral data was championed by the Curley lab in the excellent [Lee et al, 2019](https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0220596) paper. SimBA performs Kleinberg burst smoothing by using a modified version of the no-longer maintained [pyburst](https://pypi.org/project/pybursts/) package. 

