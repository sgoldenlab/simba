# SimBA (Simple Behavioral Analysis)

SimBA is a toolkit for creating supervised machine-learning classifiers of animal social and non-social behavior from pose-estimation data, without requiring a programming background.

[![pypi](https://img.shields.io/pypi/v/simba-uw-tf-dev.svg)](https://pypi.org/project/Simba-UW-tf-dev/)
[![py3.10 tests](https://github.com/sgoldenlab/simba/actions/workflows/tests_py310.yml/badge.svg)](https://github.com/sgoldenlab/simba/actions/workflows/tests_py310.yml)
[![py3.6 tests](https://github.com/sgoldenlab/simba/actions/workflows/tests_py36.yml/badge.svg)](https://github.com/sgoldenlab/simba/actions/workflows/tests_py36.yml)
![Docs](https://readthedocs.org/projects/simba-uw-tf-dev/badge/?version=latest&style=flat)
[![License: Modified BSD 3-Clause](https://img.shields.io/badge/License-Modified%20BSD%203--Clause%20(Academic%2FResearch)-blue.svg)](LICENSE)
[![Gitter chat](https://badges.gitter.im/SimBA-Resource/community.png)](https://gitter.im/SimBA-Resource/community)
[![Download: Weights](https://img.shields.io/badge/Download-Weights-orange.svg)](https://osf.io/5t4y9/)
[![SimBA: listserv](https://img.shields.io/static/v1?label=SimBA&message=listserv&color=blue)](https://docs.google.com/forms/d/e/1FAIpQLSfjbjae0XqNcl7GYOxmqvRsCveG-cmf4p4hBNNJ8gu5vPLHng/viewform)
[![DOI](https://zenodo.org/badge/206670333.svg)](https://zenodo.org/badge/latestdoi/206670333)

[![Downloads](https://pepy.tech/badge/simba-uw-tf-dev/month)](https://pepy.tech/project/simba-uw-tf-dev)
[![Downloads](https://pepy.tech/badge/simba-uw-tf-dev)](https://pepy.tech/project/simba-uw-tf-dev)
[![Live Stats](https://img.shields.io/website?url=https%3A%2F%2Fsronilsson.github.io%2Fdownload_stats%2F&up_message=online&down_message=offline&label=download%20statistics%20dashboard)](https://sronilsson.github.io/download_stats/)

**Manuscript: [Simple Behavioral Analysis (SimBA) as a platform for explainable machine learning in behavioral neuroscience](https://www.nature.com/articles/s41593-024-01649-9)**
**Pre-print: [Simple Behavioral Analysis (SimBA) – an open source toolkit for computer classification of complex social behaviors in experimental animals](https://www.biorxiv.org/content/10.1101/2020.04.19.049452v2)**

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/landing_1.gif" />
</p>

## Quickstart

```bash
pip install simba-uw-tf-dev   # Python 3.6 or 3.10
simba                         # launch the GUI
```

See **[Scenario 1](docs/Scenario1.md)** for a worked example that takes raw tracking data through to a validated classifier.

## Installation ⚙️

- [Install SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md)

- [Install SimBA using Anaconda](https://github.com/sgoldenlab/simba/blob/master/docs/anaconda_2025.md)

## Scope & inputs

- **Input:** pose-estimation from [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) (incl. [multi-animal](docs/Multi_animal_pose.md)), [SLEAP](https://sleap.ai), [DeepPoseKit](https://github.com/jgraving/DeepPoseKit), [DANNCE](https://github.com/spoonsso/dannce) (3D), [MARS](https://github.com/neuroethology/MARS), [FaceMap](https://github.com/MouseLand/facemap), [APT](https://github.com/kristinbranson/APT), [SuperAnimal-TopView](docs/superanimal_topview_project.md), [YOLO](https://github.com/ultralytics/ultralytics), or [blob tracking](docs/blob_track.md); user-defined pose schemes supported.
- **Output:** per-frame behavior classifiers with standard evaluation (precision/recall, learning curves, permutation importance) and [SHAP-based explainability](docs/SHAP.md).
- **Validation:** classifier libraries validated in mice and rats; all data, models, and annotations available on [OSF](https://osf.io/tmu6y/).
- **Analyses** — all produce descriptive statistics, export to CSV, and can be split into time-bins:
    - *Behavior:* bout counts, durations, latency to first event, event frequency, severity scoring.
    - *Movement:* velocity, distance travelled, path/trajectory metrics.
    - *Space (ROIs/zones):* time-in-zone, entries, distance moved per zone, behavior × zone; supports animal-anchored zones.
    - *Social & directional:* which animal — or body part, or ROI — each animal is oriented toward.
    - *Temporal structure:* behavior-sequence coupling ([FSTTC](docs/FSTTC.md)), burst detection & smoothing ([Kleinberg](docs/kleinberg_filter.md)).
    - *Turn-key assays:* [spontaneous alternation](docs/spontaneous_alternation.md), pup retrieval, [cue-light](docs/cue_light_tutorial.md), light/dark box, freezing, circling, etc.
    - *Unsupervised:* dimensionality reduction and clustering of behavior.
- **Visualizations** — static plots or overlays burned onto the video, mergeable into one multi-panel video (see [Visualizations](docs/Visualizations.md)):
    - *Tracking overlays:* pose keypoints, skeletons, animal-anchored bounding boxes, blob contours.
    - *Spatial:* path/trajectory plots, ROI/zone overlays, location heatmaps.
    - *Classifier:* annotated prediction videos, Gantt charts, classifier heatmaps, probability plots (incl. interactive grapher), SHAP summary plots.
    - *Movement & directionality:* distance plots, velocity/data line plots, directionality overlays (animal→animal, →body-part, →ROI).
    - *Assay-specific:* cue-light, light/dark box, spontaneous-alternation, circular/polar plots.

##  Documentation: Scenario tutorials

To faciliate the initial use of SimBA, we provide several use scenarios. We have created these scenarios around a hypothetical experiment that take a user from initial use (completely new start) all the way through analyzing a complete experiment and then adding additional experimental datasets to an initial project.

### Scenario 1: [Building classifiers from scratch](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md)

### Scenario 2: [Using a classifier on new experimental data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md)

### Scenario 3: [Updating a classifier with further annotated data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario3.md)

### Scenario 4: [Analyzing and adding new Experimental data to a previously started project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4_new.md)

## Tutorial 📚
- [Analysing animal directions in SimBA](docs/directionality_between_animals.md) 🧭
- [API](https://simba-uw-tf-dev.readthedocs.io/en/latest/index.html) 📘
- [Batch pre-process video using SimBA](docs/tutorial_process_videos.md) 🏭
- [Blob (contour) tracking in SimBA](docs/blob_track.md) 🟣
- [Bounding boxes in SimBA](docs/anchored_rois.md)📦
- [Compute feature subsets in SimBA](docs/feature_subsets.md) 📕
- [Cue-light analyses in SimBA](docs/cue_light_tutorial.md)💡💡
- [Downloading compressed data from the SimBA OSF repository](https://github.com/sgoldenlab/simba/blob/master/docs/using_OSF.md) 💾
- [Explainable machine classifications in SimBA (SHAP)](docs/SHAP.md) 🧮
- [Kleinberg markov chain classification smoothing in SimBA](docs/kleinberg_filter.md) 🔗
- [Mutual exclusivity using heuristic rules in SimBA](docs/mutual_exclusivity_heuristic_rules.md) 📗
- [Process video using SimBA tools](docs/Tools.md) 🔨
- [Recommended hardware](https://github.com/sgoldenlab/simba/blob/master/misc/system_requirements.md) 🖥️
- [Reversing the directionality of classifiers in SimBA](docs/reverse_annotations.md) ⏪
- [SimBA Advanced behavioral annotation interface](docs/advanced_labelling.md) 🏷️
- [SimBA behavioral annotation interface](docs/label_behavior.md) 🏷️
- [SimBA friendly asked questions (FAQ)](docs/FAQ.md) 📕
- [SimBA generic tutorial](docs/tutorial.md) 📘
- [Spike-time correlation coefficients in SimBA](docs/FSTTC.md) 📔
- [Spontaneous alternation in SimBA](/docs/spontaneous_alternation.md)🌽
- [Using DeepLabCut through SimBA](docs/Tutorial_DLC.md) 📗
- [Using DeepPoseKit in SimBA](docs/DeepPoseKit_in_SimBA.md) 📙
- [Using multi-animal pose (maDLC/SLEAP/APT) in SimBA](/docs/Multi_animal_pose.md) 🐭🐭
- [Using the SimBA data analysis and export dashboard](docs/plotly_dash.md) 📊
- [Using third-party annotation tools in SimBA](docs/third_party_annot.md) 🏷️
- [Using user-defined ROIs in SimBA](docs/roi_tutorial_new_2025.md) 🗺️
- [Visualization tools](docs/Visualizations.md) 👁️

## Legacy documentation: General methods

### Step 1: [Pre-process videos](docs/tutorial_process_videos.md) 

### Step 2: [Create tracking model and generate pose-estimation data](docs/Tutorial_DLC.md) 

### Step 3: [Building classfier(s)](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior)

### Step 4: [Analysis/Visualization](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-9-analyze-machine-results)

### [Click here for the full legacy *generic* tutorial on building classifiers in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md).

## Apr-03-2025: Blob (contour) tracking in SimBA

For documentation, see [THIS](https://github.com/sgoldenlab/simba/blob/master/docs/blob_track.md) tutorial on GitHub or [THIS](https://simba-uw-tf-dev.readthedocs.io/en/latest/tutorials_rst/blob_tracking.html) tutorial in the documentation. 
<p align="center">
  <img src="https://github.com/sgoldenlab/simba/blob/master/images/blob_tracking_example.gif" width="650" alt="Demo GIF">
</p>

## Feb-11-2025: SimBA ROI interface update

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_new_1.png" />
</p>

We have improved the GUI for region-of-interest segmentation and analysis, which includes new interactive controls for drawing shapes. 
[Click here to go to the new ROI documentation page.](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md)

The updates primarily serves to improve stability, but includes new methods for [drawing circles](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md#draw-circle), [interactive ROI resizing](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md#changing-roi-shapes), [interactive 
ROI moving](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md#changing-roi-locations) and aesthetics. 

The SimBA region of interest (ROI) interface allows users to define and 
draw ROIs on videos. ROI data can be used to calculate basic descriptive 
statistics based on animals movements and locations such as:

* How much time the animals have spent in different ROIs.
* How many times the animals have entered different ROIs.
* The distance animals have moved in the different ROIs.
* Calculate how animals have engaged in different classified behaviors in each ROI.
* etc....


Furthermore, the ROI data can  be used to build potentially valuable, additional, features for random forest predictive classifiers. Such features can be used to generate a machine model that classify behaviors that depend on the spatial location of body parts in relation to the ROIs. **CAUTION**: If spatial locations are irrelevant for the behaviour being classified, then such features should *not* be included in the machine model generation as they just 
only introduce noise.

## What is SimBA?
Several excellent computational frameworks exist that enable high-throughput and consistent tracking of freely moving unmarked animals. Here we introduce and distribute a pipeline that enabled users to use these pose-estimation approaches in combination with behavioral annotation and generation of supervised machine-learning behavioral predictive classifiers. We have developed this pipeline for the analysis of complex social behaviors, but have included the flexibility for users to generate predictive classifiers across other behavioral modalities with minimal effort and no specialized computational background.  

SimBA does not require computer science and programing experience, and SimBA is optimized for wide-ranging video acquisition parameters and quality. We may be able to provide support and advice for specific use instances, especially if it benefits multiple users and advances the scope of SimBA. Feel free to post issues and bugs here or contact us directly and we'll work on squashing them as they appear. We hope that users will contribute to the community!

- The SimBA pipeline requires no programing knowledge 
- Specialized commercial or custom-made equipment is not required
- Extensive annotations are not required
- The pipeline is flexible and can be used to create and validate classifiers for different behaviors and environments
- Currently included behavioral classifiers have been validated in mice and rats
- SimBA is written on Windows/MacOS and compatible with Linux

**SimBA provides several validated classifer libraries using videos filmed from above at 90° angle with pose-estimation data from 8 body parts per animal; please see our [OSF repository](https://osf.io/tmu6y/) for access to all files. SimBA now accepts any user-defined pose-estimation annotation schemes with the inclusion of the [Flexible Annotation Module in v1.1](https://github.com/sgoldenlab/simba/blob/master/docs/Pose_config.md). SimBA now supports maDLC and SLEAP for similar looking animals with the release of [maDLC/SLEAP module in v1.2](/docs/Multi_animal_pose.md).** 

**Listserv for release information:** If you would like to receive notification for new releases of SimBA, please **[fill out this form](https://forms.gle/R47RWN4stNSJBj9D9)** and you will be added to the listserv.

#### Mouse
![](https://github.com/sgoldenlab/simba/blob/master/images/mouse_videos.gif)

#### Rat
![](https://github.com/sgoldenlab/simba/blob/master/images/rat_videos.gif)

#### SimBA GUI workflow
![](https://github.com/sgoldenlab/simba/blob/master/images/SimBA_tkinter_3.png)

## Pipeline 👷
![](https://github.com/sgoldenlab/simba/blob/master/images/overallflow.PNG)

## Resources 💾

- **Data, pose models & classifiers** — [OSF repository](https://osf.io/tmu6y/) 💾
- **Trained classifiers** — [Random forest models](https://osf.io/kwge8/) 🌲
- **Install / package** — [PyPI](https://pypi.org/project/Simba-UW-tf-dev/) 📦
- **API reference** — [SimBA on ReadTheDocs](https://simba-uw-tf-dev.readthedocs.io/en/latest/index.html) 📘
- **Example notebooks** — [Run SimBA from code](https://simba-uw-tf-dev.readthedocs.io/en/latest/notebooks.html) 📓
- **Docker images** — [Docker Hub](https://hub.docker.com/repositories/goldenlab) 🐳
- **Visualization examples** — [YouTube playlist](https://www.youtube.com/playlist?list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl) 📺
- **Labelled images & tracking weights** — [DeepLabCut annotations/weights (OSF)](https://osf.io/sr3ck/) 📷
- **Community & support** — [Gitter chat](https://gitter.im/SimBA-Resource/community) 💬
- **Bug reports & feature requests** — [GitHub Issues](https://github.com/sgoldenlab/simba/issues) 🐛
- **Golden Lab** — [Sam Golden Lab, UW](https://goldenneurolab.com/) 🧪
- **Download statistics** — [Live dashboard](https://sronilsson.github.io/download_stats/) 📊

## Developer & contact 👨‍💻

SimBA is developed and maintained by **[Simon Nilsson](https://github.com/sronilsson)** ([homepage](https://sronilsson.netlify.app/)). For questions, bug reports, or feature requests, reach out via [GitHub](https://github.com/sronilsson) or [open an issue](https://github.com/sgoldenlab/simba/issues).

## License 📃
This project is licensed under the BSD 3-Clause License, modified for academic and research use only (see [LICENSE](LICENSE)). Note that the software is provided 'as is', without warranty of any kind, express or implied. 

If you find **any** part of the code or data useful for your own work, please cite us. You can view and download the citation file by clicking the <kbd>Cite this repository</kbd> button at the top right of this page. Thank you 🙏!

	@article{Goodwin2024,
  		author = {Goodwin, Nastacia L. and Choong, Jia J. and Hwang, Sophia and Pitts, Kayla and Bloom, Liana and Islam, Aasiya and Zhang, Yizhe Y. and Szelenyi, Eric R. and Tong, Xiaoyu and Newman, Emily L. and Miczek, Klaus and Wright, 	Hayden R. and McLaughlin, Ryan J. and Norville, Zane C. and Eshel, Neir and Heshmati, Mitra and Nilsson, Simon R. O. and Golden, Sam A.},
  		title = {Simple Behavioral Analysis (SimBA) as a platform for explainable machine learning in behavioral neuroscience},
  		journal = {Nature Neuroscience},
  		volume = {27},
  		pages = {1411--1424},
  		year = {2024},
  		doi = {10.1038/s41593-024-01649-9},
  		publisher = {Nature Publishing Group},
  		URL = {https://simba-uw-tf-dev.readthedocs.io/},
  		repository = {https://github.com/sgoldenlab/simba},
    		license = {BSD-3-Clause (Academic/Research Use Only)}
			}
## References 📜

[![Foo](https://github.com/sgoldenlab/simba/blob/master/images/cos_center_logo_small.original.png)](https://osf.io/d69jt/)

## Contributors 🤼
- [Simon Nilsson](https://github.com/sronilsson)
- [Jia Jie Choong](https://github.com/inoejj)
- [Sophia Hwang](https://github.com/sophihwang26)

See the [full credits page](https://simba-uw-tf-dev.readthedocs.io/en/latest/credits.html) for everyone who has contributed to SimBA.
