# SimBA (Simple Behavioral Analysis)

[![pypi](https://img.shields.io/pypi/v/simba-uw-tf-dev.svg)]([https://pepy.tech/project/simba-uw-tf-dev](https://pypi.org/project/Simba-UW-tf-dev/))
[![py3.10 tests](https://github.com/sgoldenlab/simba/actions/workflows/tests_py310.yml/badge.svg)](https://github.com/sgoldenlab/simba/actions/workflows/tests_py310.yml)
[![py3.6 tests](https://github.com/sgoldenlab/simba/actions/workflows/tests_py36.yml/badge.svg)](https://github.com/sgoldenlab/simba/actions/workflows/tests_py36.yml)
![Docs](https://readthedocs.org/projects/simba-uw-tf-dev/badge/?version=latest&style=flat)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Gitter chat](https://badges.gitter.im/USER/REPO.png)](https://gitter.im/SimBA-Resource/community)
[![Download: Weights](https://img.shields.io/badge/Download-Weights-orange.svg)](https://osf.io/5t4y9/)
[![SimBA: listserv](https://img.shields.io/static/v1?label=SimBA&message=listserv&color=blue)](https://docs.google.com/forms/d/e/1FAIpQLSfjbjae0XqNcl7GYOxmqvRsCveG-cmf4p4hBNNJ8gu5vPLHng/viewform)
[![DOI](https://zenodo.org/badge/206670333.svg)](https://zenodo.org/badge/latestdoi/206670333)

[![Downloads](https://pepy.tech/badge/simba-uw-tf-dev/month)](https://pepy.tech/project/simba-uw-tf-dev)
[![Downloads](https://pepy.tech/badge/simba-uw-tf-dev)](https://pepy.tech/project/simba-uw-tf-dev)


**Pre-print: [Simple Behavioral Analysis (SimBA) – an open source toolkit for computer classification of complex social behaviors in experimental animals](https://www.biorxiv.org/content/10.1101/2020.04.19.049452v2)**

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/landing_1.gif" />
</p>


To install SimBA via pip, use the following command:
```bash
pip install simba-uw-tf-dev
```

## Apr-16-2023: API Reference

See SimBA on [readthedocs](https://simba-uw-tf-dev.readthedocs.io/) for [API reference](https://simba-uw-tf-dev.readthedocs.io/en/latest/api.html) and [example notebooks](https://simba-uw-tf-dev.readthedocs.io/en/latest/notebooks.html) when you prefer to run methods available in SimBA through the command line over the GUI. If you find that notebook examples are lacking, please reach out to us [Gitter](https://gitter.im/SimBA-Resource/community) or by opening an issue and let's chat about typing up further examples!

## Apr-04-2023: SimBA version 1.55 release

Although core methods remain, the SimBA code and documentation has sprawled significantly. If you are curious about the methods in the GUI, try clicking the header icons which should take you to the relevant documentation.

<p align = "center">
<img src="/images/doc_inst.png" alt=""
	title="SimBA" />
<p/>

As always, reach out to us on [Gitter chat](https://gitter.im/SimBA-Resource/community) or open an issue if you have questions, bug reports of feature requests! 

## Oct-07-2021: SimBA version 1.31 release

We have released a significantly improved GUI for region-of-interest segmentation and analysis.
[Click here to go to the new ROI documentation page.](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md)

### <p align="center"> Regions of Interest (ROIs) in SimBA </p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_new_1.png" />
</p>

The SimBA region of interest (ROI) interface allows users to define and 
draw ROIs on videos. ROI data can be used to calculate basic descriptive 
statistics based on animals movements and locations such as:

* How much time the animals have spent in different ROIs.
* How many times the animals have entered different ROIs.
* The distance animals have moved in the different ROIs.
* Calculate how animals have engaged in different classified behaviors in each ROI.
etc....

Furthermore, the ROI data can  be used to build potentially valuable, additional, features for random forest predictive classifiers. Such features can be used to generate a machine model that classify behaviors that depend on the spatial location of body parts in relation to the ROIs. **CAUTION**: If spatial locations are irrelevant for the behaviour being classified, then such features should *not* be included in the machine model generation as they just 
only introduce noise.

## Feb-08-2021: SimBA version 1.3 release

It has been nearly a year since the first public iteration of SimBA was released! We would like to thank the open-source community who have supported us and provided invaluable feedback and motivation to continue developing and supporting SimBA to where it is now. We have recently passed well over 150,000 downloads via pip install across all branches, and average between ~5000 to 10,000 weekly downloads alongside a gitter community of >100 users. We have just passed 15 citations for the SimBA preprint, which was released ~8 months ago. This would not be possible without your support. Thank you.

The newest release of SimBA, v1.3, provides a significant jump in features, quality of life improvements, and bug fixes. Several are highlighted below.

Please update using `pip install simba-uw-tf==1.3.7`, [this version](https://pypi.org/project/Simba-UW-tf/) has native deeplabcut and deepposekit GUI support disabled. Hence, tensorflow is not needed. Pose-estimation developers have created excellent GUIs for their pipelines, and we do a disservice to you by not supporting the most updated versions. SimBA now supports pose-estimation dataframe imports from Deeplabcut, DeepPoseKit, SLEAP, MARS and others. If you are developing a new pose-estimation method and would like it directly supported in SimBA, please let us know! 

### Selected New Features
- Easy install of SimBA via `pip` - [Documentation](/docs/installation.md)
- Install simba using anaconda - [Documentation](/docs/anaconda_installation.md)
- Introduction of SHAP for behavioral neuroscience classifier explainability and standarization- [Documentation](/docs/SHAP.md)
- Plotly integration for immediate data visualization - [Documentation](/docs/plotly_dash.md)
- Labelling/annotating behaviors with many third-party apps - [Documentation](/docs/third_party_annot.md)
- Kleinberg Filter for smoothing - [Documentation](/docs/kleinberg_filter.md)
- ROI Visualization update - [Documentation](/docs/ROI_tutorial.md)
- User define features extraction - [Documentation](/docs/extractFeatures.md)
- Quick line plot - Allow user to make line plots with selected bodypart and tracking data (located under **Tools**)
- Many, many, many, many bug-fixes


## June-12-2020: SimBA version 1.2 release
### New Features
- Multi-animal DLC support - [Documentation](/docs/Multi_animal_pose.md)
- Multi-animal SLEAP support - [Documentation](/docs/Multi_animal_pose.md)
- SimBA 'pseudo-labelling' module - [Documentation](/docs/pseudoLabel.md)
- Easy install of SimBA via `pip` - [Documentation](/docs/installation.md)
- Plenty of new quality-of-life features (e.g., time-bin analyzes / improved visualizations options) - [Documentation](/docs/Scenario1.md)
- Many, many, many, many bug-fixes

Please join our [Gitter chat](https://gitter.im/SimBA-Resource/community) if you have any questions, or even if you would simply like to discuss potential applications for SimBA in your work.  Please come by, stay inside, wash your hands, and check on your lab mates reguarly!

## April-25-2020: SimBA pre-print manuscript release

A pre-print SimBA manuscript on bioRxiv! The manuscript details the use of SimBA for generation of social predictive classifiers in rat and mouse resident-intruder protocols - please check it out using the link above. All data, pose-estimation models, and the final classifiers generated in the manuscript, can be accessed through our [OSF repository](https://osf.io/tmu6y/) and through the [Resource](https://github.com/sgoldenlab/simba#resource-) menu further down this page.

## March-05-2020: SimBA version 1.1 release
### New Features
- Region of Interest support (ROI Module) - [Documentation](/docs/ROI_tutorial.md)
- DeepPoseKit support (DPK Module) - [Documentation](/docs/DeepPoseKit_in_SimBA.md)
- SimBA accepts user-defined pose-configurations (Flexible Annotation Module) - [Documentation](/docs/Pose_config.md)
- Interactive classifer discrimination thresholding - [Documentation](/docs/validation_tutorial.md#validate-model-on-single-video)
- Individual discrimination thresholds for classifiers - [Documentation](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data)
- Heatmap visualizations -[Documentation](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions)
- Multi-crop tool - [Documentation](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#multi-crop-videos)
- Post-classification validation method for false-positives - [Documentation](/docs/classifier_validation.md#classifier-validation)
- Many, many, many bug-fixes

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

**Installation note:** SimBA can be installed either with TensorFlow compatability (for generating DeepLabCut, DeepPoseKit and SLEAP pose-estimation models), or without TensorFlow (for stand-alone use with classifiers and other functions). Please choose the appropriate branch for your needs, using pip install. More details are found in the [Installation Documentation](https://github.com/sgoldenlab/simba/blob/master/README.md#installation-%EF%B8%8F).

**Listserv for release information:** If you would like to receive notification for new releases of SimBA, please **[fill out this form](https://forms.gle/R47RWN4stNSJBj9D9)** and you will be added to the listserv.

#### Mouse
![](https://github.com/sgoldenlab/simba/blob/master/images/mouse_videos.gif)

#### Rat
![](https://github.com/sgoldenlab/simba/blob/master/images/rat_videos.gif)

#### SimBA GUI workflow
![](https://github.com/sgoldenlab/simba/blob/master/images/SimBA_tkinter_3.png)


## Pipeline 👷
![](https://github.com/sgoldenlab/simba/blob/master/images/overallflow.PNG)

## Documentation: General methods

### Step 1: [Pre-process videos](docs/tutorial_process_videos.md) 

### Step 2: [Create tracking model and generate pose-estimation data](docs/Tutorial_DLC.md) 

### Step 3: [Building classfier(s)](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior)

### Step 4: [Analysis/Visualization](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-9-analyze-machine-results)

### [Click here for the full *generic* tutorial on building classifiers in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md).

## Scenario tutorials

To faciliate the initial use of SimBA, we provide several use scenarios. We have created these scenarios around a hypothetical experiment that take a user from initial use (completely new start) all the way through analyzing a complete experiment and then adding additional experimental datasets to an initial project.

### Scenario 1: [Building classifiers from scratch](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md)

### Scenario 2: [Using a classifier on new experimental data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md)

### Scenario 3: [Updating a classifier with further annotated data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario3.md)

### Scenario 4: [Analyzing and adding new Experimental data to a previously started project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4_new.md)

## Installation ⚙️

- [Install SimBA](docs/installation.md)

- [Install SimBA using Anaconda](docs/anaconda_installation.md)

## Tutorial 📚
- [API](https://simba-uw-tf-dev.readthedocs.io/en/latest/index.html) 📘
- [Process video using SimBA tools](docs/Tutorial_tools.md) 🔨
- [Batch pre-process video using SimBA](docs/tutorial_process_videos.md) 🏭
- [Using DeepPoseKit in SimBA](docs/DeepPoseKit_in_SimBA.md) 📙
- [Using DeepLabCut through SimBA](docs/Tutorial_DLC.md) 📗
- [SimBA generic tutorial](docs/tutorial.md) 📘
- [SimBA friendly asked questions (FAQ)](docs/FAQ.md) 📕 
- [SimBA behavioral annotation interface](docs/label_behavior.md) 🏷️
- [SimBA Advanced behavioral annotation interface](docs/advanced_labelling.md) 🏷️
- [Using third-party annotation tools in SimBA](docs/third_party_annot.md) 🏷️
- [Using user-defined ROIs in SimBA](/docs/ROI_tutorial_new.md) 🗺️
- [Using multi-animal pose (maDLC/SLEAP/APT) in SimBA](/docs/Multi_animal_pose.md) 🐭🐭
- [Using the SimBA data analysis and export dashboard](docs/plotly_dash.md) 📊
- [Explainable machine classifications in SimBA (SHAP)](docs/SHAP.md) 🧮
- [Mutual exclusivity using heuristic rules in SimBA](docs/mutual_exclusivity_heuristic_rules.md) 📗
- [Compute feature subsets in SimBA](docs/feature_subsets.md) 📕 
- [Kleinberg markov chain classification smoothing in SimBA](docs/kleinberg_filter.md) 🔗
- [Cue-light analyses in SimBA](docs/cue_light_tutorial.md)💡💡
- [Bounding boxes in SimBA](docs/anchored_rois.md)📦
- [Reversing the directionality of classifiers in SimBA](docs/reverse_annotations.md) ⏪
- [Spike-time correlation coefficients in SimBA](docs/FSTTC.md) 📔
- [Spontaneous alternation in SimBA](/docs/spontaneous_alternation.md)🌽
- [Analysing animal directions in SimBA](docs/directionality_between_animals.md) 🧭
- [Recommended hardware](https://github.com/sgoldenlab/simba/blob/master/misc/system_requirements.md) 🖥️
- [Downloading compressed data from the SimBA OSF repository](https://github.com/sgoldenlab/simba/blob/master/docs/using_OSF.md) 💾

## Resource 💾

All data (classifiers etc.) is available on our [Open Science Framework repository](https://osf.io/tmu6y/). For a schematic overview of the data respository folder structure (as of March-20-2020), click [HERE](https://github.com/sgoldenlab/simba/blob/master/images/OSF_folder_structure_031820.jpg).

### API reference
- [SimBA on readthedocs](https://simba-uw-tf-dev.readthedocs.io/en/latest/index.html)

### Models
Below is a link to download trained  behavior classification models to apply it on your dataset
- [Random forest models](https://osf.io/kwge8/) 🌲

### Docker images
- [Docker hub](https://hub.docker.com/repositories/goldenlab)


### SimBA visualization examples
- [YouTube playlist](https://www.youtube.com/playlist?list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl) 📺

### Labelled images
- [Annotated images for tracking models](https://osf.io/sr3ck/) 📷

### Tracking weights
- [DeepLabCut tracking weights](https://osf.io/sr3ck/) 🏋️

### Golden Lab webpage
- [Sam Golden Lab UW](https://goldenneurolab.com/) 🧪🧫🐁



## License 📃
This project is licensed under the GPLv3. Note that the software is provided 'as is', without warranty of any kind, express or implied. 

If you find **any** part of the code or data useful for your own work, please cite us. You can view and download the citation file by clicking the <kbd>Cite this repository</kbd> button at the top right of this page. Thank you 🙏!

    @article{Nilsson2020.04.19.049452,
      author = {Nilsson, Simon RO and Goodwin, Nastacia L. and Choong, Jia Jie and Hwang, Sophia and Wright, Hayden R and Norville, Zane C and Tong, Xiaoyu and Lin, Dayu and Bentzley, Brandon S. and Eshel, Neir and McLaughlin, Ryan J and Golden, Sam A.},
      title = {Simple Behavioral Analysis (SimBA) {\textendash} an open source toolkit for computer classification of complex social behaviors in experimental animals},
      elocation-id = {2020.04.19.049452},
      year = {2020},
      doi = {10.1101/2020.04.19.049452},
      publisher = {Cold Spring Harbor Laboratory},
      URL = {https://www.biorxiv.org/content/early/2020/04/21/2020.04.19.049452},
      eprint = {https://www.biorxiv.org/content/early/2020/04/21/2020.04.19.049452.full.pdf},
      journal = {bioRxiv}
    }

## References 📜

[![Foo](https://github.com/sgoldenlab/simba/blob/master/images/cos_center_logo_small.original.png)](https://osf.io/d69jt/)

## Contributors 🤼
- [Simon Nilsson](https://github.com/sronilsson)
- [Jia Jie Choong](https://github.com/inoejj)
- [Sophia Hwang](https://github.com/sophihwang26)

#
Author [Simon N](https://github.com/sronilsson), [JJ Choong](https://github.com/inoejj)
