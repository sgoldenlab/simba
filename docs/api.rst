📖 API Reference
================

.. raw:: html

   <div style="text-align:center; margin:4px 0 20px;">
     <video autoplay loop muted playsinline preload="metadata"
            poster="_static/img/mouse_run_simba_hat_poster.jpg"
            style="width:min(440px,82%); height:auto;"
            aria-label="Animated SimBA lab mouse in a top hat">
       <source src="_static/img/mouse_run_simba_hat.webm" type="video/webm">
     </video>
   </div>

This section provides a categorized reference of SimBA's modules and methods, grouped by their functionality such as feature extraction, plotting, transformation, and modeling.

.. admonition:: 📑 Find anything fast
   :class: tip

   **Browse the** :ref:`Module Index <modindex>` — an alphabetical list of every SimBA module, and the quickest way to find where a class or function lives.

   Or jump to the :ref:`General Index <genindex>` (every class, function and term, A–Z), or use :ref:`full-text search <search>`.

.. contents::
   :local:
   :depth: 2


|:mag:| **Blob tracking tools**
--------------------------------------------------

Track animals in videos using background subtraction and blob detection — an alternative to pose-estimation that needs no trained keypoint model. Detected blob contours are converted into geometric features such as the animal's center, nose, tail, and left/right flank points, so downstream movement, ROI, and geometry analyses work from the silhouette alone. Well suited to single-animal, high-contrast recordings where a full pose model is unnecessary.

.. toctree::
   :maxdepth: 5

   simba.blob


|:package:| **Bounding-box tools**
--------------------------------------------------

Build bounding boxes and anchored polygons around animals from pose-estimation data, then quantify how they overlap through time. Useful for detecting proximity-based social interactions — one animal entering another's space, or body parts coming into contact — without training a dedicated classifier. Overlap statistics can be aggregated per-animal and per-frame for downstream analysis.

See tutorial: `Anchored ROI (bounding box) tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md>`_

.. toctree::
   :maxdepth: 2

   simba.bounding_box_tools
   simba.yolo


|:repeat:| **Circular transformations**
--------------------------------------------------

Statistical operations for circular data such as head direction and heading angle, where values wrap around 360°. Wraparound-aware and multi-animal capable, these methods derive base angles from body-part triplets and compute quantities like mean resultant vector length, angular difference, directional dispersion, and the Rayleigh test — in both sliding-window and whole-session forms. GPU-accelerated variants are available for large datasets.

.. toctree::
   :maxdepth: 4

   simba.circular_statistics


|:wrench:| **Config reader**
--------------------------------------------------

Parse SimBA project configuration (``project_config.ini``) and expose project metadata — file paths, animal and body-part definitions, video parameters, and classifier settings. Most SimBA classes inherit from this reader so they can locate data and resolve project-wide options consistently, rather than re-parsing the config themselves.

.. toctree::
   :maxdepth: 2

   simba.config_reader


|:bulb:| **Cue-light tools**
--------------------------------------------------

Link animal behavior to the on/off state of cue lights in the arena. Detect illumination changes from ROI pixel intensity, then summarize movement and classified behavior in the periods before, during, and after each light event — for analyses of conditioned or stimulus-driven responses.

See tutorial: `Cue-light tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`_

.. toctree::
   :maxdepth: 2

   simba.cue_light_tools


|:wrench:| **Data processing tools**
--------------------------------------------------

Transform classification, tracking, and image data after feature extraction or model inference. Includes interpolation and smoothing of pose data, aggregation of classifier results, and specialized calculators such as Kleinberg burst detection, the forward spike-time tiling coefficient (FSTTC), and movement and severity scoring.

.. toctree::
   :maxdepth: 1

   simba.data_processors


|:straight_ruler:| **Feature extraction mixins**
--------------------------------------------------

Core low-level feature methods — distances, angles, velocities, areas, and body-part relationships — that the default extraction pipelines assemble into full feature sets. Use these directly to compose custom features from raw tracking coordinates when the out-of-the-box extractors don't fit your schema.

.. toctree::
   :maxdepth: 4

   simba.feature_extraction_mixins


|:pencil:| **Feature extraction wrappers**
--------------------------------------------------

Pre-configured "out-of-the-box" feature extraction modules for common pose-estimation schemas, spanning a range of single- and multi-animal body-part layouts. Each wrapper turns raw tracking coordinates into the tabular feature set that SimBA classifiers are trained and run on, so most users never have to write feature code by hand.

.. toctree::
   :maxdepth: 1

   simba.feature_extractors


|:triangular_ruler:| **Geometry transformations**
--------------------------------------------------

Transform pose-estimated body-part coordinates into geometric shapes — bounding boxes, polygons, circles, and lines — and compute spatial relationships between them. Supports buffering, unions and intersections, point-in-shape tests, inter-shape distances, overlap, and directionality. Built on Shapely, with GPU-accelerated variants for many of the heavier operations.

.. toctree::
   :maxdepth: 4

   simba.geometry_mixin


|:gear:| **Global configuration**
--------------------------------------------------

Global SimBA settings and environment variables — GPU toggles, default paths, and feature flags — that tune runtime behaviour across the package. Set these once to change how modules execute (for example, enabling numba eager compilation or GPU code paths) without editing individual scripts.

.. toctree::
   :maxdepth: 1

   simba.env


|:zap:| **GPU acceleration**
--------------------------------------------------

CUDA/CuPy-accelerated versions of SimBA's compute-heavy routines — geometry, image, background subtraction, statistics, circular statistics, time-series, and SHAP — consolidated in one place. On a supported NVIDIA GPU they produce the same results as their CPU counterparts but run orders of magnitude faster on large datasets, with per-function expected-runtime tables to help you gauge the speed-up.

.. video:: _static/img/gpu_spin.webm
   :width: 700
   :autoplay:
   :nocontrols:
   :loop:
   :muted:
   :align: center

.. toctree::
   :maxdepth: 2

   simba.gpu_helpers


|:frame_with_picture:| **Image transformations**
--------------------------------------------------

Manipulate video frames and extract visual information from tracking data — crop, slice, rotate, and greyscale frames, isolate regions around body parts, and compute pixel-based features. Includes methods for comparing image features across time (e.g. frame-to-frame change), with both CPU and GPU implementations.

.. toctree::
   :maxdepth: 5

   simba.image_transformations


|:label:| **Labeling tools**
--------------------------------------------------

Interactive tools for annotating behavioral events frame by frame. Create and edit the ground-truth labels used to train classifiers, review and correct model predictions directly on the video, and manage annotations across a project's videos.

.. toctree::
   :maxdepth: 2

   simba.labelling


|:jigsaw:| **Mixins (overview)**
--------------------------------------------------

Entry point to SimBA's mixin classes — the reusable method libraries (feature, geometry, statistics, plotting, image, network, timeseries, model) that the other topical sections draw on. Grouping shared functionality here keeps it consistent and testable across the package; most user-facing tools are thin wrappers that combine these mixins.

.. toctree::
   :maxdepth: 1

   simba.mixins


|:robot_face:| **Model tools**
--------------------------------------------------

Create, train, evaluate, and apply SimBA's behavior classifiers (random-forest based). Configure hyperparameters, run cross-validation, generate evaluation metrics and plots such as feature importances and learning curves, and batch-apply trained models to new videos to produce per-frame behavior probabilities.

.. video:: _static/img/forest_mouse_loop.mp4
   :width: 700
   :autoplay:
   :nocontrols:
   :loop:
   :muted:
   :align: center

.. toctree::
   :maxdepth: 4

   simba.model_mixin


|:link:| **Network transformations**
--------------------------------------------------

Build and analyze graphs derived from pose-estimation time-series — for example, interaction networks between animals or connectivity between body parts. Compute node and edge metrics and visualize how the network's structure changes over time.

.. toctree::
   :maxdepth: 5

   simba.networks


|:warning:| **Outlier correction**
--------------------------------------------------

Heuristic filtering of implausible body-part tracking. Location- and movement-based criteria — scaled by a reference inter-body-part distance — flag and correct jitter and sudden, physically impossible jumps before feature extraction, improving the quality of downstream features and classification.

.. toctree::
   :maxdepth: 5

   simba.outlier_tools


|:art:| **Plotting and visualization tools**
--------------------------------------------------

Visualize behavioral data and pose-tracking outputs — Gantt plots, path, distance and velocity plots, heatmaps, classification-probability charts, and data overlays rendered onto the original video. Produces both static figures and frame-by-frame videos, with multiprocessing support for large batches.

.. video:: _static/img/simba_monitor.webm
   :width: 700
   :autoplay:
   :nocontrols:
   :loop:
   :muted:
   :align: center

.. toctree::
   :maxdepth: 5

   simba.plotting


|:package:| **Pose-estimation import tools**
--------------------------------------------------

Parse, load, and standardize pose-estimation output from common trackers (e.g. DeepLabCut, SLEAP, DeepPoseKit, MARS) across H5, CSV, and JSON formats. Handles single- and multi-animal data and can interpolate and smooth tracking on import, producing the consistent internal format the rest of SimBA expects.

.. toctree::
   :maxdepth: 1

   simba.pose_importers


|:world_map:| **ROI tools**
--------------------------------------------------

Define regions-of-interest — rectangles, circles, and polygons — and analyze tracking in relation to them. Compute zone entries and exits, time and distance spent in each region, distance to ROI boundaries, and whether animals are oriented toward a region, with visualization tools for the results.

.. video:: _static/img/arena_move.mp4
   :width: 700
   :autoplay:
   :nocontrols:
   :loop:
   :muted:
   :align: center

.. toctree::
   :maxdepth: 3

   simba.roi_tools


|:bar_chart:| **Statistics transformations**
--------------------------------------------------

Compute statistical features from tracking and classification data — descriptive statistics, distribution comparisons, distance metrics, and drift/change detection — over sliding or static time windows. Underpins much of SimBA's feature extraction, and many methods have GPU-accelerated counterparts for large datasets.

.. video:: _static/img/abacus_spin.mp4
   :width: 700
   :autoplay:
   :nocontrols:
   :loop:
   :muted:
   :align: center

.. toctree::
   :maxdepth: 4

   simba.statistics_mixin


|:inbox_tray:| **Third-party label appenders**
--------------------------------------------------

Import behavioral annotations produced by external tools (e.g. BORIS, Ethovision, Observer, DeepEthogram, SOLOMON) and align them, frame by frame, to SimBA's pose-estimation and feature data. Lets you reuse existing human or third-party labels as classifier training targets without re-annotating.

.. toctree::
   :maxdepth: 1

   simba.third_party_label_appenders


|:clock1:| **Time-series transformations**
--------------------------------------------------

Analyze the temporal structure of tracking and feature signals using sliding-window methods — complexity and entropy measures, autocorrelation, and windowed descriptive statistics — to capture how behavior evolves over time. GPU-accelerated variants are available for the heavier computations.

.. video:: _static/img/whiteboard_simba.webm
   :width: 700
   :autoplay:
   :nocontrols:
   :loop:
   :muted:
   :align: center

.. toctree::
   :maxdepth: 5

   simba.timeseries


|:crystal_ball:| **Unsupervised learning**
--------------------------------------------------

Discover behavioral structure without labels. Combines dimensionality reduction (e.g. UMAP, PCA) with clustering (e.g. HDBSCAN, K-means), plus tools to embed, visualize, and quantify the resulting clusters — a workflow for surfacing candidate behavioral motifs from feature data.

.. toctree::
   :maxdepth: 2

   simba.unsupervised


|:desktop_computer:| **User Interface (UI) tools**
--------------------------------------------------

SimBA's graphical interface components — the Tkinter windows, pop-ups, and menus that drive project creation, configuration, and each step of the analysis workflow. The entry point for users who prefer clicking through SimBA rather than scripting it.

.. toctree::
   :maxdepth: 1

   simba.ui


|:gear:| **Utilities**
--------------------------------------------------

Helper methods used throughout the package — logging, CLI execution, input and argument validation, warnings, timing, and reading/writing project files. The shared plumbing that keeps SimBA's error messages, checks, and file handling consistent across every module.

.. toctree::
   :maxdepth: 1

   simba.utils


|:video_camera:| **Video processing tools**
--------------------------------------------------

Video processing tools built on OpenCV and FFmpeg — clip, crop, concatenate, downsample, re-encode, rotate, and greyscale videos, and compute average/background frames. Covers the common pre-processing steps needed to get raw recordings ready for tracking and analysis, with multiprocessing support for batch jobs.

.. video:: _static/img/betamax_simba_reveal.webm
   :width: 500
   :autoplay:
   :nocontrols:
   :loop:
   :muted:
   :align: center

.. toctree::
   :maxdepth: 5

   simba.video_processing


👁️ **YOLO Methods**
-----------------------

Methods for training YOLO models, creating training and validation datasets, and converting behavioral neuroscience-specific datasets to YOLO datasets.

Uses the `Ultralytics <https://github.com/ultralytics/ultralytics>`_ package.

.. toctree::
   :maxdepth: 2

   simba.yolo
