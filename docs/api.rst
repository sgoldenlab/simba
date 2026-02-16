📖 API Reference
=============

This section provides a categorized reference of SimBA's modules and methods, grouped by their functionality such as feature extraction, plotting, transformation, and modeling.

.. contents::
   :local:
   :depth: 2


|:mag:| **Blob tracking tools**
--------------------------------------------------

Track animals in videos using background subtraction and blob detection. Extract geometric features (center, nose, tail, left/right points) from detected blob shapes without requiring pose-estimation data.

.. toctree::
   :maxdepth: 5

   simba.blob


|:package:| **Bounding-box tools**
--------------------------------------------------

Detect animal interactions via overlapping bounding boxes.

See tutorial: `Cue-light tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`_

.. toctree::
   :maxdepth: 2

   simba.bounding_box_tools
   simba.yolo


|:repeat:| **Circular transformations**
--------------------------------------------------

Statistical operations for circular data like head direction. Wraparound-aware, multi-animal capable, and based on body-part derived base angles.

.. toctree::
   :maxdepth: 4

   simba.circular_statistics


|:wrench:| **Config reader**
--------------------------------------------------

Parse SimBA config files and access project-specific metadata.

.. toctree::
   :maxdepth: 2

   simba.config_reader


|:bulb:| **Cue-light tools**
--------------------------------------------------

Link animal behavior to cue-light on/off states.

See tutorial: `Cue-light tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`_

.. toctree::
   :maxdepth: 2

   simba.cue_light_tools


|:wrench:| **Data processing tools**
--------------------------------------------------

Transform classification, tracking, and image data.

.. toctree::
   :maxdepth: 1

   simba.data_processors


|:straight_ruler:| **Feature extraction mixins**
--------------------------------------------------

Core low-level feature methods used in SimBA's default extraction pipelines.

.. toctree::
   :maxdepth: 4

   simba.feature_extraction_mixins


|:pencil:| **Feature extraction wrappers**
--------------------------------------------------

Pre-configured "out-of-the-box" feature extraction modules for common pose-estimation schemas.

.. toctree::
   :maxdepth: 1

   simba.feature_extractors


|:triangular_ruler:| **Geometry transformations**
--------------------------------------------------

Transform pose-estimated body-part coordinates into geometric shapes (bounding boxes, polygons, circles), and compute spatial relationships like distance and intersection.

.. toctree::
   :maxdepth: 4

   simba.geometry_mixin


|:frame_with_picture:| **Image transformations**
--------------------------------------------------

Slice frames and extract visual information from tracking data; compare image features across time.

.. toctree::
   :maxdepth: 5

   simba.image_transformations


|:label:| **Labeling tools**
--------------------------------------------------

SimBA tools for annotating behavioral events.

.. toctree::
   :maxdepth: 2

   simba.labelling


|:robot_face:| **Model tools**
--------------------------------------------------

Create, train, and manage behavior classifiers in SimBA.

.. toctree::
   :maxdepth: 4

   simba.model_mixin


|:link:| **Network transformations**
--------------------------------------------------

Build and analyze graphs derived from pose-estimation time-series data.

.. toctree::
   :maxdepth: 5

   simba.networks


|:warning:| **Outlier correction**
--------------------------------------------------

Heuristic-based filtering of body-part tracking outliers.

.. toctree::
   :maxdepth: 5

   simba.outlier_tools


|:art:| **Plotting and visualization tools**
--------------------------------------------------

Visualize behavioral data and pose-tracking outputs.

.. toctree::
   :maxdepth: 5

   simba.plotting


|:package:| **Pose-estimation import tools**
--------------------------------------------------

Parse, load, and process pose-estimation data from common formats.

.. toctree::
   :maxdepth: 1

   simba.pose_importers


|:world_map:| **ROI tools**
--------------------------------------------------

Define and analyze regions-of-interest (ROIs) in relation to tracking data.

.. toctree::
   :maxdepth: 3

   simba.roi_tools


|:bar_chart:| **Statistics transformations**
--------------------------------------------------

Compute statistical features, drift, distances, and distribution comparisons in sliding or static time windows.

.. toctree::
   :maxdepth: 4

   simba.statistics_mixin


|:inbox_tray:| **Third-party label appenders**
--------------------------------------------------

Append labels from external annotation tools to pose-estimation outputs.

.. toctree::
   :maxdepth: 1

   simba.third_party_label_appenders


|:clock1:| **Time-series transformations**
--------------------------------------------------

Analyze time-series complexity using sliding window methods.

.. toctree::
   :maxdepth: 5

   simba.timeseries


|:crystal_ball:| **Unsupervised learning**
--------------------------------------------------

Clustering and dimensionality reduction methods for behavioral analysis.

.. toctree::
   :maxdepth: 2

   simba.unsupervised


|:desktop_computer:| **User Interface (UI) tools**
--------------------------------------------------

SimBA's GUI components and window-based interaction logic.

.. toctree::
   :maxdepth: 1

   simba.ui


|:gear:| **Utilities**
--------------------------------------------------

Helper methods for logging, CLI execution, argument checks, warnings, and I/O.

.. toctree::
   :maxdepth: 1

   simba.utils


|:video_camera:| **Video processing tools**
--------------------------------------------------

Video processing tools using OpenCV and FFmpeg.

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
