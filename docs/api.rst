API Reference
=============

Geometry transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods to perform geometry transformation of pose-estimation data. This includes creating bounding boxes, line objects, polygons, circles etc. from pose-estimated body-parts
and recording environment and computing metric representations of the relationships between created shapes or their attributes (sizes, distances, overlaps, intersection, etc.)

.. toctree::
   :maxdepth: 4

   simba.geometry_mixin


Circular transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike linear data, circular data wrap around in a circular or periodic manner such as two measurements of e.g., 360 vs. 1 are more similar than two measurements of 1 vs. 3. Thus, the
minimum and maximum values are connected, forming a closed loop, and we therefore need specialized statistical methods.These methods have support for multiple animals and base radial directions derived from two or three body-parts.

.. toctree::
   :maxdepth: 4

   simba.circular_statistics

Statistics transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Statistics methods used for feature extraction, drift assessment, distance computations, distribution comparisons in sliding and static windows.

.. toctree::
   :maxdepth: 4

   simba.statistics_mixin


Image transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods to slice and compute attributes of images from tracking data and comparing those image attributes across sequential images.

.. toctree::
   :maxdepth: 5

   simba.image_transformations


Time-series transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Time-series methods focused on signal complexity in sliding windows.

.. toctree::
   :maxdepth: 5

   simba.timeseries


Network transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for creating and analyze time-dependent graphs from pose-estimation data.

.. toctree::
   :maxdepth: 5

   simba.networks

Feature extraction methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature extraction methods used by "default" feature extraction wrappers. These methods are used by the "out-of-the-box" pre-defined pose-estimation schemas and compute - mainly - movement, distances and size features.

.. toctree::
   :maxdepth: 4

   simba.feature_extraction_mixins


Video processing tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for processing videos (e.g., pre- or processing functions that typically depend on ffmpeg or opencv).

.. toctree::
   :maxdepth: 5

   simba.video_processing


Plotting / visualization tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for visualizing behavior, tracking, and transformed data based on animal tracking.

.. toctree::
   :maxdepth: 5

   simba.plotting

Outlier correction tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for removing outliers based on heuristic rules of movement and location of pose-estimated body-parts

.. toctree::
   :maxdepth: 5

   simba.outlier_tools


Data processing tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for manipulating and transforming classification data, image data, and animal tracking data

.. toctree::
   :maxdepth: 1

   simba.data_processors


Unsupervised methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for unsupervised learning in SimBA

.. toctree::
   :maxdepth: 2

   simba.unsupervised

Labelling tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SimBA methods and interfaces for annotating images for behavioral states.

.. toctree::
   :maxdepth: 2

   simba.labelling

Third-party label append
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SimBA methods for appending annotation data from dedicated annotation software to pose-estimation data.

.. toctree::
   :maxdepth: 1

   simba.third_party_label_appenders

SimBA utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Helper functions for lookups, terminal printing, argument validity checks, reading and writing to disk, warning / error handling, and helpers for executing functions through CLI

.. toctree::
   :maxdepth: 1

   simba.utils

Region-of-interest (ROI) tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for drawing, defining, and analyzing tracking data in relation to ROIs

.. toctree::
   :maxdepth: 3

   simba.roi_tools

Default feature extraction wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"Standard" feature extraction methods for input pose-estimation data. These methods are used by the "out-of-the-box" pre-defined pose-estimation schemas
and compute - mainly - movement, distances and size features.

.. toctree::
   :maxdepth: 1

   simba.feature_extractors


Pose-estimation tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods to import and manipulate pose-estimation data

.. toctree::
   :maxdepth: 1

   simba.pose_importers


UI tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for SimBA graphical interfaces

.. toctree::
   :maxdepth: 1

   simba.ui


Bounding-box tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for analyzing animal interactions through `overlapping bounding boxes <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`_.

.. toctree::
   :maxdepth: 1

   simba.bounding_box_tools



Model tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for creating machine learning behavior classifiers

.. automodule:: simba.mixins.train_model_mixin
   :members:
   :show-inheritance:


Config reader methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for reading SimBA configparser.Configparser project config and associated project data

.. automodule:: simba.mixins.config_reader
   :members:
   :show-inheritance:


Cue-light tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for analyzing animal behaviours associated with `cue-light states through <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`_.

.. toctree::
   :maxdepth: 2

   simba.cue_light_tools

