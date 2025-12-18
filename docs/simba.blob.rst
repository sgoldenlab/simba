SimBA Blob Tracking Methods
===============================

This section documents SimBA's blob tracking functionality, which uses background subtraction and blob detection to track animals in videos without requiring pose-estimation data.

Main Executor
-----------------------------------------------------------------------

The main class for executing blob tracking workflows.

.. autoclass:: simba.video_processors.blob_tracking_executor.BlobTrackingExecutor
   :members:
   :show-inheritance:

User Interfaces
-----------------------------------------------------------------------

GUI tools for configuring and running blob tracking.

.. autoclass:: simba.ui.blob_tracker_ui.BlobTrackingUI
   :members:
   :show-inheritance:

.. autoclass:: simba.ui.blob_quick_check_interface.BlobQuickChecker
   :members:
   :show-inheritance:

Blob Detection Functions
-----------------------------------------------------------------------

Core functions for detecting and extracting blob vertices from videos.

.. autofunction:: simba.data_processors.find_animal_blob_location.get_blob_vertices_from_video

.. autofunction:: simba.data_processors.find_animal_blob_location.get_nose_tail_from_vertices

.. autofunction:: simba.data_processors.find_animal_blob_location.get_left_right_points

.. autofunction:: simba.data_processors.find_animal_blob_location.stabilize_body_parts

Background Subtraction: CPU Multiprocessing
-----------------------------------------------------------------------

CPU-based background subtraction using multiprocessing for parallel frame processing.

.. autofunction:: simba.video_processors.video_processing.video_bg_subtraction_mp

Background Subtraction: GPU (CuPy)
-----------------------------------------------------------------------

GPU-accelerated background subtraction using CuPy for faster processing on NVIDIA GPUs.

.. autofunction:: simba.data_processors.cuda.image.bg_subtraction_cupy

Background Subtraction: GPU (CUDA)
-----------------------------------------------------------------------

GPU-accelerated background subtraction using CUDA for faster processing on NVIDIA GPUs.

.. autofunction:: simba.data_processors.cuda.image.bg_subtraction_cuda


Compute Average Frame
-----------------------------------------------------------------------

Functions for computing average frames from videos, used as background references for blob tracking.

CPU-based average frame computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: simba.video_processors.video_processing.create_average_frm

GPU-based average frame computation (CuPy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: simba.data_processors.cuda.image.create_average_frm_cupy

GPU-based average frame computation (CUDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: simba.data_processors.cuda.image.create_average_frm_cuda


Data Import
-----------------------------------------------------------------------

Import blob tracking data into SimBA projects.

.. autoclass:: simba.pose_importers.simba_blob_importer.SimBABlobImporter
   :members:
   :show-inheritance:

Visualization
-----------------------------------------------------------------------

Tools for visualizing blob tracking results.

.. autoclass:: simba.plotting.blob_plotter.BlobPlotter
   :members:
   :show-inheritance:

.. autoclass:: simba.plotting.blob_visualizer.BlobVisualizer
   :members:
   :show-inheritance:

.. autoclass:: simba.plotting.geometry_plotter.GeometryPlotter
   :members:
   :show-inheritance:
