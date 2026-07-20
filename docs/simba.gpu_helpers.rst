GPU acceleration
==============================

.. video:: _static/img/gpu_spin.webm
   :width: 700
   :autoplay:
   :nocontrols:
   :loop:
   :muted:
   :align: center

SimBA ships CUDA/CuPy-accelerated implementations of many of its most compute-heavy
routines — geometry, image, statistics, circular-statistics, time-series and SHAP
operations. On a supported NVIDIA GPU these run the same analyses as their CPU
counterparts, but orders of magnitude faster on large datasets.

This page is a single consolidated view of every GPU-accelerated module. Each block
is also documented in its topical section (e.g. Geometry, Statistics); the canonical,
cross-referenceable entries live there, so the listings here mirror them for a
one-stop GPU reference.

.. contents::
   :local:
   :depth: 1


Geometry (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.geometry
   :noindex:
   :members:
   :show-inheritance:


Image (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.image
   :noindex:
   :members:
   :show-inheritance:


Background subtraction (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.bg_subtractor
   :noindex:
   :members:
   :show-inheritance:


Pose plotting (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.pose_plotter_nvenc
   :noindex:
   :members:
   :show-inheritance:


Greyscale conversion (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.greyscale_nvenc
   :noindex:
   :members:
   :show-inheritance:


CLAHE (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.clahe_nvenc
   :noindex:
   :members:
   :show-inheritance:


Egocentric rotation (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.egocentric_rotator_nvenc
   :noindex:
   :members:
   :show-inheritance:


Geometry overlay (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.geometry_plotter_nvenc
   :noindex:
   :members:
   :show-inheritance:


Statistics (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.statistics
   :noindex:
   :members:
   :show-inheritance:


Circular statistics (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.circular_statistics
   :noindex:
   :members:
   :show-inheritance:


Time-series statistics (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.timeseries
   :members:
   :show-inheritance:


SHAP explanations (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.create_shap_log
   :noindex:
   :members:
   :show-inheritance:


Data transformations (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.data
   :noindex:
   :members:
   :show-inheritance:


GPU utilities
------------------------------
Low-level CUDA device primitives (``@cuda.jit(device=True)``) that the other GPU
kernels compose, plus the public NVDEC video-decoder factory.

.. automodule:: simba.data_processors.cuda.utils
   :members:
   :private-members:
   :show-inheritance:
