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

SimBA also has whole-video GPU tools (NVDEC decode -> CUDA -> NVENC encode) for background
subtraction, greyscale, CLAHE, egocentric rotation, and pose, geometry and YOLO pose overlays;
see `GPU video tools (NVDEC/NVENC)`_.

This page is a single consolidated view of every GPU-accelerated module. Each block
is also documented in its topical section (e.g. Geometry, Statistics); the canonical,
cross-referenceable entries live there, so the listings here mirror them for a
one-stop GPU reference.

.. contents::
   :local:
   :depth: 2


Geometry (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.geometry
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Image (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.image
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Statistics (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.statistics
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Circular statistics (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.circular_statistics
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Time-series statistics (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.timeseries
   :members:
   :undoc-members:
   :show-inheritance:


Feature extraction (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.feature_extraction
   :members:
   :undoc-members:
   :show-inheritance:


Feature extraction supplement (GPU)
------------------------------------
.. automodule:: simba.data_processors.cuda.feature_extraction_supplement
   :members:
   :undoc-members:
   :show-inheritance:


SHAP explanations (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.create_shap_log
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Data transformations (GPU)
------------------------------
.. automodule:: simba.data_processors.cuda.data
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


GPU video tools (NVDEC/NVENC)
------------------------------
Whole-video tools that run every step on the GPU: frames are decoded by the NVDEC engine straight into GPU
memory, processed by a CUDA kernel, and re-encoded by the NVENC engine, so the video never passes through
the CPU. Each class has a CPU counterpart (linked from its documentation) and a runtime table.

.. note::
   These tools need an NVIDIA GPU and `PyNvVideoCodec <https://pypi.org/project/pynvvideocodec/>`_
   (``pip install pynvvideocodec==2.1.0``, the version SimBA is tested with; also in ``requirements_gpu.txt``).
   Background subtraction also runs without it, on a slower ffmpeg-decode + CUDA + cv2-encode path; the
   others raise an error. The GPU functions in the other sections of this page need only CuPy / numba.


Background subtraction (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: simba.video_processors.bg_subtractor_nvenc
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Pose plotting (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: simba.plotting.pose_plotter_nvenc
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Greyscale conversion (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: simba.video_processors.greyscale_nvenc
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


CLAHE (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: simba.video_processors.clahe_nvenc
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Egocentric rotation (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: simba.video_processors.egocentric_rotator_nvenc
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


Geometry overlay (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: simba.plotting.geometry_plotter_nvenc
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


YOLO pose plotting (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: simba.plotting.yolo_pose_visualizer_nvenc
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:


GPU utilities
------------------------------
Low-level CUDA device primitives (``@cuda.jit(device=True)``) that the other GPU
kernels compose, plus the shared NVDEC/NVENC helpers (decoder and encoder factories,
decodability checks, frame timing and muxing) that the GPU video tools are built on.

.. automodule:: simba.data_processors.cuda.utils
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:
