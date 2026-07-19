Video processing tools
======================

.. contents:: On this page
   :local:
   :depth: 1

Video processing tools

----------------------------------------------



.. automodule:: simba.video_processors.video_processing
   :noindex:
   :members:
   :undoc-members:



Multi-cropper

---------------------------------------------



.. autoclass:: simba.video_processors.multi_cropper.MultiCropper
   :members:
   :undoc-members:



Crop ROI selector (rectangles)

------------------------------------------------



.. autoclass:: simba.video_processors.roi_selector.ROISelector
   :members:
   :undoc-members:





Crop ROI selector (circles)

------------------------------------------------



.. autoclass:: simba.video_processors.roi_selector_circle.ROISelectorCircle
   :members:
   :undoc-members:





Crop ROI selector (polygons)

------------------------------------------------



.. autoclass:: simba.video_processors.roi_selector_polygon.ROISelectorPolygon
   :members:
   :undoc-members:





Interactive CLAHE

------------------------------------------------



.. automodule:: simba.video_processors.clahe_ui
   :noindex:
   :members:
   :show-inheritance:



Interactive brightness / contrast

------------------------------------------------



.. automodule:: simba.video_processors.brightness_contrast_ui
   :noindex:
   :members:
   :show-inheritance:





Batch video process executor

-----------------------------------------------------------------------



.. automodule:: simba.video_processors.batch_process_create_ffmpeg_commands
   :noindex:
   :members:
   :show-inheritance:



Egocentrically rotate videos

-----------------------------------------------------------------------



.. automodule:: simba.video_processors.egocentric_video_rotator
   :members:
   :show-inheritance:



Egocentrically rotate videos - GPU (NVDEC/NVENC)

-----------------------------------------------------------------------



.. autoclass:: simba.data_processors.cuda.egocentric_rotator_nvenc.EgocentricRotatorNVENC
   :members:
   :show-inheritance:



Greyscale videos - GPU (NVDEC/NVENC)

-----------------------------------------------------------------------



.. autoclass:: simba.data_processors.cuda.greyscale_nvenc.GreyscaleNVENC
   :members:
   :show-inheritance:



CLAHE videos - GPU (NVDEC/NVENC)

-----------------------------------------------------------------------



.. autoclass:: simba.data_processors.cuda.clahe_nvenc.ClaheNVENC
   :members:
   :show-inheritance:



Asynchronous frame reader

------------------------------------------------



.. automodule:: simba.video_processors.async_frame_reader
   :members:
   :show-inheritance:



Asynchronous frame reader (GPU)

------------------------------------------------



.. automodule:: simba.video_processors.async_frame_reader_gpu
   :noindex:
   :members:
   :show-inheritance:



Save videos as frame images

------------------------------------------------



.. automodule:: simba.video_processors.videos_to_frames
   :noindex:
   :members:
   :show-inheritance:

