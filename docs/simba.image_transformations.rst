Image transformations
=====================

.. contents:: On this page
   :local:
   :depth: 1

Image mixin
----------------------------------------------

.. autoclass:: simba.mixins.image_mixin.ImageMixin
   :members:
   :undoc-members:
   :inherited-members:


Image GPU methods
----------------------------------------------

.. automodule:: simba.data_processors.cuda.image
   :members:
   :undoc-members:
   :show-inheritance:


Egocentric alignment
----------------------------------------------

Rotate and translate frames (and their pose data) into an egocentric reference frame, so a chosen body-part anchor is centred and a body axis points in a fixed direction across the whole video.

.. autoclass:: simba.data_processors.egocentric_aligner.EgocentricalAligner
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: simba.video_processors.egocentric_video_rotator.EgocentricVideoRotator
   :members:
   :undoc-members:
   :noindex:


CLAHE contrast enhancement (GPU)
----------------------------------------------

.. autoclass:: simba.data_processors.cuda.clahe_nvenc.ClaheNVENC
   :members:
   :undoc-members:
   :noindex:


Greyscale conversion (GPU)
----------------------------------------------

.. autoclass:: simba.data_processors.cuda.greyscale_nvenc.GreyscaleNVENC
   :members:
   :undoc-members:
   :noindex:


Egocentric rotation (GPU)
----------------------------------------------

.. autoclass:: simba.data_processors.cuda.egocentric_rotator_nvenc.EgocentricRotatorNVENC
   :members:
   :undoc-members:
   :noindex: