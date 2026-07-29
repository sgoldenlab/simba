Geometry transformations
========================

.. contents:: On this page
   :local:
   :depth: 1

Geometry mixin
----------------------------------------------

.. autoclass:: simba.mixins.geometry_mixin.GeometryMixin
   :members:
   :undoc-members:
   :inherited-members:


Geometry GPU methods
----------------------------------------------

.. automodule:: simba.data_processors.cuda.geometry
   :members:
   :undoc-members:
   :show-inheritance:


Geometry plotter
----------------------------------------------

Render the geometric shapes (bounding boxes, polygons, circles, lines) produced by the geometry methods as overlays on the original video, for visual inspection and figures.

.. autoclass:: simba.plotting.geometry_plotter.GeometryPlotter
   :members:
   :undoc-members:
   :noindex:


Geometry plotter (GPU)
----------------------------------------------

.. autoclass:: simba.data_processors.cuda.geometry_plotter_nvenc.GeometryPlotterNVENC
   :members:
   :undoc-members:
   :noindex: