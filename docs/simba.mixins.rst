Mixins (overview)
====================

Much of SimBA's analysis logic lives in **mixin classes** — reusable libraries of
static/instance methods that the pipelines, GUI and CLI all draw on. Rather than
call the pipelines, you can import a mixin directly and use its methods on your own
arrays and DataFrames.

Each mixin is documented in the topical section that matches what it does. Use this
page as the map; follow a link for the full method reference.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Mixin
     - Documented in
   * - **Geometry** — convert body-parts into polygons/circles/lines and compute spatial relationships
     - :doc:`Geometry transformations <simba.geometry_mixin>`
   * - **Statistics** — sliding/static-window statistics, distances, drift and distribution tests
     - :doc:`Statistics transformations <simba.statistics_mixin>`
   * - **Circular statistics** — wraparound-aware angle/heading statistics
     - :doc:`Circular transformations <simba.circular_statistics>`
   * - **Feature extraction** — low-level feature primitives used by the extraction pipelines
     - :doc:`Feature extraction mixins <simba.feature_extraction_mixins>`
   * - **Time-series** — time-series complexity and windowed descriptors
     - :doc:`Time-series transformations <simba.timeseries>`
   * - **Image** — frame slicing and visual-feature extraction from tracking data
     - :doc:`Image transformations <simba.image_transformations>`
   * - **Network** — build and analyse graphs from pose time-series
     - :doc:`Network transformations <simba.networks>`
   * - **Plotting** — shared plotting/visualization helpers
     - :doc:`Plotting and visualization tools <simba.plotting>`
   * - **Model / training** — train, grid-search and run inference with classifiers
     - :doc:`Model tools <simba.model_mixin>`
   * - **Config reader** — parse SimBA project config and metadata
     - :doc:`Config reader <simba.config_reader>`

.. seealso::
   GPU-accelerated versions of several of these (geometry, image, statistics,
   circular statistics, time-series) are collected under :doc:`GPU acceleration <simba.gpu_helpers>`.
