Mixins (overview)
====================

Much of SimBA's analysis logic lives in **mixin classes** — reusable libraries of
static/instance methods that the pipelines, GUI and CLI all draw on. Rather than
call the pipelines, you can import a mixin directly and use its methods on your own
arrays and DataFrames.

Each mixin is documented in the topical section that matches what it does. Use this
page as the map; follow a link for the full method reference. Every class below lives
in ``simba.mixins`` — e.g. ``from simba.mixins.geometry_mixin import GeometryMixin``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mixin class
     - Documented in
   * - **GeometryMixin** — convert body-parts into polygons/circles/lines and compute spatial relationships
     - :doc:`Geometry transformations <simba.geometry_mixin>`
   * - **Statistics** — sliding/static-window statistics, distances, drift and distribution tests
     - :doc:`Statistics transformations <simba.statistics_mixin>`
   * - **CircularStatisticsMixin** — wraparound-aware angle/heading statistics
     - :doc:`Circular transformations <simba.circular_statistics>`
   * - **FeatureExtractionMixin** — low-level feature primitives used by the extraction pipelines
     - :doc:`Feature extraction mixins <simba.feature_extraction_mixins>`
   * - **FeatureExtractionSupplemental** — additional feature methods the default extractors don't call
     - :doc:`Feature extraction mixins <simba.feature_extraction_mixins>`
   * - **TimeseriesFeatureMixin** — time-series complexity and windowed descriptors
     - :doc:`Time-series transformations <simba.timeseries>`
   * - **ImageMixin** — frame slicing and visual-feature extraction from tracking data
     - :doc:`Image transformations <simba.image_transformations>`
   * - **NetworkMixin** — build and analyse graphs from pose time-series
     - :doc:`Network transformations <simba.networks>`
   * - **PlottingMixin** — shared plotting/visualization helpers
     - :doc:`Plotting and visualization tools <simba.plotting>`
   * - **TrainModelMixin** — train, grid-search and run inference with classifiers
     - :doc:`Model tools <simba.model_mixin>`
   * - **UMLMixin** — define and fit the UMAP / HDBSCAN models behind the unsupervised pipeline
     - :doc:`Unsupervised learning <simba.unsupervised>`
   * - **ConfigReader** — parse SimBA project config and metadata
     - :doc:`Config reader <simba.config_reader>`
   * - **PoseImporterMixin** — locate pose files, pair them with videos, run the multi-animal ID assignment UI
     - :doc:`Pose-estimation import tools <simba.pose_importers>`
   * - **AnnotatorMixin** — shared tkinter frames and callbacks behind the annotation interfaces
     - :doc:`Labeling tools <simba.labelling>`
   * - **PopUpMixin** — base class for SimBA's pop-up windows: drop-downs, checkboxes, entry boxes, listboxes
     - :doc:`User Interface (UI) tools <simba.ui>`
   * - **AbstractFeatureExtraction** — abstract base class a custom feature extractor implements
     - `Abstract base classes`_ (below)

.. seealso::
   GPU-accelerated versions of several of these (geometry, image, statistics,
   circular statistics, time-series) are collected under :doc:`GPU acceleration <simba.gpu_helpers>`.


Abstract base classes
--------------------------------------------

.. automodule:: simba.mixins.abstract_classes
   :members:
   :undoc-members:
   :show-inheritance:
