Feature extraction wrappers
===========================

.. contents:: On this page
   :local:
   :depth: 1

Default feature extractor for 2 animals and 14 body-parts
---------------------------------------------------------

.. autoclass:: simba.feature_extractors.feature_extractor_14bp.ExtractFeaturesFrom14bps
   :members:
   :show-inheritance:
   :undoc-members:

Default feature extractor for 1 animals and 16 body-parts
---------------------------------------------------------

.. automodule:: simba.feature_extractors.feature_extractor_16bp
   :members:
   :show-inheritance:
   :undoc-members:

Default feature extractor for 1 animals and 4 body-parts
--------------------------------------------------------

.. automodule:: simba.feature_extractors.feature_extractor_4bp
   :members:
   :show-inheritance:
   :undoc-members:

Default feature extractor for 1 animals and 7 body-parts
--------------------------------------------------------

.. automodule:: simba.feature_extractors.feature_extractor_7bp
   :members:
   :show-inheritance:
   :undoc-members:

Default feature extractor for 1 animals and 8 body-parts
--------------------------------------------------------

.. automodule:: simba.feature_extractors.feature_extractor_8bp
   :members:
   :show-inheritance:
   :undoc-members:

Default feature extractor for 2 animals and 16 body-parts
---------------------------------------------------------------------

.. automodule:: simba.feature_extractors.feature_extractor_8bps_2_animals
   :members:
   :show-inheritance:
   :undoc-members:

Default feature extractor for 1 animals and 9 body-parts
--------------------------------------------------------

.. automodule:: simba.feature_extractors.feature_extractor_9bp
   :members:
   :show-inheritance:
   :undoc-members:

Default feature extractor for user-defined body-parts
------------------------------------------------------------------

.. automodule:: simba.feature_extractors.feature_extractor_user_defined
   :members:
   :show-inheritance:
   :undoc-members:

Feature extractor for feature subset family
-------------------------------------------------

.. automodule:: simba.feature_extractors.feature_subsets
   :members:
   :show-inheritance:
   :undoc-members:

Jitted methods for convex-hull related calculations
-----------------------------------------------

.. automodule:: simba.feature_extractors.perimeter_jit
   :members:
   :show-inheritance:
   :undoc-members:

Rearing and grooming feature extraction wrapper
-----------------------------------------------

.. autoclass:: simba.feature_extractors.mitra_feature_extractor.MitraFeatureExtractor
   :members:
   :show-inheritance:
   :undoc-members:

Straub tail feature extraction wrapper
-----------------------------------------------

.. autoclass:: simba.feature_extractors.straub_tail_analyzer.StraubTailAnalyzer
   :members:
   :show-inheritance:
   :undoc-members:

Mexican cave fish feature extraction wrapper
-----------------------------------------------

.. autoclass:: simba.feature_extractors.cave_fish_featurizer.CaveFishFeaturizer
   :members:
   :show-inheritance:
   :undoc-members:

Rat social behavior feature extraction wrapper
-----------------------------------------------

.. autoclass:: simba.feature_extractors.rat_social_featurizer.RatSocialFeaturizer
   :members:
   :show-inheritance:
   :undoc-members:

Gerbil single body-part feature extraction wrapper
-----------------------------------------------

.. autoclass:: simba.feature_extractors.gerbil_featurizer.GerbilFeaturizer
   :members:
   :show-inheritance:
   :undoc-members:

Aggression feature extractor (11/25)
-----------------------------------------------

.. autoclass:: simba.feature_extractors.aggression_feature_extractor.AgressionFeatureExtractor
   :members:
   :show-inheritance:
   :undoc-members:

AMBER pipeline feature extractor
-----------------------------------------------

.. autoclass:: simba.feature_extractors.amber_feature_extractor.AmberFeatureExtractor
   :members:
   :show-inheritance:
   :undoc-members:

Boundary rearing feature extractor
-----------------------------------------------

.. autoclass:: simba.feature_extractors.boundary_rearing_analyzer.BoundaryRearingFeaturizer
   :members:
   :show-inheritance:
   :undoc-members:

Riptortus pedestris feature extractor
-----------------------------------------------

.. autoclass:: simba.feature_extractors.riptortus_featurizer.RiptortusFeaturizer
   :members:
   :show-inheritance:
   :undoc-members:

Custom feature extractor
-----------------------------------------------

.. autoclass:: simba.utils.custom_feature_extractor.CustomFeatureExtractor
   :members:
   :show-inheritance:
   :undoc-members:


Wing-wave feature extraction wrapper
--------------------------------------------

.. automodule:: simba.feature_extractors.wingwave_extractor
   :members:
   :undoc-members:
   :show-inheritance:


Pose configurations
-------------------

The pose configurations selectable when creating a SimBA project. The number on each
marker is that body-part's column order in ``bp_names.csv``; single-animal cards name
each part beside its marker, multi-animal cards name each part once in a legend.

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Schematic
     - Configuration
     - Body-parts
   * - .. image:: _static/img/pose_configurations/1.png
          :width: 110
          :alt: 1 animal; 4 body-parts
     - 1 animal; 4 body-parts
     - Ear_left, Ear_right, Nose, Tail_base
   * - .. image:: _static/img/pose_configurations/2.png
          :width: 110
          :alt: 1 animal; 7 body-parts
     - 1 animal; 7 body-parts
     - adds Center, Lat_left, Lat_right
   * - .. image:: _static/img/pose_configurations/3.png
          :width: 110
          :alt: 1 animal; 8 body-parts
     - 1 animal; 8 body-parts
     - adds Tail_end
   * - .. image:: _static/img/pose_configurations/4.png
          :width: 110
          :alt: 1 animal; 9 body-parts
     - 1 animal; 9 body-parts
     - Nose, ears, hands, feet, Back, Tail
   * - .. image:: _static/img/pose_configurations/5.png
          :width: 110
          :alt: 2 animals; 8 body-parts
     - 2 animals; 8 body-parts
     - 4 body-parts per animal
   * - .. image:: _static/img/pose_configurations/6.png
          :width: 110
          :alt: 2 animals; 14 body-parts
     - 2 animals; 14 body-parts
     - 7 body-parts per animal
   * - .. image:: _static/img/pose_configurations/7.png
          :width: 110
          :alt: 2 animals; 16 body-parts
     - 2 animals; 16 body-parts
     - 8 body-parts per animal
   * - .. image:: _static/img/pose_configurations/8.png
          :width: 110
          :alt: MARS
     - MARS
     - Nose, ears, Neck, hips, Tail, per animal
   * - .. image:: _static/img/pose_configurations/9.png
          :width: 110
          :alt: Multi-animals; 4 body-parts
     - Multi-animals; 4 body-parts
     - 4 body-parts, interchangeable animals
   * - .. image:: _static/img/pose_configurations/10.png
          :width: 110
          :alt: Multi-animals; 7 body-parts
     - Multi-animals; 7 body-parts
     - 7 body-parts, interchangeable animals
   * - .. image:: _static/img/pose_configurations/11.png
          :width: 110
          :alt: Multi-animals; 8 body-parts
     - Multi-animals; 8 body-parts
     - 8 body-parts, interchangeable animals
   * - .. image:: _static/img/pose_configurations/12.png
          :width: 110
          :alt: 3D tracking
     - 3D tracking
     - user-defined, three dimensional
   * - .. image:: _static/img/pose_configurations/amber.png
          :width: 110
          :alt: AMBER
     - AMBER
     - dam plus up to 12 pups
   * - .. image:: _static/img/pose_configurations/14.png
          :width: 110
          :alt: SimBA BLOB Tracking
     - SimBA BLOB Tracking
     - silhouette extremes, no pose estimation
   * - .. image:: _static/img/pose_configurations/15.png
          :width: 110
          :alt: FaceMap
     - FaceMap
     - facial landmarks
   * - .. image:: _static/img/pose_configurations/16.png
          :width: 110
          :alt: SuperAnimal-TopView
     - SuperAnimal-TopView
     - 27 body-parts, top view
