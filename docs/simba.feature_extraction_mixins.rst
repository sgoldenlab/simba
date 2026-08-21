Feature extraction mixins
===========================

.. contents:: On this page
   :local:
   :depth: 1

Low-level feature methods from :class:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin` and :class:`~simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental`, grouped by purpose.

Distances between body-parts
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.euclidean_distance

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.bodypart_distance

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.keypoint_distances

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.change_in_bodypart_euclidean_distance

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.cdist

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.cdist_3d

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.find_midpoints

Angles
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.angle3pt

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.angle3pt_vectorized

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.three_point_angle

Convex hull & bounding shapes
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.convex_hull_calculator_mp

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.minimum_bounding_rectangle

ROI membership
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance_roi

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_inside_rectangle_roi

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_inside_polygon_roi

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.is_inside_circle

Directionality
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.check_directionality_viable

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.check_directionality_cords

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.jitted_line_crosses_to_nonstatic_targets

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.jitted_line_crosses_to_static_targets

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.line_crosses_to_static_targets

Movement
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_bodypart_movement

Similarity & value distributions
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.cosine_similarity

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.count_values_in_range

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.windowed_frequentist_distribution_tests

Smoothing & array shifting
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.dataframe_gaussian_smoother

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.dataframe_savgol_smoother

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.create_shifted_df

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.create_shifted_array

Header utilities
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.get_bp_headers

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.get_feature_extraction_headers

.. automethod:: simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.insert_default_headers_for_feature_extraction

Supplementary feature methods
------------------------------------------

Rolling ratios & category switches
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.peak_ratio

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.rolling_peak_count_ratio

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.rolling_categorical_switches_ratio

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.consecutive_time_series_categories_count

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.rolling_horizontal_vs_vertical_movement

Distances to borders & frame-wise change
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.euclidean_distance_timeseries_change

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.border_distances

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.img_edge_distances

Movement & velocity summaries
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.velocity_aggregator

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.distance_and_velocity

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.movement_stats_from_bouts_df

Path & sequence analysis
------------------------------------------

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.find_path_loops

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.spontaneous_alternations

.. automethod:: simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.sequential_lag_analysis
