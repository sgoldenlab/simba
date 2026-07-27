Statistics transformations
==========================================

.. contents:: On this page
   :local:
   :depth: 1

Statistical feature methods from :class:`~simba.mixins.statistics_mixin.Statistics`, grouped by purpose. Many have GPU-accelerated counterparts (see below).

Descriptive statistics
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.geometric_mean

.. automethod:: simba.mixins.statistics_mixin.Statistics.pct_in_top_n

.. automethod:: simba.mixins.statistics_mixin.Statistics.symmetry_index

Effect sizes & association
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.eta_squared

.. automethod:: simba.mixins.statistics_mixin.Statistics.d_prime

.. automethod:: simba.mixins.statistics_mixin.Statistics.relative_risk

.. automethod:: simba.mixins.statistics_mixin.Statistics.youden_j

Hypothesis tests
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.chow_test

.. automethod:: simba.mixins.statistics_mixin.Statistics.cochrans_q

.. automethod:: simba.mixins.statistics_mixin.Statistics.grubbs_test

.. automethod:: simba.mixins.statistics_mixin.Statistics.kruskal_scipy

.. automethod:: simba.mixins.statistics_mixin.Statistics.mcnemar

.. automethod:: simba.mixins.statistics_mixin.Statistics.one_way_anova_scipy

.. automethod:: simba.mixins.statistics_mixin.Statistics.pairwise_tukeyhsd_scipy

.. automethod:: simba.mixins.statistics_mixin.Statistics.wilcoxon

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_shapiro_wilks

.. automethod:: simba.mixins.statistics_mixin.Statistics.hartley_fmax

.. automethod:: simba.mixins.statistics_mixin.Statistics.windowed_frequentist_distribution_tests

Distribution distances & drift
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.population_stability_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_population_stability_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.jensen_shannon_divergence

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_jensen_shannon_divergence

.. automethod:: simba.mixins.statistics_mixin.Statistics.kullback_leibler_divergence

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_kullback_leibler_divergence

.. automethod:: simba.mixins.statistics_mixin.Statistics.wasserstein_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_wasserstein_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.hellinger_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.total_variation_distance

Vector similarity & distance metrics
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.cosine_similarity

.. automethod:: simba.mixins.statistics_mixin.Statistics.gower_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.jaccard_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.kumar_hassebrook_similarity

.. automethod:: simba.mixins.statistics_mixin.Statistics.manhattan_distance_cdist

.. automethod:: simba.mixins.statistics_mixin.Statistics.normalized_google_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.wave_hedges_distance

Cluster-validity indices
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.banfeld_raftery_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.bouguessa_wang_sun_v2

.. automethod:: simba.mixins.statistics_mixin.Statistics.c_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.cop_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.davis_bouldin

.. automethod:: simba.mixins.statistics_mixin.Statistics.dunn_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.dunn_symmetry_idx

.. automethod:: simba.mixins.statistics_mixin.Statistics.i_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.krzanowski_lai_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.mclain_rao_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.pbm_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.ray_turi_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.rmsstd

.. automethod:: simba.mixins.statistics_mixin.Statistics.s_dbw_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.scott_symons_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.sd_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.silhouette_score

.. automethod:: simba.mixins.statistics_mixin.Statistics.wemmert_gancarski_index

.. automethod:: simba.mixins.statistics_mixin.Statistics.xie_beni

Clustering agreement
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.adjusted_mutual_info

.. automethod:: simba.mixins.statistics_mixin.Statistics.adjusted_rand

.. automethod:: simba.mixins.statistics_mixin.Statistics.fowlkes_mallows

.. automethod:: simba.mixins.statistics_mixin.Statistics.get_clustering_purity

Outlier detection
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.elliptic_envelope

.. automethod:: simba.mixins.statistics_mixin.Statistics.hbos

.. automethod:: simba.mixins.statistics_mixin.Statistics.isolation_forest

.. automethod:: simba.mixins.statistics_mixin.Statistics.local_outlier_factor

Body-part movement & geometry
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.bodypart_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.change_in_bodypart_euclidean_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.framewise_bodypart_movement

.. automethod:: simba.mixins.statistics_mixin.Statistics.keypoint_distances

.. automethod:: simba.mixins.statistics_mixin.Statistics.three_point_angle

.. automethod:: simba.mixins.statistics_mixin.Statistics.convex_hull_calculator_mp

.. automethod:: simba.mixins.statistics_mixin.Statistics.minimum_bounding_rectangle

.. automethod:: simba.mixins.statistics_mixin.Statistics.line_crosses_to_static_targets

.. automethod:: simba.mixins.statistics_mixin.Statistics.check_directionality_cords

.. automethod:: simba.mixins.statistics_mixin.Statistics.check_directionality_viable

.. automethod:: simba.mixins.statistics_mixin.Statistics.find_collinear_features

Smoothing & array shifting
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.dataframe_gaussian_smoother

.. automethod:: simba.mixins.statistics_mixin.Statistics.dataframe_savgol_smoother

.. automethod:: simba.mixins.statistics_mixin.Statistics.create_shifted_array

.. automethod:: simba.mixins.statistics_mixin.Statistics.create_shifted_df

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_dominant_frequencies

DataFrame & header utilities
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.get_bp_headers

.. automethod:: simba.mixins.statistics_mixin.Statistics.get_feature_extraction_headers

.. automethod:: simba.mixins.statistics_mixin.Statistics.insert_default_headers_for_feature_extraction

Statistics GPU methods
------------------------------------------

.. automodule:: simba.data_processors.cuda.statistics
   :members:
   :undoc-members:
   :show-inheritance:
