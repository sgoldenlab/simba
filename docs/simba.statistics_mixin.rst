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

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_cumulative_mean

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_z_scores

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_skew

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_kurtosis

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_iqr

.. automethod:: simba.mixins.statistics_mixin.Statistics.cov_matrix

Effect sizes & association
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.eta_squared

.. automethod:: simba.mixins.statistics_mixin.Statistics.d_prime

.. automethod:: simba.mixins.statistics_mixin.Statistics.relative_risk

.. automethod:: simba.mixins.statistics_mixin.Statistics.youden_j

.. automethod:: simba.mixins.statistics_mixin.Statistics.cohens_d

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_cohens_d

.. automethod:: simba.mixins.statistics_mixin.Statistics.cohens_h

.. automethod:: simba.mixins.statistics_mixin.Statistics.cohens_kappa

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_eta_squared

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_relative_risk

.. automethod:: simba.mixins.statistics_mixin.Statistics.phi_coefficient

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_phi_coefficient

.. automethod:: simba.mixins.statistics_mixin.Statistics.concordance_ratio

.. automethod:: simba.mixins.statistics_mixin.Statistics.yule_coef

Correlation
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.pearsons_r

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_pearsons_r

.. automethod:: simba.mixins.statistics_mixin.Statistics.spearman_rank_correlation

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_spearman_rank_correlation

.. automethod:: simba.mixins.statistics_mixin.Statistics.kendall_tau

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_kendall_tau

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_autocorrelation

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

.. automethod:: simba.mixins.statistics_mixin.Statistics.independent_samples_t

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_independent_sample_t

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_independent_samples_t

.. automethod:: simba.mixins.statistics_mixin.Statistics.one_way_anova

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_one_way_anova

.. automethod:: simba.mixins.statistics_mixin.Statistics.kruskal_wallis

.. automethod:: simba.mixins.statistics_mixin.Statistics.mann_whitney

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_mann_whitney

.. automethod:: simba.mixins.statistics_mixin.Statistics.two_sample_ks

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_two_sample_ks

.. automethod:: simba.mixins.statistics_mixin.Statistics.levenes

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_levenes

.. automethod:: simba.mixins.statistics_mixin.Statistics.rolling_barletts_test

.. automethod:: simba.mixins.statistics_mixin.Statistics.brunner_munzel

.. automethod:: simba.mixins.statistics_mixin.Statistics.chi_square

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

.. automethod:: simba.mixins.statistics_mixin.Statistics.circular_euclidean_kantorovich

Vector similarity & distance metrics
------------------------------------------

.. automethod:: simba.mixins.statistics_mixin.Statistics.cosine_similarity

.. automethod:: simba.mixins.statistics_mixin.Statistics.gower_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.jaccard_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.kumar_hassebrook_similarity

.. automethod:: simba.mixins.statistics_mixin.Statistics.manhattan_distance_cdist

.. automethod:: simba.mixins.statistics_mixin.Statistics.normalized_google_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.wave_hedges_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.mahalanobis_distance_cdist

.. automethod:: simba.mixins.statistics_mixin.Statistics.czebyshev_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_czebyshev_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.hamming_distance

.. automethod:: simba.mixins.statistics_mixin.Statistics.bray_curtis_dissimilarity

.. automethod:: simba.mixins.statistics_mixin.Statistics.sokal_sneath

.. automethod:: simba.mixins.statistics_mixin.Statistics.sokal_michener

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

.. automethod:: simba.mixins.statistics_mixin.Statistics.calinski_harabasz

.. automethod:: simba.mixins.statistics_mixin.Statistics.kmeans_1d

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

.. automethod:: simba.mixins.statistics_mixin.Statistics.mad_median_rule

.. automethod:: simba.mixins.statistics_mixin.Statistics.sliding_mad_median_rule

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
