Time-series transformations
===========================

.. contents:: On this page
   :local:
   :depth: 1

Time-series feature methods from :class:`~simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin`, grouped by purpose. Sliding-window variants are listed beside their whole-session counterparts.

Complexity & fractal measures
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.hjort_parameters

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_hjort_parameters

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.petrosian_fractal_dimension

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_petrosian_fractal_dimension

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.higuchi_fractal_dimension

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.permutation_entropy

Extrema, crossings & strikes
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.local_maxima_minima

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.crossings

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_crossings

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.longest_strike

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_longest_strike

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.spike_finder

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.spike_train_finder

Percentiles & distribution position
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.percentile_difference

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_percentile_difference

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.percent_beyond_n_std

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_percent_beyond_n_std

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.percent_in_percentile_window

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_percent_in_percentile_window

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_pct_in_top_n

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_unique

Descriptive statistics over windows
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_variance

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_descriptive_statistics

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_window_stats

Frequency domain & correlation
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.dominant_frequencies

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.benford_correlation

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_benford_correlation

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_two_signal_crosscorrelation

Event timing
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.time_since_previous_threshold

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.time_since_previous_target_value

Stationarity & causality
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_stationary

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.granger_tests

Kinematics
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.acceleration

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_displacement

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.mean_squared_jerk

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_mean_squared_jerk

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.avg_kinetic_energy

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_avg_kinetic_energy

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.momentum_magnitude

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_momentum_magnitude

Path shape & dispersion
------------------------------------------

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.line_length

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_line_length

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.linearity_index

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_linearity_index

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.entropy_of_directional_changes

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_entropy_of_directional_changes

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.path_curvature

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_path_curvature

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.spatial_density

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_spatial_density

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.path_aspect_ratio

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_path_aspect_ratio

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.radial_eccentricity

.. automethod:: simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.radial_dispersion_index

Time-series GPU methods
------------------------------------------

.. automodule:: simba.data_processors.cuda.timeseries
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:
