Circular transformations
==========================================

.. contents:: On this page
   :local:
   :depth: 1

Wraparound-aware circular-statistics methods from :class:`~simba.mixins.circular_statistics.CircularStatisticsMixin`.

Direction (instantaneous)
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.two_point_direction

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.three_point_direction

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.direction_two_bps

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.direction_three_bps

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.degrees_to_cardinal

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_bearing

Central tendency & dispersion
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.circular_mean

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_mean

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.circular_std

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_std

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.mean_resultant_vector_length

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_mean_resultant_vector_length

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.circular_range

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_range

Turning direction & angular change
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.preferred_turning_direction

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_preferred_turning_direction

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.rotational_direction

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.instantaneous_angular_velocity

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_angular_diff

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.agg_angular_diff_timebins

Circular uniformity tests
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.rayleigh

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_rayleigh_z

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.hodges_ajne

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_hodges_ajne

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.watsons_u

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.rao_spacing

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_rao_spacing

Two-sample tests
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.kuipers_two_sample_test

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_kuipers_two_sample_test

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.watson_williams_test

Circular correlation
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.circular_correlation

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_correlation

Angular hotspots & circle fitting
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.circular_hotspots

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_hotspots

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.fit_circle

Circular GPU methods
------------------------------------------

.. automodule:: simba.data_processors.cuda.circular_statistics
   :members:
   :undoc-members:
   :show-inheritance:
