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

Turning direction
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.preferred_turning_direction

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_preferred_turning_direction

Circular uniformity tests
------------------------------------------

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.hodges_ajne

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_hodges_ajne

.. automethod:: simba.mixins.circular_statistics.CircularStatisticsMixin.watsons_u

Circular GPU methods
------------------------------------------

.. automodule:: simba.data_processors.cuda.circular_statistics
   :members:
   :undoc-members:
   :show-inheritance:
