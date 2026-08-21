Network transformations
===========================

.. contents:: On this page
   :local:
   :depth: 1

Graph and diversity-index methods from :class:`~simba.mixins.network_mixin.NetworkMixin`, for representing animals or body-parts as network nodes and quantifying their interactions.

Graph construction
------------------------------------------

.. automethod:: simba.mixins.network_mixin.NetworkMixin.create_graph

.. automethod:: simba.mixins.network_mixin.NetworkMixin.create_multigraph

Centrality & community detection
------------------------------------------

.. automethod:: simba.mixins.network_mixin.NetworkMixin.graph_page_rank

.. automethod:: simba.mixins.network_mixin.NetworkMixin.multigraph_page_rank

.. automethod:: simba.mixins.network_mixin.NetworkMixin.graph_katz_centrality

.. automethod:: simba.mixins.network_mixin.NetworkMixin.graph_current_flow_closeness_centrality

.. automethod:: simba.mixins.network_mixin.NetworkMixin.girvan_newman

Graph visualization
------------------------------------------

.. automethod:: simba.mixins.network_mixin.NetworkMixin.visualize

Diversity & similarity indices
------------------------------------------

.. automethod:: simba.mixins.network_mixin.NetworkMixin.simpson_index

.. automethod:: simba.mixins.network_mixin.NetworkMixin.berger_parker

.. automethod:: simba.mixins.network_mixin.NetworkMixin.shannon_diversity_index

.. automethod:: simba.mixins.network_mixin.NetworkMixin.margalef_diversification_index

.. automethod:: simba.mixins.network_mixin.NetworkMixin.menhinicks_index

.. automethod:: simba.mixins.network_mixin.NetworkMixin.brillouins_index

.. automethod:: simba.mixins.network_mixin.NetworkMixin.sorensen_dice_coefficient
