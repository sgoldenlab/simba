Geometry transformations
===========================

.. contents:: On this page
   :local:
   :depth: 1

Geometric methods from :class:`~simba.mixins.geometry_mixin.GeometryMixin`, grouped by purpose. Most single-shape methods have a ``multiframe_`` counterpart that processes a full video in parallel.

Body-parts to geometries
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.bodyparts_to_polygon

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.bodyparts_to_points

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.bodyparts_to_circle

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.bodyparts_to_line

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.bodyparts_to_multistring_skeleton

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.to_linestring

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.contours_to_geometries

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.keypoints_to_axis_aligned_bounding_box

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.delaunay_triangulate_keypoints

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.simba_roi_to_geometries

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.sleap_csv_to_geometries

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.filter_low_p_bps_for_shapes

Shape transformations
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.buffer_shape

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.parallel_offset_polygon

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.adjust_geometry_locations

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.minimum_rotated_rectangle

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.smooth_geometry_bspline

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.union

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.difference

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.symmetric_difference

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.geometries_to_exterior_keypoints

Overlap, containment & contact
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.compute_pct_shape_overlap

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.compute_shape_overlap

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.is_shape_covered

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.is_containing

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.is_touching

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.crosses

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.points_in_polygon

Measurements
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.area

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.length

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.shape_distance

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.get_center

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.get_shape_statistics

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.rank_shapes

Lines, paths & trajectory similarity
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.static_point_lineside

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.point_lineside

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.extend_line_to_bounding_box_edges

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.line_split_bounding_box

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.locate_line_point

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.linear_frechet_distance

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.hausdorff_distance

Image grids & occupancy
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.bucket_img_into_grid_points

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.bucket_img_into_grid_square

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.bucket_img_into_grid_hexagon

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.cumsum_coord_geometries

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.cumsum_bool_geometries

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.cumsum_animal_geometries_grid

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.geometry_transition_probabilities

Image content within geometries
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.get_geometry_brightness_intensity

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.geometry_histocomparison

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.geometry_contourcomparison

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multifrm_geometry_histocomparison

Visualization
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.view_shapes

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.geometry_video

Multiprocess (multi-frame) variants
------------------------------------------

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_bodyparts_to_polygon

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_bodypart_to_point

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_bodyparts_to_circle

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_bodyparts_to_line

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_bodyparts_to_multistring_skeleton

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_delaunay_triangulate_keypoints

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_buffer_shapes

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_minimum_rotated_rectangle

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_union

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_difference

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_symmetric_difference

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_compute_pct_shape_overlap

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_compute_shape_overlap

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_is_shape_covered

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_area

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_length

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_shape_distance

.. automethod:: simba.mixins.geometry_mixin.GeometryMixin.multiframe_hausdorff_distance

Geometry GPU methods
------------------------------------------

.. automodule:: simba.data_processors.cuda.geometry
   :members:
   :undoc-members:
   :show-inheritance:

Geometry plotter
------------------------------------------

Render the geometric shapes (bounding boxes, polygons, circles, lines) produced by the geometry methods as overlays on the original video, for visual inspection and figures.

.. autoclass:: simba.plotting.geometry_plotter.GeometryPlotter
   :members:
   :undoc-members:
   :noindex:

Geometry plotter (GPU)
------------------------------------------

.. autoclass:: simba.data_processors.cuda.geometry_plotter_nvenc.GeometryPlotterNVENC
   :members:
   :undoc-members:
   :noindex:
