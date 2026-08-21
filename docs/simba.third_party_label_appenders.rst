Third-party label appenders
===========================

.. contents:: On this page
   :local:
   :depth: 1

BENTO
-----------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.BENTO_appender
   :members:
   :undoc-members:
   :show-inheritance:

BORIS
-----------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.BORIS_appender
   :members:
   :undoc-members:
   :show-inheritance:

BORIS source cleaner
-----------------------------------------------------------

.. autoclass:: simba.third_party_label_appenders.boris_source_cleaner.BorisSourceCleaner
   :members:
   :undoc-members:
   :show-inheritance:

Deepethogram
------------------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.deepethogram_importer
   :members:
   :undoc-members:
   :show-inheritance:

Ethovison
--------------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.ethovision_import
   :members:
   :undoc-members:
   :show-inheritance:

Noldus Observer
--------------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.observer_importer
   :members:
   :undoc-members:
   :show-inheritance:

Solomon coder
-------------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.solomon_importer
   :members:
   :undoc-members:
   :show-inheritance:

Shah appender
-------------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.shah_appender
   :members:
   :undoc-members:
   :show-inheritance:

Generic third-party appender tool
------------------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.third_party_appender
   :members:
   :undoc-members:
   :show-inheritance:

Third-party annotation tools
------------------------------------------------------------------

.. automodule:: simba.third_party_label_appenders.tools
   :members:
   :undoc-members:
   :show-inheritance:


Annotation format converters
---------------------------------------

.. automodule:: simba.third_party_label_appenders.converters
   :members:
   :undoc-members:
   :show-inheritance:


COCO key-points -> YOLO pose-estimation format conversion
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.coco_keypoints_to_yolo.COCOKeypoints2Yolo
   :members:
   :undoc-members:
   :show-inheritance:


COCO key-points -> YOLO bounding box conversion
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.coco_keypoints_to_yolo_bbox.COCOKeypoints2YoloBbox
   :members:
   :undoc-members:
   :show-inheritance:


COCO key-points -> YOLO segmentation conversion
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.coco_keypoints_to_yolo_seg.COCOKeypoints2YoloSeg
   :members:
   :undoc-members:
   :show-inheritance:


SAM3 -> YOLO segmentation project
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.sam3_to_yolo_seg.SAM3ToYoloSeg
   :members:
   :undoc-members:
   :show-inheritance:


SAM3 -> YOLO bounding-box (detection) project
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.sam3_to_yolo_bbox.SAM3ToYoloBBox
   :members:
   :undoc-members:
   :show-inheritance:


Merge multiple YOLO projects
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.merge_yolo_projects.MergeYoloProjects
   :members:
   :undoc-members:
   :show-inheritance:


Multi-animal DeepLabCut predictions -> YOLO pose-estimation annotations format conversion
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.dlc_ma_h5_to_yolo.MADLCH52Yolo
   :members:
   :undoc-members:
   :show-inheritance:


DeepLabCut predictions -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.dlc_to_yolo.DLC2Yolo
   :members:
   :undoc-members:
   :show-inheritance:

Lightning Pose annotations -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.litpose_to_yolo_keypoints.LitPose2YOLO
   :members:
   :undoc-members:
   :show-inheritance:

Lightning Pose annotations -> YOLO bounding box annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.litpose_to_yolo_bbox.LitPose2YOLOBbox
   :members:
   :undoc-members:
   :show-inheritance:


Merge Lightning Pose projects
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.litpose_merge_projects.LitPoseMergeProjects
   :members:
   :undoc-members:
   :show-inheritance:


Crop Lightning Pose annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.litpose_crop_annotations.CropLPAnnotations
   :members:
   :undoc-members:
   :show-inheritance:


Crop Lightning Pose annotations (bounding box square)
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.litpose_crop_annotations_bbox_square.CropLPAnnotationsBboxSquare
   :members:
   :undoc-members:
   :show-inheritance:


Create Lightning Pose bounding box files
---------------------------------------

.. autofunction:: simba.third_party_label_appenders.transform.utils.get_litpose_project_bboxes
   :noindex:


Multi-animal DeepLabCut -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.dlc_multi_to_yolo.MultiDLC2Yolo
   :members:
   :undoc-members:
   :show-inheritance:


DeepLabCut single-to-multi-animal format converter
---------------------------------------

.. automodule:: simba.third_party_label_appenders.transform.dlc_single_to_multi_format_converter
   :members:
   :undoc-members:
   :show-inheritance:


DeepLabCut annotations -> Labelme annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.dlc_to_labelme.DLC2Labelme
   :members:
   :undoc-members:
   :show-inheritance:


Labelme annotations -> DeepLabCut annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.labelme_to_dlc.Labelme2DLC
   :members:
   :undoc-members:
   :show-inheritance:


Labelme annotations -> DataFrame
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.labelme_to_df.LabelMe2DataFrame
   :members:
   :undoc-members:
   :show-inheritance:


Labelme annotations -> YOLO bounding box annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.labelme_to_yolo.LabelmeBoundingBoxes2YoloBoundingBoxes
   :members:
   :undoc-members:
   :show-inheritance:


Labelme points -> YOLO keypoints annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.labelme_to_yolo_keypoints.LabelmeKeypoints2YoloKeypoints
   :members:
   :undoc-members:
   :show-inheritance:


Labelme points -> YOLO segmentation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.labelme_to_yolo_seg.LabelmeKeypoints2YoloSeg
   :members:
   :undoc-members:
   :show-inheritance:

SimBA ROIs -> YOLO bounding box annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.simba_roi_to_yolo.SimBAROI2Yolo
   :members:
   :undoc-members:
   :show-inheritance:

SimBA pose-estimation -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.simba_to_yolo.SimBA2Yolo
   :members:
   :undoc-members:
   :show-inheritance:

SimBA pose-estimation -> YOLO segmentation annotations
------------------------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.simba_to_yolo_seg.SimBA2YoloSegmentation
   :members:
   :undoc-members:
   :show-inheritance:

SLEAP CSV predictions -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.sleap_csv_to_yolo.Sleap2Yolo
   :members:
   :undoc-members:
   :show-inheritance:

SLEAP H5 predictions -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.sleap_h5_to_yolo.SleapH52Yolo
   :members:
   :undoc-members:
   :show-inheritance:

SLEAP annotations -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.sleap_to_yolo.SleapAnnotations2Yolo
   :members:
   :undoc-members:
   :show-inheritance:

Annotation conversion utilities
---------------------------------------
.. automodule:: simba.third_party_label_appenders.transform.utils
   :members:
   :undoc-members:
   :show-inheritance:


YOLO labels -> YOLO detection project
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.yolo_labels_to_yolo_project.YoloLabels2YoloProject
   :members:
   :undoc-members:
   :show-inheritance:


Visualize YOLO annotations on images
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.yolo_to_imgs.Yolo2Imgs
   :members:
   :undoc-members:
   :show-inheritance:







