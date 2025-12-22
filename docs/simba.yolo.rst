YOLO methods
===============================

Methods for training YOLO models, creating training and validation datasets, and
converting behavioral neuroscience specific datasets to YOLO datasets.

Utilities
-----------------------------------------------------------------------

.. automodule:: simba.utils.yolo
   :members:
   :show-inheritance:


Bounding-box inference
-----------------------------------------------------------------------

.. autoclass:: simba.model.yolo_inference.YoloInference
   :members:
   :show-inheritance:

Pose-estimation inference
-----------------------------------------------------------------------

.. autoclass:: simba.model.yolo_pose_inference.YOLOPoseInference
   :members:
   :show-inheritance:


YOLO pose-estimation segmentation visualizer
-----------------------------------------------------------------------

.. autoclass:: simba.plotting.yolo_seg_visualizer.YOLOSegmentationVisualizer
   :members:
   :show-inheritance:

YOLO pose-estimation segmentation inference
-----------------------------------------------------------------------

.. autoclass:: simba.model.yolo_seg_inference.YOLOSegmentationInference
   :members:
   :show-inheritance:


Pose-estimation track inference
-----------------------------------------------------------------------

.. autoclass:: simba.model.yolo_pose_track_inference.YOLOPoseTrackInference
   :members:
   :show-inheritance:

Pose-estimation track plotting
-----------------------------------------------------------------------

.. autoclass:: simba.plotting.yolo_pose_track_visualizer.YOLOPoseTrackVisualizer
   :members:
   :show-inheritance:


Pose-estimation plotting
-----------------------------------------------------------------------

.. autoclass:: simba.plotting.yolo_pose_visualizer.YOLOPoseVisualizer
   :members:
   :show-inheritance:

Bounding box plotting
-----------------------------------------------------------------------

.. autoclass:: simba.plotting.yolo_visualize.YOLOVisualizer
   :members:
   :show-inheritance:

COCO key-points -> YOLO pose-estimation format conversion
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.coco_keypoints_to_yolo.COCOKeypoints2Yolo
   :members:
   :show-inheritance:


COCO key-points -> YOLO bounding box conversion
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.coco_keypoints_to_yolo_bbox.COCOKeypoints2YoloBbox
   :members:
   :show-inheritance:



COCO key-points -> YOLO segmentation conversion
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.coco_keypoints_to_yolo_seg.COCOKeypoints2YoloSeg
   :members:
   :show-inheritance:


Multi-animal DeepLabCut predictions -> YOLO pose-estimation annotations format conversion
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.dlc_ma_h5_to_yolo.MADLCH52Yolo
   :members:
   :show-inheritance:


DeepLabCut predictions -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.dlc_to_yolo.DLC2Yolo
   :members:
   :show-inheritance:



Labelme annotations -> YOLO bounding box annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.labelme_to_yolo.LabelmeBoundingBoxes2YoloBoundingBoxes
   :members:
   :show-inheritance:


Labelme points -> YOLO keypoints annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.labelme_to_yolo_keypoints.LabelmeKeypoints2YoloKeypoints
   :members:
   :show-inheritance:


Labelme points -> YOLO segmentation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.labelme_to_yolo_seg.LabelmeKeypoints2YoloSeg
   :members:
   :show-inheritance:

SimBA ROIs -> YOLO bounding box annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.simba_roi_to_yolo.SimBAROI2Yolo
   :members:
   :show-inheritance:

SimBA pose-estimation -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.simba_to_yolo.SimBA2Yolo
   :members:
   :show-inheritance:


SimBA pose-estimation -> YOLO segmentation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.simba_to_yolo_seg.SimBA2YoloSegmentation
   :members:
   :show-inheritance:

SLEAP CSV predictions -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.sleap_csv_to_yolo.Sleap2Yolo
   :members:
   :show-inheritance:

SLEAP H5 predictions -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.sleap_h5_to_yolo.SleapH52Yolo
   :members:
   :show-inheritance:

SLEAP annotations -> YOLO pose-estimation annotations
---------------------------------------

.. autoclass:: simba.third_party_label_appenders.transform.sleap_to_yolo.SleapAnnotations2Yolo
   :members:
   :show-inheritance:
