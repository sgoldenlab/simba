Related Software
================

A non-exhaustive map of the animal pose-estimation, tracking, and
behavior-analysis ecosystem that SimBA is commonly used alongside, grouped by
primary purpose. Many tools span several categories; each is listed under its
main use.

.. contents:: Categories
   :local:
   :depth: 1

Pose estimation
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DeepLabCut <https://www.deeplabcut.org/>`__
     - Open source
     - Markerless 2D/3D pose estimation
   * - `SLEAP <https://sleap.ai/>`__
     - Open source
     - Multi-animal pose estimation
   * - `DeepPoseKit <https://github.com/jgraving/deepposekit>`__
     - Open source
     - Pose estimation toolkit
   * - `Lightning Pose <https://github.com/danbider/lightning-pose>`__
     - Open source
     - Semi-supervised, multi-view pose estimation
   * - `Facemap <https://github.com/MouseLand/facemap>`__
     - Open source
     - Mouse orofacial tracking and neural prediction
   * - `OpenPose <https://github.com/CMU-Perceptual-Computing-Lab/openpose>`__
     - Open source
     - Real-time multi-person 2D keypoint detection
   * - `MMPose <https://github.com/open-mmlab/mmpose>`__
     - Open source
     - General-purpose pose estimation toolbox (OpenMMLab)
   * - `YOLO (Ultralytics) <https://github.com/ultralytics/ultralytics>`__
     - Open source
     - Real-time keypoint/pose estimation (SimBA-supported); also object detection and segmentation

3D pose estimation
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DeepFly3D <https://github.com/NeLy-EPFL/DeepFly3D>`__
     - Open source
     - 3D pose estimation for tethered Drosophila
   * - `Anipose <https://github.com/lambdaloop/anipose>`__
     - Open source
     - 3D pose estimation from synchronized cameras
   * - `DANNCE <https://github.com/spoonsso/dannce>`__
     - Open source
     - 3D landmark detection from multi-view video

Tracking and identity
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `TRex <https://trex.run/>`__
     - Open source
     - Real-time, identity-preserving tracking
   * - `idtracker.ai <https://idtracker.ai/>`__
     - Open source
     - Markerless identity tracking for animal groups
   * - `AlphaTracker <https://github.com/ZexinChen/AlphaTracker>`__
     - Open source
     - Multi-animal tracking, pose, and behavioral clustering
   * - `Tracktor <https://github.com/vivekhsridhar/tracktor>`__
     - Open source
     - OpenCV-based single- and multi-object tracker
   * - `ToxTrac <https://sourceforge.net/projects/toxtrac/>`__
     - Open source
     - Fast tracker for one or several animals
   * - `ezTrack <https://github.com/DeniseCaiLab/ezTrack>`__
     - Open source
     - Blob-based location and freezing tracking
   * - `LiveMouseTracker <https://micecraft.org/lmt>`__
     - Open source
     - Long-term mouse tracking via RFID and depth cameras
   * - `TrackMate <https://imagej.net/plugins/trackmate/>`__
     - Open source
     - ImageJ/Fiji object-tracking plugin
   * - `C-Trax <https://ctrax.sourceforge.net/>`__
     - Open source
     - Tracking of walking flies in groups

Supervised behavior classification
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `JAABA <https://www.janelia.org/open-science/jaaba>`__
     - Open source
     - Supervised behavior classification from trajectories
   * - `DeepEthogram <https://github.com/jbohnslav/deepethogram>`__
     - Open source
     - Supervised behavior classification from raw video
   * - `MARS <https://github.com/neuroethology/MARS>`__
     - Open source
     - Pose estimation and social behavior classification in mice
   * - `BehaviorDEPOT <https://github.com/DeNardoLab/BehaviorDEPOT>`__
     - Open source
     - Pose-guided behavior detection and analysis
   * - `LabGym <https://github.com/umyelab/LabGym>`__
     - Open source
     - Tracking and behavior classification via Mask R-CNN
   * - `SIPEC <https://github.com/damaggu/SIPEC>`__
     - Open source
     - End-to-end deep-learning behavioral analysis pipeline

Unsupervised behavior discovery
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `B-SOID <https://github.com/YttriLab/B-SOID>`__
     - Open source
     - Unsupervised behavior discovery from pose
   * - `A-SOID <https://github.com/YttriLab/A-SOID>`__
     - Open source
     - Active-learning behavior classification from pose
   * - `Keypoint-MoSeq <https://github.com/dattalab/keypoint-moseq>`__
     - Open source
     - Unsupervised behavioral syllable discovery from keypoints
   * - `VAME <https://github.com/EthoML/VAME>`__
     - Open source
     - Unsupervised behavioral motif discovery from pose
   * - `DeepOF <https://github.com/mlfpm/deepof>`__
     - Open source
     - Behavioral analysis of DeepLabCut/SLEAP tracking
   * - `TREBA <https://github.com/neuroethology/TREBA>`__
     - Open source
     - Trajectory embeddings for behavior representation learning
   * - `MotionMapper <https://github.com/gordonberman/MotionMapper>`__
     - Open source
     - Unsupervised behavioral mapping from postural dynamics
   * - `MoSeq <https://github.com/dattalab/moseq2-app>`__
     - Open source
     - Depth-video behavioral syllable segmentation

Object detection
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `YOLO (Ultralytics) <https://github.com/ultralytics/ultralytics>`__
     - Open source
     - Real-time object detection, segmentation, and pose framework
   * - `Detectron2 <https://github.com/facebookresearch/detectron2>`__
     - Open source
     - Object detection and instance segmentation library
   * - `Segment Anything (SAM 2) <https://github.com/facebookresearch/sam2>`__
     - Open source
     - Promptable image and video segmentation

Analysis and real-time pipelines
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DLC-Analyzer <https://github.com/ETHZ-INS/DLCAnalyzer>`__
     - Open source
     - Analysis and visualization of DeepLabCut output
   * - `DeepLabCut-Live <https://github.com/DeepLabCut/DeepLabCut-live>`__
     - Open source
     - Real-time pose estimation on live video
   * - `DeepLabStream <https://github.com/SchwarzNeuroconLab/DeepLabStream>`__
     - Open source
     - Real-time, closed-loop pose-based feedback
   * - `Bonsai <https://bonsai-rx.org/>`__
     - Open source
     - Visual reactive programming for experiment pipelines
   * - `AMBER-pipeline <https://github.com/lapphe/AMBER-pipeline>`__
     - Open source
     - Automated rodent maternal-behavior analysis

Manual annotation
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `BORIS <https://www.boris.unito.it/>`__
     - Open source
     - Event-logging and manual video annotation
   * - `Solomon Coder <https://solomon.andraspeter.com/>`__
     - Freeware
     - Manual behavior coding tool
   * - `Noldus Observer XT <https://noldus.com/observer-xt-human>`__
     - Commercial
     - Manual behavior annotation suite
   * - `CVAT <https://github.com/cvat-ai/cvat>`__
     - Open source
     - Image and video annotation for detection, segmentation, and keypoints
   * - `Label Studio <https://github.com/HumanSignal/label-studio>`__
     - Open source
     - Multi-type data labeling and annotation platform
   * - `VIA (VGG Image Annotator) <https://www.robots.ox.ac.uk/~vgg/software/via/>`__
     - Open source
     - Lightweight image, audio, and video annotation tool
   * - `ELAN <https://archive.mpi.nl/tla/elan>`__
     - Open source
     - Time-aligned annotation of video and audio

Commercial platforms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `Ethovision XT <https://www.noldus.com/ethovision>`__
     - Commercial
     - Video tracking and analysis
   * - `CatWalk XT <https://www.noldus.com/catwalk>`__
     - Commercial
     - Gait analysis platform
