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
   * - `SuperAnimal / DLC Model Zoo <https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html>`__
     - Open source
     - Pretrained cross-species DeepLabCut models, usable without new labelling
   * - `DeepGraphPose <https://github.com/paninski-lab/deepgraphpose>`__
     - Open source
     - Pose estimation using graph-based spatiotemporal priors

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
   * - `OpenMonkeyStudio <https://github.com/OpenMonkeyStudio>`__
     - Open source
     - Markerless 3D pose estimation for freely moving macaques

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
   * - `AnimalTA <https://vchiara.eu/index.php/animalta>`__
     - Open source
     - GUI tracking of multiple animals across varied setups
   * - `BioTracker <https://github.com/BioroboticsLab/biotracker_core>`__
     - Open source
     - Modular video-tracking framework
   * - `FlyTracker <https://github.com/kristinbranson/FlyTracker>`__
     - Open source
     - Tracking and feature extraction for interacting flies
   * - `FicTrac <https://github.com/rjdmoore/fictrac>`__
     - Open source
     - Spherical-treadmill path tracking for tethered insects
   * - `ZebraZoom <https://zebrazoom.org/>`__
     - Open source
     - Zebrafish larva and adult behaviour tracking
   * - `Stytra <https://github.com/portugueslab/stytra>`__
     - Open source
     - Zebrafish tracking with closed-loop stimulus control

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
   * - `DeepAction <https://github.com/carlwharris/DeepAction>`__
     - Open source
     - Video-based behaviour classification with confidence-based review
   * - `DLC2Action <https://github.com/amathislab/DLC2action>`__
     - Open source
     - Deep-learning behaviour segmentation from pose
   * - `OpenLabCluster <https://github.com/shlizee/OpenLabCluster>`__
     - Open source
     - Active-learning clustering and classification from keypoints
   * - `CBAS <https://github.com/jones-lab-tamu/CBAS>`__
     - Open source
     - Circadian behavioural analysis suite for long recordings

Unsupervised behavior discovery
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `B-SOiD <https://github.com/YttriLab/B-SOID>`__
     - Open source
     - Unsupervised behavior discovery from pose
   * - `A-SOiD <https://github.com/YttriLab/A-SOID>`__
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

YOLO (Ultralytics) also belongs here; it is listed under `Pose estimation`_ because
that is the capability SimBA supports directly.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `Detectron2 <https://github.com/facebookresearch/detectron2>`__
     - Open source
     - Object detection and instance segmentation library
   * - `Segment Anything (SAM 2) <https://github.com/facebookresearch/sam2>`__
     - Open source
     - Promptable image and video segmentation

Vocalisation and audio
----------------------

Behavioural video is frequently paired with ultrasonic vocalisation (USV) recording;
these tools cover the audio half of that pipeline.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DeepSqueak <https://github.com/DrCoffey/DeepSqueak>`__
     - Open source
     - Deep-learning detection and classification of USVs
   * - `DAS (Deep Audio Segmenter) <https://github.com/janclemenslab/das>`__
     - Open source
     - Deep-learning annotation of acoustic signals
   * - `USVSEG <https://github.com/rtachi-lab/usvseg>`__
     - Open source
     - Segmentation of rodent ultrasonic vocalisations
   * - `MUPET <https://github.com/mvansegbroeck-zz/mupet>`__
     - Open source
     - Mouse ultrasonic profile extraction and syllable clustering
   * - `VocalMat <https://github.com/ahof1704/VocalMat>`__
     - Open source
     - Detection and classification of mouse vocalisations
   * - `AVA <https://github.com/pearsonlab/autoencoded-vocal-analysis>`__
     - Open source
     - Unsupervised latent-space analysis of vocal repertoires
   * - `Avisoft SASLab <https://avisoft.com/>`__
     - Commercial
     - Bioacoustic recording and sound analysis

Neural data alignment
---------------------

Not behaviour tools themselves, but the packages SimBA output is most often aligned
against when behaviour is paired with neural recording.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `GuPPy <https://github.com/LernerLab/GuPPy>`__
     - Open source
     - Fiber photometry analysis in Python
   * - `pMAT <https://github.com/djamesbarker/pMAT>`__
     - Open source
     - Photometry modular analysis tool
   * - `CaImAn <https://github.com/flatironinstitute/CaImAn>`__
     - Open source
     - Calcium imaging motion correction and source extraction
   * - `suite2p <https://github.com/MouseLand/suite2p>`__
     - Open source
     - Calcium imaging processing and cell detection
   * - `Minian <https://github.com/miniscope/minian>`__
     - Open source
     - Miniscope calcium imaging analysis pipeline

Data standards and interoperability
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `movement <https://github.com/neuroinformatics-unit/movement>`__
     - Open source
     - Analysis of pose tracks from DeepLabCut, SLEAP, and others
   * - `NWB <https://nwb.org/>`__
     - Open source
     - Neurodata Without Borders standard for neurophysiology data
   * - `ndx-pose <https://github.com/rly/ndx-pose>`__
     - Open source
     - NWB extension for storing pose-estimation data

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
   * - `BENTO <https://github.com/neuroethology/bentoMAT>`__
     - Open source
     - Synchronised annotation of behaviour, pose, and neural traces

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
   * - `ANY-maze <https://www.any-maze.com/>`__
     - Commercial
     - Video tracking and automated behavioural testing (Stoelting)
   * - `CleverSys (TopScan, HomeCageScan) <https://cleversysinc.com/>`__
     - Commercial
     - Automated behaviour recognition and home-cage scoring
   * - `Noldus PhenoTyper <https://noldus.com/phenotyper>`__
     - Commercial
     - Instrumented home-cage observation
   * - `Noldus DanioVision <https://noldus.com/daniovision>`__
     - Commercial
     - Zebrafish larva activity tracking
   * - `ViewPoint (ZebraLab) <https://www.viewpoint.fr/>`__
     - Commercial
     - Zebrafish and rodent behaviour tracking
   * - `Panlab SMART <https://www.panlab.com/en/>`__
     - Commercial
     - Video tracking for rodent behavioural tests
   * - `TSE Systems <https://www.tse-systems.com/>`__
     - Commercial
     - Behavioural phenotyping and metabolic systems
   * - `DigiGait <https://mousespecifics.com/digigait/>`__
     - Commercial
     - Treadmill-based gait analysis
