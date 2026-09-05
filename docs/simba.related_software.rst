Related Software
================

A non-exhaustive map of the animal pose-estimation, tracking, and
behavior-analysis ecosystem that SimBA is commonly used alongside, grouped by
primary purpose. Many tools span several categories; each is listed under its
main use.

Entries marked |simba-import| produce data SimBA reads directly -- pose formats
through its pose-import tools, and frame-wise annotations through its
:doc:`third-party label appenders <simba.third_party_label_appenders>`.


.. |simba-import| raw:: html

   <span class="simba-tag simba-tag--simba">SimBA import</span>

.. |oss| raw:: html

   <span class="simba-tag simba-tag--oss">Open source</span>

.. |comm| raw:: html

   <span class="simba-tag simba-tag--comm">Commercial</span>

.. |free| raw:: html

   <span class="simba-tag simba-tag--free">Freeware</span>

.. contents:: Categories
   :local:
   :depth: 1

.. _Pose estimation:

|:straight_ruler:| **Pose estimation**
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DeepLabCut <https://www.deeplabcut.org/>`__
     - |oss|
     - |simba-import| Markerless 2D/3D pose estimation
   * - `SLEAP <https://sleap.ai/>`__
     - |oss|
     - |simba-import| Multi-animal pose estimation
   * - `DeepPoseKit <https://github.com/jgraving/deepposekit>`__
     - |oss|
     - |simba-import| Pose estimation toolkit
   * - `Lightning Pose <https://github.com/danbider/lightning-pose>`__
     - |oss|
     - Semi-supervised, multi-view pose estimation
   * - `Facemap <https://github.com/MouseLand/facemap>`__
     - |oss|
     - |simba-import| Mouse orofacial tracking and neural prediction
   * - `OpenPose <https://github.com/CMU-Perceptual-Computing-Lab/openpose>`__
     - |oss|
     - Real-time multi-person 2D keypoint detection
   * - `MMPose <https://github.com/open-mmlab/mmpose>`__
     - |oss|
     - General-purpose pose estimation toolbox (OpenMMLab)
   * - `YOLO (Ultralytics) <https://github.com/ultralytics/ultralytics>`__
     - |oss|
     - |simba-import| Real-time keypoint/pose estimation; also object detection and segmentation
   * - `SuperAnimal / DLC Model Zoo <https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html>`__
     - |oss|
     - |simba-import| Pretrained cross-species DeepLabCut models, usable without new labelling
   * - `APT (Animal Part Tracker) <https://kristinbranson.github.io/APT/>`__
     - |oss|
     - |simba-import| Multi-animal body-part tracking with a labelling GUI; exports .trk (Branson lab)
   * - `DeepGraphPose <https://github.com/paninski-lab/deepgraphpose>`__
     - |oss|
     - Pose estimation using graph-based spatiotemporal priors

|:triangular_ruler:| **3D pose estimation**
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DeepFly3D <https://github.com/NeLy-EPFL/DeepFly3D>`__
     - |oss|
     - 3D pose estimation for tethered Drosophila
   * - `Anipose <https://github.com/lambdaloop/anipose>`__
     - |oss|
     - 3D pose estimation from synchronized cameras
   * - `DANNCE <https://github.com/spoonsso/dannce>`__
     - |oss|
     - |simba-import| 3D landmark detection from multi-view video
   * - `OpenMonkeyStudio <https://github.com/OpenMonkeyStudio>`__
     - |oss|
     - Markerless 3D pose estimation for freely moving macaques

|:world_map:| **Tracking and identity**
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `TRex <https://trex.run/>`__
     - |oss|
     - Real-time, identity-preserving tracking
   * - `idtracker.ai <https://idtracker.ai/>`__
     - |oss|
     - Markerless identity tracking for animal groups
   * - `AlphaTracker <https://github.com/ZexinChen/AlphaTracker>`__
     - |oss|
     - Multi-animal tracking, pose, and behavioral clustering
   * - `Tracktor <https://github.com/vivekhsridhar/tracktor>`__
     - |oss|
     - OpenCV-based single- and multi-object tracker
   * - `ToxTrac <https://sourceforge.net/projects/toxtrac/>`__
     - |oss|
     - Fast tracker for one or several animals
   * - `ezTrack <https://github.com/DeniseCaiLab/ezTrack>`__
     - |oss|
     - Blob-based location and freezing tracking
   * - `LiveMouseTracker <https://micecraft.org/lmt>`__
     - |oss|
     - Long-term mouse tracking via RFID and depth cameras
   * - `TrackMate <https://imagej.net/plugins/trackmate/>`__
     - |oss|
     - ImageJ/Fiji object-tracking plugin
   * - `C-Trax <https://ctrax.sourceforge.net/>`__
     - |oss|
     - Tracking of walking flies in groups
   * - `AnimalTA <https://vchiara.eu/index.php/animalta>`__
     - |oss|
     - GUI tracking of multiple animals across varied setups
   * - `BioTracker <https://github.com/BioroboticsLab/biotracker_core>`__
     - |oss|
     - Modular video-tracking framework
   * - `FlyTracker <https://github.com/kristinbranson/FlyTracker>`__
     - |oss|
     - Tracking and feature extraction for interacting flies
   * - `FicTrac <https://github.com/rjdmoore/fictrac>`__
     - |oss|
     - Spherical-treadmill path tracking for tethered insects
   * - `ZebraZoom <https://zebrazoom.org/>`__
     - |oss|
     - Zebrafish larva and adult behaviour tracking
   * - `Stytra <https://github.com/portugueslab/stytra>`__
     - |oss|
     - Zebrafish tracking with closed-loop stimulus control

|:label:| **Supervised behavior classification**
------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `JAABA <https://www.janelia.org/open-science/jaaba>`__
     - |oss|
     - Supervised behavior classification from trajectories
   * - `DeepEthogram <https://github.com/jbohnslav/deepethogram>`__
     - |oss|
     - |simba-import| Supervised behavior classification from raw video
   * - `MARS <https://github.com/neuroethology/MARS>`__
     - |oss|
     - |simba-import| Pose estimation and social behavior classification in mice
   * - `BehaviorDEPOT <https://github.com/DeNardoLab/BehaviorDEPOT>`__
     - |oss|
     - Pose-guided behavior detection and analysis
   * - `LabGym <https://github.com/umyelab/LabGym>`__
     - |oss|
     - Tracking and behavior classification via Mask R-CNN
   * - `SIPEC <https://github.com/damaggu/SIPEC>`__
     - |oss|
     - End-to-end deep-learning behavioral analysis pipeline
   * - `DeepAction <https://github.com/carlwharris/DeepAction>`__
     - |oss|
     - Video-based behaviour classification with confidence-based review
   * - `DLC2Action <https://github.com/amathislab/DLC2action>`__
     - |oss|
     - Deep-learning behaviour segmentation from pose
   * - `OpenLabCluster <https://github.com/shlizee/OpenLabCluster>`__
     - |oss|
     - Active-learning clustering and classification from keypoints
   * - `CBAS <https://github.com/jones-lab-tamu/CBAS>`__
     - |oss|
     - Circadian behavioural analysis suite for long recordings

|:crystal_ball:| **Unsupervised behavior discovery**
----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `B-SOiD <https://github.com/YttriLab/B-SOID>`__
     - |oss|
     - Unsupervised behavior discovery from pose
   * - `A-SOiD <https://github.com/YttriLab/A-SOID>`__
     - |oss|
     - Active-learning behavior classification from pose
   * - `Keypoint-MoSeq <https://github.com/dattalab/keypoint-moseq>`__
     - |oss|
     - Unsupervised behavioral syllable discovery from keypoints
   * - `VAME <https://github.com/EthoML/VAME>`__
     - |oss|
     - Unsupervised behavioral motif discovery from pose
   * - `DeepOF <https://github.com/mlfpm/deepof>`__
     - |oss|
     - Behavioral analysis of DeepLabCut/SLEAP tracking
   * - `TREBA <https://github.com/neuroethology/TREBA>`__
     - |oss|
     - Trajectory embeddings for behavior representation learning
   * - `MotionMapper <https://github.com/gordonberman/MotionMapper>`__
     - |oss|
     - Unsupervised behavioral mapping from postural dynamics
   * - `MoSeq <https://github.com/dattalab/moseq2-app>`__
     - |oss|
     - Depth-video behavioral syllable segmentation

|:frame_with_picture:| **Object detection**
-------------------------------------------

YOLO (Ultralytics) also belongs here; it is listed under `Pose estimation`_ because
that is the capability SimBA supports directly.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `Detectron2 <https://github.com/facebookresearch/detectron2>`__
     - |oss|
     - Object detection and instance segmentation library
   * - `Segment Anything (SAM 2) <https://github.com/facebookresearch/sam2>`__
     - |oss|
     - Promptable image and video segmentation

|:microphone:| **Vocalisation and audio**
-----------------------------------------

Behavioural video is frequently paired with ultrasonic vocalisation (USV) recording;
these tools cover the audio half of that pipeline.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DeepSqueak <https://github.com/DrCoffey/DeepSqueak>`__
     - |oss|
     - Deep-learning detection and classification of USVs
   * - `DAS (Deep Audio Segmenter) <https://github.com/janclemenslab/das>`__
     - |oss|
     - Deep-learning annotation of acoustic signals
   * - `USVSEG <https://github.com/rtachi-lab/usvseg>`__
     - |oss|
     - Segmentation of rodent ultrasonic vocalisations
   * - `MUPET <https://github.com/mvansegbroeck-zz/mupet>`__
     - |oss|
     - Mouse ultrasonic profile extraction and syllable clustering
   * - `VocalMat <https://github.com/ahof1704/VocalMat>`__
     - |oss|
     - Detection and classification of mouse vocalisations
   * - `AVA <https://github.com/pearsonlab/autoencoded-vocal-analysis>`__
     - |oss|
     - Unsupervised latent-space analysis of vocal repertoires
   * - `Avisoft SASLab <https://avisoft.com/>`__
     - |comm|
     - Bioacoustic recording and sound analysis

|:brain:| **Neural data alignment**
-----------------------------------

Not behaviour tools themselves, but the packages SimBA output is most often aligned
against when behaviour is paired with neural recording.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `NeuroMotive <https://www.blackrockmicro.com/neuroscience-research-products/ephys-behavioral-systems/neuromotive-tracking-system/>`__
     - |comm|
     - Video tracking synchronised to Blackrock neural acquisition, for event-aligned and closed-loop analysis
   * - `Implantable telemetry <https://www.datasci.com/telemetry>`__
     - |comm|
     - Implanted physiological telemetry (blood pressure, ECG) in freely moving animals (DSI)
   * - `GuPPy <https://github.com/LernerLab/GuPPy>`__
     - |oss|
     - Fiber photometry analysis in Python
   * - `pMAT <https://github.com/djamesbarker/pMAT>`__
     - |oss|
     - Photometry modular analysis tool
   * - `CaImAn <https://github.com/flatironinstitute/CaImAn>`__
     - |oss|
     - Calcium imaging motion correction and source extraction
   * - `suite2p <https://github.com/MouseLand/suite2p>`__
     - |oss|
     - Calcium imaging processing and cell detection
   * - `Minian <https://github.com/miniscope/minian>`__
     - |oss|
     - Miniscope calcium imaging analysis pipeline

|:jigsaw:| **Data standards and interoperability**
--------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `movement <https://github.com/neuroinformatics-unit/movement>`__
     - |oss|
     - Analysis of pose tracks from DeepLabCut, SLEAP, and others
   * - `NWB <https://nwb.org/>`__
     - |oss|
     - Neurodata Without Borders standard for neurophysiology data
   * - `ndx-pose <https://github.com/rly/ndx-pose>`__
     - |oss|
     - NWB extension for storing pose-estimation data

|:zap:| **Analysis and real-time pipelines**
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DLC-Analyzer <https://github.com/ETHZ-INS/DLCAnalyzer>`__
     - |oss|
     - Analysis and visualization of DeepLabCut output
   * - `DeepLabCut-Live <https://github.com/DeepLabCut/DeepLabCut-live>`__
     - |oss|
     - Real-time pose estimation on live video
   * - `DeepLabStream <https://github.com/SchwarzNeuroconLab/DeepLabStream>`__
     - |oss|
     - Real-time, closed-loop pose-based feedback
   * - `Bonsai <https://bonsai-rx.org/>`__
     - |oss|
     - Visual reactive programming for experiment pipelines
   * - `AMBER-pipeline <https://github.com/lapphe/AMBER-pipeline>`__
     - |oss|
     - Automated rodent maternal-behavior analysis

|:pencil:| **Manual annotation**
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `BORIS <https://www.boris.unito.it/>`__
     - |oss|
     - |simba-import| Event-logging and manual video annotation
   * - `Solomon Coder <https://solomon.andraspeter.com/>`__
     - |free|
     - |simba-import| Manual behavior coding tool
   * - `Noldus Observer XT <https://noldus.com/observer-xt-human>`__
     - |comm|
     - |simba-import| Manual behavior annotation suite
   * - `CVAT <https://github.com/cvat-ai/cvat>`__
     - |oss|
     - Image and video annotation for detection, segmentation, and keypoints
   * - `Label Studio <https://github.com/HumanSignal/label-studio>`__
     - |oss|
     - Multi-type data labeling and annotation platform
   * - `VIA (VGG Image Annotator) <https://www.robots.ox.ac.uk/~vgg/software/via/>`__
     - |oss|
     - Lightweight image, audio, and video annotation tool
   * - `ELAN <https://archive.mpi.nl/tla/elan>`__
     - |oss|
     - Time-aligned annotation of video and audio
   * - `BENTO <https://github.com/neuroethology/bentoMAT>`__
     - |oss|
     - |simba-import| Synchronised annotation of behaviour, pose, and neural traces

.. _Home-cage monitoring:

|:house:| **Home-cage monitoring**
----------------------------------

Systems that record rodents continuously in their home cage, over days to months, rather
than during a scheduled assay in an arena. These are instrument-plus-software platforms
rather than software alone, but they are the most common upstream source of the
long-duration video and activity data that behaviour-classification pipelines are applied
to. Grouped by sensing modality, since that determines both what can be measured and
whether individual animals can be resolved in a socially housed cage.

Video-based
~~~~~~~~~~~

.. rst-class:: simba-modality

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - System
     - Modality
     - Description
   * - `EnVision <https://www.jax.org/envision>`__
     - Video
     - Imaging embedded in Allentown IVC racks; individual behaviour from group-housed mice (The Jackson Laboratory)
   * - `DOME smart lids <https://oldenlabs.com/products/dome-smart-cages/>`__
     - Near-IR video + audio
     - Camera lid retrofitted to existing racks; computer-vision keypoints, multi-animal via purpose-made ear tags (Olden Labs)
   * - `iMouse <https://imouse.info/>`__
     - Video
     - Rack-mounted side-view cameras with proprietary ML; retrofits standard cages (iMouse GmbH)
   * - `PhenoTyper <https://noldus.com/phenotyper>`__
     - IR video
     - Instrumented home cage for up to two rodents; pairs with EthoVision XT (Noldus)
   * - `PhenoRack <https://www.viewpoint.fr/product/rodent/rodents-behavior-monitoring/phenorack>`__
     - Video
     - 24/7 home-cage activity for up to 32 individually housed animals (Viewpoint)
   * - `HomeCageScan / PhenoCube <https://cleversysinc.com/CleverSysInc/csi_products/homecagescan/>`__
     - Video
     - Automated recognition of unconstrained home-cage behaviours; individually housed (CleverSys)
   * - `BlackBox <https://www.blackboxbio.com/>`__
     - Video
     - Postural dynamics, gait and weight distribution from video; single-housed (BlackBox Bio)
   * - `Trackpaw <https://trackpaw.se/>`__
     - Video
     - Non-invasive in-cage weight, activity and respiratory metrics in mice (TrackPaw Scientific)
   * - `BioSyft <https://biosyft.io/>`__
     - Video
     - Automated behavioural analytics for preclinical rodent research (BioSyft)
   * - `SmartCage System <https://maze.conductscience.com/>`__
     - Video + IR beams
     - All-in-one automated home-cage recording and activity monitoring (MazeEngineers)

Individual ID in group housing (RFID)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rst-class:: simba-modality

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - System
     - Modality
     - Description
   * - `Home Cage Analyser (HCA) <https://www.actualanalytics.com/>`__
     - RFID + IR video
     - Individual metrics from socially housed mice or rats; three animals optimal (Actual Analytics)
   * - `IntelliCage <https://www.tse-systems.com/products/intellicage/>`__
     - RFID + nose-poke/lick
     - Home-cage behaviour and cognitive testing for up to 16 mice or 8 rats (TSE Systems)
   * - `ColonyRack <https://www.phenosys.com/behavior-research/monitoring-tracking/>`__
     - RFID
     - Rack-scale tracking of individual animals in large, semi-natural groups (PhenoSys)
   * - `M3 MultiMouseMonitor <https://www.phenosys.com/behavior-research/monitoring-tracking/>`__
     - RFID
     - Real-time position of individual animals within a group cage (PhenoSys)
   * - `UID Mouse Matrix <https://www.uidevices.com/home-cage-monitoring/>`__
     - RFID
     - Continuous temperature, locomotor activity and zone preference per animal (Unified Information Devices)
   * - `AnyCage Lite <https://www.uidevices.com/anycage-lite/>`__
     - RFID
     - Continuous non-invasive body temperature via UCT-2112 microchips; single-cage and small-cohort studies (Unified Information Devices)
   * - `HomeLab <https://www.neurocage.com/homelab>`__
     - RFID + IoT
     - Unique animal IDs tracked across connected cages; mice (NeuroCage)
   * - `UCT-2112 temperature microchip <https://www.uidevices.com/laboratory-animal-temperature/>`__
     - Passive RFID
     - Implantable chip returning ID and body temperature on a single scan; a component rather than a system (Unified Information Devices)

Cage-level activity and environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rst-class:: simba-modality

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - System
     - Modality
     - Description
   * - `DVC (Digital Ventilated Cage) <https://digitalcage-tecniplast.com/>`__
     - Electronic sensing board
     - 24/7 locomotion and environmental metrics read from the rack itself, at cage level (Tecniplast)
   * - `LABORAS <https://www.metris.nl/en/products/laboras/laboras_information/>`__
     - Vibration / force
     - Behaviour classified from vibration and force signals for up to eight solitary animals (Metris)
   * - `InfraMot <https://www.tse-systems.com/products/phenomaster/>`__
     - Passive IR
     - Total activity from radiated body heat via a lid sensor; now a PhenoMaster NG module (TSE Systems)
   * - `Mouse E-Motion <https://infra-e-motion.de/en/>`__
     - IR
     - Single-animal movement plus cage temperature, humidity and light (INFRA-E-MOTION)
   * - `Actimeter <https://www.imetronic.com/devices/actimeter/>`__
     - IR
     - Locomotor activity, circadian rhythm and novelty reactivity; individually housed (Imetronic)
   * - `Activity Cage <https://ugobasile.com/products/47105-activity-cage>`__
     - IR beam frame
     - Horizontal and vertical (rearing) activity for individuals or groups (Ugo Basile)
   * - `Pallidus MR1 <https://store.mcci.com/products/pallidus-smart-sensor>`__
     - Wireless cage sensor
     - Cage-level temperature, light, humidity and activity reported over LoRaWAN (Pallidus Sensing)
   * - `MOSHERS <https://nc3rs.org.uk/our-portfolio/mouse-smart-hoppers-moshers>`__
     - Camera + AI hopper
     - Individual food intake in group-housed mice; NC3Rs CRACK IT challenge (Research Devices)
   * - `SmartWaiter <https://www.cibertec.es/en/>`__
     - Automatic feeder
     - Feeding and drinking patterns at cage level (Cibertec)

Metabolic phenotyping cages
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rst-class:: simba-modality

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - System
     - Modality
     - Description
   * - `PhenoMaster NG <https://www.tse-systems.com/products/phenomaster/>`__
     - Calorimetry + IR frames
     - Indirect calorimetry with activity, feeding and drinking; single or RFID-based social housing (TSE Systems)
   * - `Promethion <https://www.sablesys.com/products/promethion-line/>`__
     - Indirect calorimetry
     - Continuous oxygen, carbon dioxide, water vapour, methane and stable-isotope measurement in rodents (Sable Systems)
   * - `Promethion Core <https://www.sablesys.com/products/promethion-core-line/>`__
     - Calorimetry + behaviour
     - Metabolic data synchronised to behavioural events; single-housed (Sable Systems)
   * - `Oxymax-CLAMS <https://www.colinst.com/products/oxymax-clams>`__
     - Indirect calorimetry
     - Energy expenditure in mice and rats; single or group housing depending on cage type (Columbus Instruments)


|:briefcase:| **Commercial platforms**
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `Ethovision XT <https://www.noldus.com/ethovision>`__
     - |comm|
     - |simba-import| Video tracking and analysis
   * - `CatWalk XT <https://www.noldus.com/catwalk>`__
     - |comm|
     - Gait analysis platform
   * - `ANY-maze <https://www.any-maze.com/>`__
     - |comm|
     - Video tracking and automated behavioural testing (Stoelting)
   * - `CleverSys TopScan <https://cleversysinc.com/>`__
     - |comm|
     - Automated behaviour recognition in arena assays; see `Home-cage monitoring`_ for HomeCageScan
   * - `Noldus DanioVision <https://noldus.com/daniovision>`__
     - |comm|
     - Zebrafish larva activity tracking
   * - `ViewPoint (ZebraLab) <https://www.viewpoint.fr/>`__
     - |comm|
     - Zebrafish and rodent behaviour tracking; see `Home-cage monitoring`_ for PhenoRack
   * - `Panlab SMART <https://www.panlab.com/en/>`__
     - |comm|
     - Video tracking for rodent behavioural tests
   * - `TSE Systems <https://www.tse-systems.com/>`__
     - |comm|
     - Conditioning, treadmill and inhalation systems; see `Home-cage monitoring`_ for IntelliCage, PhenoMaster and InfraMot
   * - `DigiGait <https://mousespecifics.com/digigait/>`__
     - |comm|
     - Treadmill-based gait analysis
