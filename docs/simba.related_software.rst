Related Software
================

A non-exhaustive map of the animal pose-estimation, tracking, and
behavior-analysis ecosystem that SimBA is commonly used alongside, grouped by
primary purpose. Many tools span several categories; each is listed under its
main use.

Entries marked |simba-import| produce data SimBA reads directly -- pose formats
through its pose-import tools, and frame-wise annotations through its
:doc:`third-party label appenders <simba.third_party_label_appenders>`.

.. seealso::

   `The Behaviour Forum <https://www.thebehaviourforum.org/>`__ is a discussion venue for
   experimental methods in animal behaviour, and a good place to ask which of these tools
   suits a given experiment.

   Other catalogues of behavior tools:

   * `OpenBehavior <https://edspace.american.edu/openbehavior/>`__ -- open-source tools for behavioral
     neuroscience research, including hardware and software.
   * `Home Cage Monitoring Technologies Catalog <https://www.thebehaviourdatabase.org/catalogue.php>`__
     (The Behaviour Database) -- home-cage monitoring systems.


.. |simba-import| raw:: html

   <span class="simba-tag simba-tag--simba">SimBA import</span>

.. |oss| raw:: html

   <span class="simba-tag simba-tag--oss">Open source</span>

.. |comm| raw:: html

   <span class="simba-tag simba-tag--comm">Commercial</span>

.. |free| raw:: html

   <span class="simba-tag simba-tag--free">Freeware</span>

Each tool links to its paper where one exists. Open-source tools whose GitHub repository has not
been updated in two years are tagged *no updates since* (or *archived*), so check that they still
run before relying on them.

.. raw:: html
   :file: _generated/related_software_extras.html

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
     - |simba-import| Markerless 2D/3D pose estimation · `paper <https://doi.org/10.1038/s41593-018-0209-y>`__
   * - `SLEAP <https://sleap.ai/>`__
     - |oss|
     - |simba-import| Multi-animal pose estimation · `paper <https://doi.org/10.1038/s41592-022-01426-1>`__
   * - `DeepPoseKit <https://github.com/jgraving/deepposekit>`__
     - |oss|
     - |simba-import| Pose estimation toolkit · `paper <https://doi.org/10.7554/eLife.47994>`__
   * - `Lightning Pose <https://github.com/danbider/lightning-pose>`__
     - |oss|
     - Semi-supervised, multi-view pose estimation · `paper <https://doi.org/10.1038/s41592-024-02319-1>`__
   * - `Facemap <https://github.com/MouseLand/facemap>`__
     - |oss|
     - |simba-import| Mouse orofacial tracking and neural prediction · `paper <https://doi.org/10.1038/s41593-023-01490-6>`__
   * - `OpenPose <https://github.com/CMU-Perceptual-Computing-Lab/openpose>`__
     - |oss|
     - Real-time multi-person 2D keypoint detection · `paper <https://doi.org/10.1109/TPAMI.2019.2929257>`__
   * - `MMPose <https://github.com/open-mmlab/mmpose>`__
     - |oss|
     - General-purpose pose estimation toolbox (OpenMMLab)
   * - `YOLO (Ultralytics) <https://github.com/ultralytics/ultralytics>`__
     - |oss|
     - |simba-import| Real-time keypoint/pose estimation; also object detection and segmentation
   * - `SuperAnimal / DLC Model Zoo <https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html>`__
     - |oss|
     - |simba-import| Pretrained cross-species DeepLabCut models, usable without new labelling · `paper <https://doi.org/10.1038/s41467-024-48792-2>`__
   * - `APT (Animal Part Tracker) <https://kristinbranson.github.io/APT/>`__
     - |oss|
     - |simba-import| Multi-animal body-part tracking with a labelling GUI; exports .trk (Branson lab)
   * - `DeepGraphPose <https://github.com/paninski-lab/deepgraphpose>`__
     - |oss|
     - Pose estimation using graph-based spatiotemporal priors · `paper <https://doi.org/10.1101/2020.08.20.259705>`__

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
     - 3D pose estimation for tethered Drosophila · `paper <https://doi.org/10.7554/eLife.48571>`__
   * - `Anipose <https://github.com/lambdaloop/anipose>`__
     - |oss|
     - 3D pose estimation from synchronized cameras · `paper <https://doi.org/10.1016/j.celrep.2021.109730>`__
   * - `DANNCE <https://github.com/spoonsso/dannce>`__
     - |oss|
     - |simba-import| 3D landmark detection from multi-view video · `paper <https://doi.org/10.1038/s41592-021-01106-6>`__
   * - `OpenMonkeyStudio <https://github.com/OpenMonkeyStudio>`__
     - |oss|
     - Markerless 3D pose estimation for freely moving macaques · `paper <https://doi.org/10.1038/s41467-020-18441-5>`__

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
     - Real-time, identity-preserving tracking · `paper <https://doi.org/10.7554/eLife.64000>`__
   * - `idtracker.ai <https://idtracker.ai/>`__
     - |oss|
     - Markerless identity tracking for animal groups · `paper <https://doi.org/10.1038/s41592-018-0295-5>`__
   * - `AlphaTracker <https://github.com/ZexinChen/AlphaTracker>`__
     - |oss|
     - Multi-animal tracking, pose, and behavioral clustering · `paper <https://doi.org/10.3389/fnbeh.2023.1111908>`__
   * - `Tracktor <https://github.com/vivekhsridhar/tracktor>`__
     - |oss|
     - OpenCV-based single- and multi-object tracker · `paper <https://doi.org/10.1111/2041-210X.13166>`__
   * - `ToxTrac <https://sourceforge.net/projects/toxtrac/>`__
     - |oss|
     - Fast tracker for one or several animals · `paper <https://doi.org/10.1111/2041-210X.12874>`__
   * - `ezTrack <https://github.com/DeniseCaiLab/ezTrack>`__
     - |oss|
     - Blob-based location and freezing tracking · `paper <https://doi.org/10.1038/s41598-019-56408-9>`__
   * - `LiveMouseTracker <https://micecraft.org/lmt>`__
     - |oss|
     - Long-term mouse tracking via RFID and depth cameras · `paper <https://doi.org/10.1038/s41551-019-0396-1>`__
   * - `TrackMate <https://imagej.net/plugins/trackmate/>`__
     - |oss|
     - ImageJ/Fiji object-tracking plugin · `paper <https://doi.org/10.1038/s41592-022-01507-1>`__
   * - `C-Trax <https://ctrax.sourceforge.net/>`__
     - |oss|
     - Tracking of walking flies in groups · `paper <https://doi.org/10.1038/nmeth.1328>`__
   * - `AnimalTA <https://vchiara.eu/index.php/animalta>`__
     - |oss|
     - GUI tracking of multiple animals across varied setups · `paper <https://doi.org/10.1111/2041-210x.14115>`__
   * - `BioTracker <https://github.com/BioroboticsLab/biotracker_core>`__
     - |oss|
     - Modular video-tracking framework · `paper <https://doi.org/10.48550/arxiv.1803.07985>`__
   * - `FlyTracker <https://github.com/kristinbranson/FlyTracker>`__
     - |oss|
     - Tracking and feature extraction for interacting flies · `paper <https://doi.org/10.1007/978-3-319-10605-2_50>`__
   * - `FicTrac <https://github.com/rjdmoore/fictrac>`__
     - |oss|
     - Spherical-treadmill path tracking for tethered insects · `paper <https://doi.org/10.1016/j.jneumeth.2014.01.010>`__
   * - `ZebraZoom <https://zebrazoom.org/>`__
     - |oss|
     - Zebrafish larva and adult behaviour tracking · `paper <https://doi.org/10.3389/fncir.2013.00107>`__
   * - `Stytra <https://github.com/portugueslab/stytra>`__
     - |oss|
     - Zebrafish tracking with closed-loop stimulus control · `paper <https://doi.org/10.1371/journal.pcbi.1006699>`__
   * - `OCTRON <https://octron-tracking.github.io/OCTRON-docs/>`__
     - |oss|
     - Segment-Anything-based segmentation and tracking in a napari GUI · `paper <https://doi.org/10.64898/2025.12.20.695663>`__
   * - `FastTrack <https://www.fasttrack.sh/>`__
     - |oss|
     - Fast 2D tracking of multiple objects, with manual correction · `paper <https://doi.org/10.1371/journal.pcbi.1008697>`__
   * - `Tierpsy Tracker <https://github.com/Tierpsy/tierpsy-tracker>`__
     - |oss|
     - Multi-worm tracking and phenotyping for C. elegans · `paper <https://doi.org/10.1038/s41592-018-0112-1>`__
   * - `Argos <https://argos.readthedocs.io/en/latest/>`__
     - |oss|
     - Tracking multiple animals in cluttered or visually complex environments · `paper <https://doi.org/10.1111/2041-210x.13776>`__

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
     - Supervised behavior classification from trajectories · `paper <https://doi.org/10.1038/nmeth.2281>`__
   * - `DeepEthogram <https://github.com/jbohnslav/deepethogram>`__
     - |oss|
     - |simba-import| Supervised behavior classification from raw video · `paper <https://doi.org/10.7554/eLife.63377>`__
   * - `MARS <https://github.com/neuroethology/MARS>`__
     - |oss|
     - |simba-import| Pose estimation and social behavior classification in mice · `paper <https://doi.org/10.7554/eLife.63720>`__
   * - `BehaviorDEPOT <https://github.com/DeNardoLab/BehaviorDEPOT>`__
     - |oss|
     - Pose-guided behavior detection and analysis · `paper <https://doi.org/10.7554/eLife.74314>`__
   * - `LabGym <https://github.com/umyelab/LabGym>`__
     - |oss|
     - Tracking and behavior classification via Mask R-CNN · `paper <https://doi.org/10.1016/j.crmeth.2023.100415>`__
   * - `SIPEC <https://github.com/SIPEC-Animal-Data-Analysis/SIPEC>`__
     - |oss|
     - End-to-end deep-learning behavioral analysis pipeline · `paper <https://doi.org/10.1038/s42256-022-00477-5>`__
   * - `DeepAction <https://github.com/carlwharris/DeepAction>`__
     - |oss|
     - Video-based behaviour classification with confidence-based review · `paper <https://doi.org/10.1038/s41598-023-29574-0>`__
   * - `DLC2Action <https://github.com/amathislab/DLC2action>`__
     - |oss|
     - Deep-learning behaviour segmentation from pose · `paper <https://doi.org/10.1101/2025.09.27.678941>`__
   * - `OpenLabCluster <https://github.com/shlizee/OpenLabCluster>`__
     - |oss|
     - Active-learning clustering and classification from keypoints · `paper <https://doi.org/10.3389/fnsys.2025.1630654>`__
   * - `CBAS <https://github.com/jones-lab-tamu/CBAS>`__
     - |oss|
     - Circadian behavioural analysis suite for long recordings · `paper <https://doi.org/10.1016/j.crmeth.2025.101050>`__
   * - `JABS <https://github.com/KumarLabJax/JABS-behavior-classifier>`__
     - |oss|
     - Pose pipeline and supervised classifier training for mice (JAX) · `paper <https://doi.org/10.7554/elife.107259>`__
   * - `LISBET <https://github.com/BelloneLab/lisbet>`__
     - |oss|
     - Transformer-based social behaviour segmentation from pose · `paper <https://doi.org/10.48550/arxiv.2311.04069>`__
   * - `FERAL <https://github.com/Skovorp/feral>`__
     - |oss|
     - Supervised behaviour segmentation from raw video using video-understanding models; no pose required · `paper <https://doi.org/10.1101/2025.11.16.688666>`__
   * - `PainFace-Recognition <https://github.com/sofiabroome/painface-recognition>`__
     - |oss|
     - Recurrent-network recognition of equine pain expressions from video · `paper <https://doi.org/10.1109/cvpr.2019.01295>`__

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
     - Unsupervised behavior discovery from pose · `paper <https://doi.org/10.1038/s41467-021-25420-x>`__
   * - `A-SOiD <https://github.com/YttriLab/A-SOID>`__
     - |oss|
     - Active-learning behavior classification from pose · `paper <https://doi.org/10.1038/s41592-024-02200-1>`__
   * - `Keypoint-MoSeq <https://github.com/dattalab/keypoint-moseq>`__
     - |oss|
     - Unsupervised behavioral syllable discovery from keypoints · `paper <https://doi.org/10.1038/s41592-024-02318-2>`__
   * - `VAME <https://github.com/EthoML/VAME>`__
     - |oss|
     - Unsupervised behavioral motif discovery from pose · `paper <https://doi.org/10.1038/s42003-022-04080-7>`__
   * - `DeepOF <https://github.com/mlfpm/deepof>`__
     - |oss|
     - Behavioral analysis of DeepLabCut/SLEAP tracking · `paper <https://doi.org/10.21105/joss.05394>`__
   * - `TREBA <https://github.com/neuroethology/TREBA>`__
     - |oss|
     - Trajectory embeddings for behavior representation learning · `paper <https://doi.org/10.1109/cvpr46437.2021.00290>`__
   * - `MotionMapper <https://github.com/gordonberman/MotionMapper>`__
     - |oss|
     - Unsupervised behavioral mapping from postural dynamics · `paper <https://doi.org/10.1098/rsif.2014.0672>`__
   * - `MoSeq <https://github.com/dattalab/moseq2-app>`__
     - |oss|
     - Depth-video behavioral syllable segmentation · `paper <https://doi.org/10.1016/j.neuron.2015.11.031>`__

|:footprints:| **Kinematics and trajectory analysis**
------------------------------------------------------

Libraries for deriving movement metrics -- speed, turning, path shape, search strategy,
collective structure -- from pose or centroid tracks, rather than classifying behaviour.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Software
     - Type
     - Description
   * - `DLC2Kinematics <https://github.com/AdaptiveMotorControlLab/DLC2Kinematics>`__
     - |oss|
     - Kinematic quantities from DeepLabCut pose output
   * - `traja <https://github.com/traja-team/traja>`__
     - |oss|
     - Trajectory analysis and visualisation built on pandas · `paper <https://doi.org/10.21105/joss.03202>`__
   * - `PyRat <https://github.com/pyratlib/pyrat>`__
     - |oss|
     - Python library for behavioural and trajectory analysis of pose data · `paper <https://doi.org/10.3389/fnins.2022.779106>`__
   * - `animovement <https://animovement.dev/>`__
     - |oss|
     - R toolbox for analysing movement across space and time
   * - `Rtrack <https://rupertoverall.net/Rtrack/index.html>`__
     - |oss|
     - Unbiased search-strategy classification for water-maze and related assays · `paper <https://doi.org/10.1101/2020.02.27.967372>`__
   * - `ColonyTrack <https://rupertoverall.net/ColonyTrack>`__
     - |oss|
     - Analysis of long-term tracking data from group-housed colonies
   * - `swaRm <https://swarm-lab.github.io/swaRm/>`__
     - |oss|
     - R package for individual and collective movement metrics

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
     - Promptable image and video segmentation · `paper <https://doi.org/10.48550/arxiv.2408.00714>`__

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
     - Deep-learning detection and classification of USVs · `paper <https://doi.org/10.1038/s41386-018-0303-6>`__
   * - `DAS (Deep Audio Segmenter) <https://github.com/janclemenslab/das>`__
     - |oss|
     - Deep-learning annotation of acoustic signals · `paper <https://doi.org/10.7554/eLife.68837>`__
   * - `USVSEG <https://github.com/rtachi-lab/usvseg>`__
     - |oss|
     - Segmentation of rodent ultrasonic vocalisations · `paper <https://doi.org/10.1371/journal.pone.0228907>`__
   * - `MUPET <https://github.com/mvansegbroeck-zz/mupet>`__
     - |oss|
     - Mouse ultrasonic profile extraction and syllable clustering · `paper <https://doi.org/10.1016/j.neuron.2017.04.005>`__
   * - `VocalMat <https://github.com/ahof1704/VocalMat>`__
     - |oss|
     - Detection and classification of mouse vocalisations · `paper <https://doi.org/10.7554/elife.59161>`__
   * - `AVA <https://github.com/pearsonlab/autoencoded-vocal-analysis>`__
     - |oss|
     - Unsupervised latent-space analysis of vocal repertoires · `paper <https://doi.org/10.7554/eLife.67855>`__
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
     - Fiber photometry analysis in Python · `paper <https://doi.org/10.1038/s41598-021-03626-9>`__
   * - `pMAT <https://github.com/djamesbarker/pMAT>`__
     - |oss|
     - Photometry modular analysis tool · `paper <https://doi.org/10.1016/j.pbb.2020.173093>`__
   * - `CaImAn <https://github.com/flatironinstitute/CaImAn>`__
     - |oss|
     - Calcium imaging motion correction and source extraction · `paper <https://doi.org/10.7554/eLife.38173>`__
   * - `suite2p <https://github.com/MouseLand/suite2p>`__
     - |oss|
     - Calcium imaging processing and cell detection · `paper <https://doi.org/10.1101/061507>`__
   * - `Minian <https://github.com/miniscope/minian>`__
     - |oss|
     - Miniscope calcium imaging analysis pipeline · `paper <https://doi.org/10.7554/eLife.70661>`__

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
     - Neurodata Without Borders standard for neurophysiology data · `paper <https://doi.org/10.7554/eLife.78362>`__
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
     - Analysis and visualization of DeepLabCut output · `paper <https://doi.org/10.1038/s41386-020-0776-y>`__
   * - `DeepLabCut-Live <https://github.com/DeepLabCut/DeepLabCut-live>`__
     - |oss|
     - Real-time pose estimation on live video · `paper <https://doi.org/10.7554/eLife.61909>`__
   * - `DeepLabStream <https://github.com/SchwarzNeuroconLab/DeepLabStream>`__
     - |oss|
     - Real-time, closed-loop pose-based feedback · `paper <https://doi.org/10.1038/s42003-021-01654-9>`__
   * - `Bonsai <https://bonsai-rx.org/>`__
     - |oss|
     - Visual reactive programming for experiment pipelines · `paper <https://doi.org/10.3389/fninf.2015.00007>`__
   * - `AMBER-pipeline <https://github.com/lapphe/AMBER-pipeline>`__
     - |oss|
     - Automated rodent maternal-behavior analysis · `paper <https://doi.org/10.1038/s41598-023-45495-4>`__
   * - `AmadeusGPT <https://github.com/AdaptiveMotorControlLab/AmadeusGPT>`__
     - |oss|
     - Natural-language querying of behaviour from pose data · `paper <https://doi.org/10.48550/arxiv.2307.04858>`__

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
     - |simba-import| Event-logging and manual video annotation · `paper <https://doi.org/10.1111/2041-210X.12584>`__
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
     - Lightweight image, audio, and video annotation tool · `paper <https://doi.org/10.1145/3343031.3350535>`__
   * - `Labelme <https://github.com/wkentaro/labelme>`__
     - |oss|
     - |simba-import| Polygon, bounding-box, and keypoint image annotation; SimBA converts its labels to YOLO and DeepLabCut training data
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

.. seealso::

   The `Home Cage Monitoring Technologies Catalogue
   <https://www.thebehaviourdatabase.org/catalogue.php>`__, maintained by COST Action
   TEATIME, is a considerably more comprehensive index of these systems than the summary
   below. It is filterable by monitored behaviour, sensing technology, species, cage type,
   DIY versus commercial, and machine-learning support.

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
     - Rack-mounted side-view cameras with proprietary ML; retrofits standard cages (iMouse GmbH) · `paper <https://doi.org/10.1101/2023.08.06.552188>`__
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
   * - `Vium Digital Vivarium <https://recursion.com/news/recursion-acquires-vium-bolstering-its-efforts-to-industrialize-drug-discovery>`__
     - Video + sensors
     - Smart housing with continuous video and computer-vision digital biomarkers; acquired by Recursion in 2020 and no longer sold (Vium)

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
     - Unique animal IDs tracked across connected cages; mice (NeuroCage) · `paper <https://doi.org/10.1371/journal.pone.0220751>`__
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
   * - `BehaviorCloud <https://maze.conductscience.com/portfolio/behaviorcloud-subscription/>`__
     - |comm|
     - Cloud platform for maze-test video tracking, manual scoring and data sharing (Conduct Science)

.. _Tools from papers citing SimBA:

|:books:| **Tools from papers citing SimBA**
--------------------------------------------

Alternative and complementary tools for behavioral analysis: tools, models, datasets and hardware
introduced by published papers and preprints that cite SimBA, compiled from the citation record.
Many of these tools do some or all of what SimBA does, using a different approach, species or
setup, so they are worth comparing when choosing a tool for your experiment. Others handle one step
of the pipeline, such as pose estimation or tracking, and can be used alongside SimBA. The table
is updated weekly. Search or filter by category, and click a row for a short description and code
link.

.. raw:: html
   :file: _generated/related_tools_table.html


.. raw:: html

   <div style="text-align:center; margin:34px 0 6px;">
     <video autoplay loop muted playsinline preload="metadata"
            poster="_static/img/notebook_simba_poster.jpg"
            style="width:min(460px,86%); height:auto;"
            aria-label="A laptop opening to the SimBA splash screen">
       <source src="_static/img/notebook_simba.webm" type="video/webm">
       <source src="_static/img/notebook_simba.mp4" type="video/mp4">
     </video>
   </div>
