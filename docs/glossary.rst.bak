📖 Glossary
============

Common terms used throughout the SimBA documentation. Terms defined here can be
cross-referenced from anywhere in the docs with the ``:term:`` role
(e.g. ``:term:`ROI``` renders as :term:`ROI`).

.. glossary::
   :sorted:

   annotation
   labelling
      The process of marking, frame-by-frame, whether a :term:`behavior` is present or
      absent in a video. These human labels are the ground truth used to train a
      :term:`classifier`.

   body-part
      A single tracked point on an animal (e.g. ``nose``, ``left_ear``, ``tail_base``),
      produced by :term:`pose estimation` and stored as ``x``, ``y`` (and probability)
      columns.

   bounding box
      The smallest axis-aligned (or rotated) rectangle that encloses an animal or a set
      of body-parts; used for overlap, area and proximity computations.

   bout
      A continuous, uninterrupted episode of a behavior — i.e. a run of consecutive
      frames classified as the same :term:`behavior`. Bout-level statistics summarise the
      count, duration and timing of these episodes.

   behavior
      target
      The action a SimBA :term:`classifier` is trained to detect (e.g. attack, grooming,
      freezing). Also referred to as the *target*.

   circular statistics
      Statistics for angular/directional data (degrees), where 359° and 1° are close.
      Used in SimBA for heading, turning and directional analyses.

   classifier
      A supervised machine-learning model (typically a random forest) trained on
      annotated :term:`features <feature>` to predict the presence of a :term:`behavior`
      on each frame.

   convex hull
      The smallest convex polygon enclosing a set of body-parts; a common basis for animal
      area, shape and overlap metrics.

   DeepLabCut
      DLC
      A popular open-source :term:`pose estimation` toolbox. SimBA imports DLC tracking
      data (single- and multi-animal).

   directionality
      Whether, and where, an animal is facing — e.g. toward another animal, a body-part or
      a :term:`ROI`.

   egocentric alignment
      Re-centering and rotating each frame so a chosen body-part is fixed in position and
      orientation, removing the animal's global location/heading from the analysis.

   feature
      feature extraction
      A numeric quantity computed per frame from :term:`pose estimation` data (distances,
      velocities, angles, areas, etc.). Feature extraction turns raw tracking into the
      inputs a :term:`classifier` learns from.

   FPS
      Frames per second — the video frame rate. Required to convert frame counts to seconds
      and to compute time-based metrics.

   FSTTC
      Forward Spike Time Tiling Coefficient — a measure of the temporal association between
      two behaviors (how often one tends to follow another within a time window), adapted
      from spike-train analysis.

   Gantt plot
      A timeline visualization showing when each :term:`behavior` occurs across a session
      as horizontal bars.

   heatmap
      A spatial visualization of where an animal spends time (location heatmap) or where a
      :term:`behavior` occurs, binned over the arena.

   interpolation
      Filling in missing body-part coordinates (e.g. dropped/occluded frames) by estimating
      values from neighbouring frames.

   Kleinberg smoothing
      burst detection
      A burst-detection algorithm (Kleinberg, 2003) applied to classifier output to merge
      fragmented detections into coherent :term:`bouts <bout>` and remove noise.

   maDLC
      Multi-animal DeepLabCut — the multi-animal variant of :term:`DeepLabCut`.

   machine results
      The per-video CSV files (in ``project_folder/csv/machine_results``) holding the
      classifier predictions for each frame.

   outlier correction
      Detecting and correcting implausible body-part coordinates (location- and
      movement-based) before feature extraction.

   pose estimation
      Tracking the 2D positions of animal :term:`body-parts <body-part>` across video
      frames, using tools such as :term:`DeepLabCut`, :term:`SLEAP` or YOLO.

   project config
      The ``project_config.ini`` file at the root of a SimBA project, storing all project
      settings (paths, body-parts, classifiers, thresholds).

   px/mm
      pixels per millimeter
      The conversion factor between image pixels and real-world millimetres, used to report
      distances/speeds in physical units. Set per video via a known reference length.

   ROI
      Region of Interest — a user-defined shape (rectangle, circle or polygon) drawn on the
      video frame, used to quantify time spent, entries, movement and directionality within
      specific areas.

   SHAP
      SHapley Additive exPlanations — a model-interpretability method giving each feature a
      contribution score, used in SimBA to explain *why* a :term:`classifier` made a
      prediction.

   SLEAP
      An open-source multi-animal :term:`pose estimation` framework whose output SimBA can
      import.

   smoothing
      Reducing frame-to-frame jitter in tracking data (e.g. Savitzky–Golay or Gaussian) to
      stabilise body-part trajectories.

   video info
      The per-project table (``video_info.csv``) mapping each video to its :term:`FPS`,
      resolution and :term:`px/mm`.

   random forest
      The default supervised algorithm behind a SimBA :term:`classifier`: an ensemble of
      decision trees whose votes give a per-frame :term:`behavior` probability.

   cross-validation
      Splitting annotated data into train/test folds to estimate how well a
      :term:`classifier` generalises to unseen frames, guarding against over-fitting.

   feature importance
      A ranking of how much each :term:`feature` contributes to a :term:`classifier`'s
      decisions (e.g. Gini importance, permutation importance, or :term:`SHAP`).

   precision
      recall
      F1
      Standard classification metrics. Precision = fraction of predicted-positive frames
      that are correct; recall = fraction of true behavior frames detected; F1 = their
      harmonic mean.

   confusion matrix
      A table of predicted vs. true labels (true/false positives and negatives) used to
      evaluate a :term:`classifier`.

   discrimination threshold
      probability threshold
      The probability cut-off above which a frame is scored as the :term:`behavior`.
      Raising it makes detection stricter (higher :term:`precision`), lowering it more
      permissive (higher :term:`recall`).

   minimum bout length
      The shortest allowed :term:`bout` duration; shorter detected episodes are removed as
      noise during post-classification smoothing.

   ethogram
      A catalogue of the distinct behaviors an animal performs, and (in a session) their
      occurrence over time.

   keypoint
      Synonym for :term:`body-part` — a tracked point produced by :term:`pose estimation`.

   p
      pose confidence
      The probability/likelihood score (0–1) that :term:`pose estimation` assigns to each
      tracked :term:`body-part`, indicating tracking reliability.

   occlusion
      When a :term:`body-part` is hidden (by another animal, an object or self) and so is
      poorly tracked or missing — often handled by :term:`interpolation`.

   multi-animal tracking
      identity
      Tracking several animals at once while maintaining each individual's identity across
      frames (and recovering it after :term:`occlusion`), e.g. via :term:`maDLC` or
      :term:`SLEAP`.

   YOLO
      A fast real-time object/keypoint detection model family; SimBA supports YOLO-based
      detection and :term:`pose estimation` workflows.

   geometry
      Representing animals/arenas as shapes (points, lines, :term:`convex hull`\ s,
      polygons, circles) via Shapely, enabling area, overlap, distance and containment
      computations.

   centroid
      The geometric centre of a set of :term:`body-parts <body-part>` (or a shape); often
      used as a single location for an animal.

   velocity
      An animal's speed of movement (distance per unit time), typically derived from the
      frame-to-frame displacement of a :term:`body-part` or :term:`centroid`, in
      :term:`px/mm`-scaled units.

   sliding window
      rolling window
      A fixed-length time window slid across the data to compute time-resolved
      :term:`features <feature>` (e.g. mean velocity over the last 0.5 s).

   time bins
      Dividing a session into fixed-duration intervals (e.g. 60 s) to report how metrics
      change over the course of a recording.

   sequential analysis
      Analysing the order and timing of behaviors — which tend to precede or follow others
      (see :term:`FSTTC`) — to uncover behavioral structure.

   severity scoring
      Grading the intensity of a detected :term:`behavior` (e.g. attack severity) using
      movement/feature-based criteria.

   path plot
      A visualization tracing an animal's movement trajectory through the arena over time.

   validation
      Checking a trained :term:`classifier` on a held-out or new video — including the
      one-click "validation video" with the predicted probability overlaid frame-by-frame.

   aggregate statistics
      Session- or video-level summaries of classifier output (total time, :term:`bout`
      counts, mean bout duration, latency, etc.) saved to the project ``logs``.

   clustering
      embedding
      Unsupervised grouping of behavioral data without labels — e.g. projecting
      :term:`features <feature>` with UMAP/t-SNE and clustering the result to discover
      behavioral motifs.
