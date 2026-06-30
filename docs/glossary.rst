📖 Glossary
============

Common terms used throughout the SimBA documentation. Terms defined here can be
cross-referenced from anywhere in the docs with the ``:term:`` role
(e.g. ``:term:`ROI``` renders as :term:`ROI`).

.. raw:: html

   <style>
   #gloss-tools{position:sticky;top:0;z-index:5;background:#fff;padding:12px 0 10px;margin:0 0 10px;border-bottom:1px solid #e2e8f0;}
   #gloss-search{width:100%;box-sizing:border-box;font-size:15px;padding:9px 12px;border:1px solid #cbd5e1;border-radius:8px;}
   #gloss-search:focus{outline:none;border-color:#2a7fb8;box-shadow:0 0 0 3px rgba(42,127,184,.15);}
   #gloss-az{display:flex;flex-wrap:wrap;gap:2px;margin-top:9px;}
   #gloss-az a{font-size:12px;font-weight:700;color:#2a7fb8;text-decoration:none;padding:2px 7px;border-radius:5px;line-height:1.4;}
   #gloss-az a:hover{background:#eaf3fa;}
   #gloss-az a.disabled{color:#cbd5e1;pointer-events:none;}
   #gloss-count{font-size:12px;color:#6b7280;margin-top:7px;}
   #gloss-empty{display:none;color:#6b7280;font-style:italic;margin:14px 0;}
   dl.glossary dt{scroll-margin-top:130px;}
   dl.glossary dt.gloss-grp-term{display:inline;}
   dl.glossary dt.gloss-syn .headerlink{display:none;}
   dl.glossary dt.gloss-syn::after{content:",\00a0";font-weight:inherit;color:inherit;}
   mark.gloss-hl{background:#fde68a;color:inherit;padding:0 1px;border-radius:2px;}
   #gloss-top{position:fixed;right:22px;bottom:22px;z-index:20;width:42px;height:42px;border:none;border-radius:50%;background:#2a7fb8;color:#fff;font-size:20px;line-height:42px;text-align:center;cursor:pointer;box-shadow:0 3px 10px rgba(0,0,0,.25);opacity:0;pointer-events:none;transition:opacity .2s;}
   #gloss-top.show{opacity:.92;pointer-events:auto;}
   #gloss-top:hover{opacity:1;background:#21567a;}
   </style>
   <div id="gloss-tools">
     <input id="gloss-search" type="search" placeholder="Filter terms… — press / to focus (e.g. ROI, classifier, angle)" aria-label="Filter glossary terms">
     <div id="gloss-az"></div>
     <div id="gloss-count"></div>
   </div>
   <p id="gloss-empty">No terms match your filter.</p>
   <button id="gloss-top" title="Back to top" aria-label="Back to top">&#8593;</button>
   <script>
   (function(){
     function init(){
       var dl=document.querySelector('dl.glossary');
       if(!dl){return;}
       var entries=[],cur=null,kids=dl.children,i;
       for(i=0;i<kids.length;i++){
         var el=kids[i];
         if(el.tagName==='DT'){
           if(!cur||cur.closed){cur={dts:[],dd:null,closed:false};entries.push(cur);}
           cur.dts.push(el);
         }else if(el.tagName==='DD'){
           if(cur){cur.dd=el;cur.closed=true;}
         }
       }
       entries.forEach(function(e){
         var terms=e.dts.map(function(d){return (d.textContent||'').replace(//g,'').trim();});
         e.terms=terms;
         e.text=(terms.join(' ')+' '+(e.dd?e.dd.textContent:'')).toLowerCase();
         var fl=(terms[0]||'#').charAt(0).toUpperCase();
         e.letter=/[A-Z]/.test(fl)?fl:'#';
       });
       // synonym terms: when several terms share one definition (Sphinx places it under the LAST
       // term, leaving the earlier ones looking like orphaned headings). Render the group as one
       // combined heading line, e.g. "annotation, labelling", followed by the single definition.
       entries.forEach(function(e){
         if(e.dts.length<2){return;}
         e.dts.forEach(function(d,idx){
           d.classList.add('gloss-grp-term');
           if(idx<e.dts.length-1){d.classList.add('gloss-syn');}
         });
       });
       var az=document.getElementById('gloss-az'),present={};
       entries.forEach(function(e){if(!present[e.letter]){present[e.letter]=e;}});
       'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').forEach(function(L){
         var a=document.createElement('a');a.textContent=L;
         if(present[L]&&present[L].dts[0].id){a.href='#'+present[L].dts[0].id;}
         else{a.className='disabled';}
         az.appendChild(a);
       });
       var countEl=document.getElementById('gloss-count'),emptyEl=document.getElementById('gloss-empty'),total=entries.length;
       function clearMarks(){
         var marks=dl.querySelectorAll('mark.gloss-hl'),k;
         for(k=marks.length-1;k>=0;k--){var m=marks[k],p=m.parentNode;p.replaceChild(document.createTextNode(m.textContent),m);p.normalize();}
       }
       function markEl(el,q){
         var walker=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,null,false),nodes=[],n;
         while((n=walker.nextNode())){nodes.push(n);}
         nodes.forEach(function(node){
           var t=node.nodeValue,lt=t.toLowerCase(),idx=lt.indexOf(q);
           if(idx<0){return;}
           var frag=document.createDocumentFragment(),pos=0;
           while(idx>=0){
             if(idx>pos){frag.appendChild(document.createTextNode(t.slice(pos,idx)));}
             var mk=document.createElement('mark');mk.className='gloss-hl';mk.textContent=t.slice(idx,idx+q.length);
             frag.appendChild(mk);pos=idx+q.length;idx=lt.indexOf(q,pos);
           }
           if(pos<t.length){frag.appendChild(document.createTextNode(t.slice(pos)));}
           node.parentNode.replaceChild(frag,node);
         });
       }
       function update(q){
         q=(q||'').trim().toLowerCase();var shown=0;
         clearMarks();
         entries.forEach(function(e){
           var match=!q||e.text.indexOf(q)>=0;
           e.dts.forEach(function(d){d.style.display=match?'':'none';});
           if(e.dd){e.dd.style.display=match?'':'none';}
           if(match){
             shown++;
             if(q){e.dts.forEach(function(d){markEl(d,q);});if(e.dd){markEl(e.dd,q);}}
           }
         });
         countEl.textContent=q?(shown+' of '+total+' terms'):(total+' terms');
         emptyEl.style.display=(q&&shown===0)?'block':'none';
       }
       var inp=document.getElementById('gloss-search');
       inp.addEventListener('input',function(){update(inp.value);});
       update('');
       // press "/" anywhere to jump to the filter box
       document.addEventListener('keydown',function(ev){
         if(ev.key!=='/'||ev.metaKey||ev.ctrlKey||ev.altKey){return;}
         var ae=document.activeElement,tag=(ae&&ae.tagName||'').toLowerCase();
         if(tag==='input'||tag==='textarea'||(ae&&ae.isContentEditable)){return;}
         ev.preventDefault();inp.focus();
       });
       // floating back-to-top button
       var topBtn=document.getElementById('gloss-top');
       function onScroll(){
         var y=window.scrollY||document.documentElement.scrollTop||0;
         if(y>500){topBtn.classList.add('show');}else{topBtn.classList.remove('show');}
       }
       window.addEventListener('scroll',onScroll,{passive:true});onScroll();
       topBtn.addEventListener('click',function(){window.scrollTo({top:0,behavior:'smooth'});});
     }
     if(document.readyState!=='loading'){init();}
     else{document.addEventListener('DOMContentLoaded',init);}
   })();
   </script>

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

   blob tracking
   contour tracking
      Markerless tracking that segments each animal as a single connected region
      (a *blob*) via background subtraction, instead of tracking individual
      :term:`body-part` keypoints. Useful when full :term:`pose estimation` is
      unnecessary or unavailable.

   third-party annotation tool
      External behavior-annotation software whose frame-by-frame labels SimBA can import and
      append to extracted :term:`features <feature>` as ground-truth :term:`annotation`.
      Supported tools include :term:`BORIS`, :term:`Ethovision`, :term:`Observer`,
      :term:`Solomon`, :term:`DeepEthogram` and :term:`BENTO`.

   BORIS
      Behavioral Observation Research Interactive Software — a free, open-source event-logging
      program for manual behavioral coding; SimBA imports its exported annotations.

   Ethovision
      Noldus EthoVision XT — commercial video-tracking and behavioral-analysis software; SimBA
      can import its exported annotations.

   Observer
      Noldus The Observer XT — commercial event-logging software for manual behavioral
      annotation; importable into SimBA.

   Solomon
      Solomon Coder — a free manual event-logging / ethogram coding tool; SimBA imports its
      coded annotations.

   DeepEthogram
      An open-source supervised deep-learning tool that classifies behaviors directly from raw
      video frames; SimBA can import its predictions as labels.

   BENTO
      A MATLAB GUI (Caltech) for browsing, annotating and analysing synchronised behavioral,
      tracking and neural data; SimBA imports its annotations.

   SuperAnimal-TopView
      A zero-shot, pre-trained top-view mouse :term:`pose estimation` model (from the
      DeepLabCut SuperAnimal family) that SimBA can use without user training.

   FaceMap
      An open-source keypoint/behavioral-motion tracking tool whose output SimBA can
      import as a :term:`pose estimation` source.

   AMBER
      A pose-estimation pipeline for maternal-pup interaction analysis whose tracking
      data SimBA can import.

   arena
      The experimental enclosure (open field, home cage, box, etc.) in which an animal is
      recorded. SimBA maps arena pixels to real-world units via :term:`px/mm` and can restrict
      analyses to it or to :term:`ROIs <ROI>` drawn within it.

   background subtraction
      A markerless segmentation technique that models the static scene and flags pixels
      differing from it as the moving animal; the basis of SimBA's :term:`blob tracking`.

   class imbalance
      The common situation where the :term:`behavior` of interest is present in far fewer
      frames than it is absent, which can bias a :term:`classifier` toward predicting
      "absent". Addressed with re-sampling (see :term:`oversampling`, :term:`undersampling`,
      :term:`SMOTE`).

   oversampling
      Re-balancing training data by duplicating (or synthesising) minority-class
      (:term:`behavior`-present) frames so the :term:`classifier` sees them more often. See
      :term:`class imbalance`.

   undersampling
      Re-balancing training data by randomly dropping majority-class
      (:term:`behavior`-absent) frames to a target ratio of absent-to-present frames. See
      :term:`class imbalance`.

   SMOTE
      Synthetic Minority Over-sampling Technique — generates new synthetic minority-class
      training examples by interpolating between existing ones, rather than plain
      duplication. One of SimBA's :term:`oversampling` options.

   hyperparameters
      The configurable settings of a :term:`classifier` fixed before training — for a
      :term:`random forest`: number of trees, max features, min samples per leaf, split
      criterion. Set per classifier in the SimBA training interface.

   latency
      The time from the start of a session (or a trigger) to the first occurrence of a
      :term:`behavior`; reported per video in SimBA's :term:`aggregate statistics`.

   directing
      An animal orienting toward a target — another animal, a :term:`body-part` or an
      :term:`ROI`. SimBA quantifies directing (see :term:`directionality`) from the angle
      between an animal's heading and the target.

   CLAHE
      Contrast Limited Adaptive Histogram Equalization - a local contrast-enhancement
      step in SimBA's video tools, used to improve tracking on low-contrast footage.

   UMAP
      Uniform Manifold Approximation and Projection - a dimensionality-reduction method
      used in SimBA's unsupervised workflows to embed high-dimensional
      :term:`features <feature>` into 2-D for :term:`clustering` and visualization.

   anchored ROI
   animal-anchored ROI
      An :term:`ROI` (bounding box or shape) attached to, and moving with, an animal or
      :term:`body-part` across frames - as opposed to a fixed, frame-static :term:`ROI`.

   cue light
      An experimentally controlled light stimulus; SimBA's cue-light tools detect when
      each light is on or off and quantify behavior and movement relative to those states.

   spontaneous alternation
      A Y- or T-maze assay of spatial working memory, scored as the tendency to visit
      maze arms in non-repeating sequences. SimBA derives alternation metrics from
      :term:`pose estimation`.
