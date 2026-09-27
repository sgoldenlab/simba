🗂️ Example Datasets
==========================================================

Videos, pose-estimation models, trained classifiers and training data shared by the SimBA team,
grouped by experimental set-up. Use them to follow the tutorials, test SimBA on real data, or as a
starting point for your own classifiers.

.. raw:: html

   <style>
     .simba-ds-notes { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:18px 0 22px; }
     @media (max-width:820px){ .simba-ds-notes { grid-template-columns:1fr; } }
     .simba-ds-note { border-left:4px solid #21567a; background:#f4f8fb; border-radius:0 12px 12px 0;
        padding:12px 16px; font-size:13.5px; color:#2c3e50; line-height:1.5; }
     .simba-ds-note.warn { border-left-color:#c77c1b; background:#fdf6ec; }
     .simba-ds-note b.h { display:block; font-size:14px; margin-bottom:4px; color:#21567a; }
     .simba-ds-note.warn b.h { color:#8a4f0b; }
     .simba-ds-note ol { margin:4px 0 0 !important; padding-left:18px; }
     .simba-ds-note li { margin:0 0 3px; list-style:decimal; }
     .simba-ds-note code { font-size:12px; }
     .simba-ds-grid { display:grid; grid-template-columns:repeat(2, 1fr); gap:20px; margin:0 0 8px; }
     @media (max-width:820px){ .simba-ds-grid { grid-template-columns:1fr; } }
     .simba-ds-card { display:flex; flex-direction:column; border:1px solid #e1e4e8; border-radius:14px;
        padding:20px 22px 16px; background:#fff; box-shadow:0 4px 16px rgba(0,0,0,.08); }
     .simba-ds-card .top { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
     .simba-ds-card .ico { font-size:28px; line-height:1; }
     .simba-ds-card h3 { margin:0 !important; font-size:17.5px; color:#21567a !important; line-height:1.25; }
     .simba-ds-tags { display:flex; flex-wrap:wrap; gap:5px; margin:0 0 10px; }
     .simba-ds-tags span { font-size:11px; font-weight:700; padding:2px 8px; border-radius:9px; background:#e8f4fb; color:#21567a; }
     .simba-ds-tags span.ext { background:#fdf0e6; color:#983412; }
     .simba-ds-card p.d { margin:0 0 10px; font-size:13.5px; color:#4b5563; line-height:1.5; }
     .simba-ds-list { margin:0 0 10px; border-top:1px solid #eef2f5; }
     a.simba-ds-row { display:flex; align-items:baseline; gap:8px; padding:8px 4px; border-bottom:1px solid #eef2f5;
        text-decoration:none !important; transition:background .13s ease; }
     a.simba-ds-row:hover { background:#f3f8fb; }
     a.simba-ds-row .k { flex:0 0 92px; font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.03em; color:#2f8f9d; }
     a.simba-ds-row .n { flex:1; font-size:13.5px; color:#21567a; font-weight:600; }
     a.simba-ds-row .s { flex:0 0 auto; font-size:12px; color:#8b95a1; white-space:nowrap; }
     a.simba-ds-row::after { content:"→"; color:#a7bccc; font-weight:600; }
     .simba-ds-foot { margin-top:auto; font-size:12.5px; color:#6a7884; line-height:1.5; }
     .simba-ds-foot b { color:#4b5563; }
     .simba-ds-foot .warn { color:#8a4f0b; }
   </style>

   <div class="simba-ds-notes">
     <div class="simba-ds-note">
       <b class="h">⬇ Downloading</b>
       Large folders are split into zip volumes (<code>.zip.001</code>, <code>.zip.002</code>, &hellip;).
       <ol>
         <li>Download <b>all</b> volumes in the folder into one directory.</li>
         <li>Extract only the first volume (<code>.zip.001</code>) with <a href="https://www.7-zip.org/" target="_blank" rel="noopener">7-Zip</a>; it reads the others automatically.</li>
       </ol>
     </div>
     <div class="simba-ds-note warn">
       <b class="h">⚠ Classifier compatibility</b>
       The shared classifier files (<code>.sav</code>) were saved with an older scikit-learn. They load in SimBA&rsquo;s
       <b>Python 3.6</b> install (scikit-learn 0.22) but <b>not</b> on Python 3.9+ installs (scikit-learn 1.4).
       On newer installs, retrain from the included training data instead
       (<a href="Scenario3.html">Scenario 3</a>).
     </div>
   </div>

   <div class="simba-ds-grid">

     <div class="simba-ds-card">
       <div class="top"><span class="ico">🐭</span><h3>Mouse resident-intruder</h3></div>
       <div class="simba-ds-tags"><span>Mouse</span><span>2 animals</span><span>Videos</span><span>Pose models</span><span>Classifiers</span></div>
       <p class="d">A resident mouse and an intruder in the home cage, filmed from above. Videos come in RGB, CLAHE-enhanced and greyscale versions, with matching DeepLabCut models for white-vs-black mouse pairs.</p>
       <div class="simba-ds-list">
         <a class="simba-ds-row" href="https://osf.io/sr3ck/files/osfstorage/5e6914630cd06c00b90014b6" target="_blank" rel="noopener"><span class="k">Videos</span><span class="n">Home-cage &amp; CSDS, RGB / CLAHE / greyscale</span><span class="s">67 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/n6zke/" target="_blank" rel="noopener"><span class="k">Pose model</span><span class="n">DeepLabCut, RGB</span><span class="s">~100 MB</span></a>
         <a class="simba-ds-row" href="https://osf.io/yc4ph/" target="_blank" rel="noopener"><span class="k">Pose model</span><span class="n">DeepLabCut, greyscale</span><span class="s">~100 MB</span></a>
         <a class="simba-ds-row" href="https://osf.io/cqd2y/" target="_blank" rel="noopener"><span class="k">Pose model</span><span class="n">DeepLabCut, CLAHE</span><span class="s">~100 MB</span></a>
         <a class="simba-ds-row" href="https://osf.io/7fgwn/" target="_blank" rel="noopener"><span class="k">Labelled images</span><span class="n">Pose training frames, RGB</span><span class="s">39 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/3mc7g/" target="_blank" rel="noopener"><span class="k">Classifiers</span><span class="n">Attack, anogenital sniff, pursuit, tail rattle, lateral threat</span><span class="s">0.75 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/kwge8/files/osfstorage/5f1a53b8320805001515bfb3" target="_blank" rel="noopener"><span class="k">Projects</span><span class="n">Example SimBA projects</span><span class="s">2.3 GB</span></a>
       </div>
       <div class="simba-ds-foot"><b>Use with:</b> <a href="Scenario2.html">Scenario 2</a> (run classifiers on new data)</div>
     </div>

     <div class="simba-ds-card">
       <div class="top"><span class="ico">🐀</span><h3>Rat resident-intruder</h3></div>
       <div class="simba-ds-tags"><span>Rat</span><span>2 animals</span><span>Videos</span><span>Pose model</span><span>Classifiers</span><span>Training data</span></div>
       <p class="d">A resident rat and an intruder in the home cage, filmed in RGB, with a DeepLabCut model, seven trained classifiers and the annotated training data behind them.</p>
       <div class="simba-ds-list">
         <a class="simba-ds-row" href="https://osf.io/sr3ck/files/osfstorage/5e69146e4a60a500acbb7dba" target="_blank" rel="noopener"><span class="k">Videos</span><span class="n">Home-cage, RGB</span><span class="s">5.8 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/2pmqc/" target="_blank" rel="noopener"><span class="k">Pose model</span><span class="n">DeepLabCut, white vs black rat, RGB</span><span class="s">~100 MB</span></a>
         <a class="simba-ds-row" href="https://osf.io/mep5q/" target="_blank" rel="noopener"><span class="k">Classifiers</span><span class="n">Attack, anogenital sniff, lateral threat, submissive, approach, avoidance, boxing</span><span class="s">1.8 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/kwge8/files/osfstorage/5e6ace1f0cd06c017c001dcc" target="_blank" rel="noopener"><span class="k">Training data</span><span class="n">Annotated features (490 per frame)</span><span class="s">2.0 GB</span></a>
       </div>
       <div class="simba-ds-foot"><b>Use with:</b> <a href="Scenario2.html">Scenario 2</a> (run classifiers) &middot; <a href="Scenario3.html">Scenario 3</a> (retrain or improve)</div>
     </div>

     <div class="simba-ds-card">
       <div class="top"><span class="ico">🎥</span><h3>CRIM13 (Caltech Resident-Intruder Mouse)</h3></div>
       <div class="simba-ds-tags"><span>Mouse</span><span>2 animals</span><span class="ext">Raw videos: Caltech</span><span>Pose model</span><span>Classifiers</span><span>Training data</span></div>
       <p class="d">237 pairs of synchronised top- and side-view videos (88 hours, 8 million frames) annotated for 13 social behaviours, from Caltech. SimBA adds a DeepLabCut model, 11 trained classifiers and training data, so you can go from raw video to classified behaviour with public files only.</p>
       <div class="simba-ds-list">
         <a class="simba-ds-row" href="https://doi.org/10.22002/D1.1892" target="_blank" rel="noopener"><span class="k">Videos</span><span class="n">CRIM13 videos &amp; annotations (CaltechDATA, CC-BY)</span><span class="s">250 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/v48xc/" target="_blank" rel="noopener"><span class="k">Pose model</span><span class="n">DeepLabCut, white vs black, CLAHE</span><span class="s">~100 MB</span></a>
         <a class="simba-ds-row" href="https://osf.io/kym42/" target="_blank" rel="noopener"><span class="k">Labelled images</span><span class="n">Pose training frames, CLAHE</span><span class="s">0.8 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/wu398/" target="_blank" rel="noopener"><span class="k">Classifiers</span><span class="n">Approach, attack, chase, circle, clean, copulation, drink, eat, sniff, up, walk away</span><span class="s">13.7 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/kwge8/files/osfstorage/5e6acdc80cd06c017c001d0a" target="_blank" rel="noopener"><span class="k">Training data</span><span class="n">Annotated features (490 per frame)</span><span class="s">16 GB</span></a>
       </div>
       <div class="simba-ds-foot"><b>Cite:</b> the CRIM13 videos are by Burgos-Artizzu, Doll&aacute;r, Lin, Anderson &amp; Perona &mdash; please cite their dataset (<a href="https://doi.org/10.22002/D1.1892" target="_blank" rel="noopener">doi:10.22002/D1.1892</a>).<br><b>Use with:</b> <a href="Scenario2.html">Scenario 2</a> &middot; <a href="Scenario3.html">Scenario 3</a></div>
     </div>

     <div class="simba-ds-card">
       <div class="top"><span class="ico">🍼</span><h3>AMBER: maternal behaviour</h3></div>
       <div class="simba-ds-tags"><span>Rodent</span><span>Dam &amp; pups</span><span>Example video</span><span>Classifiers</span><span>SHAP</span></div>
       <p class="d">Classifiers from the AMBER pipeline for scoring maternal care: active and passive nursing, licking/grooming, nest attendance, dam eating and drinking, and self-directed grooming. Includes an example video and SHAP explanations.</p>
       <div class="simba-ds-list">
         <a class="simba-ds-row" href="https://osf.io/e3dyc/files/osfstorage/6525897c9b0cf3016978746f" target="_blank" rel="noopener"><span class="k">Video</span><span class="n">Example video</span><span class="s">25 MB</span></a>
         <a class="simba-ds-row" href="https://osf.io/e3dyc/files/osfstorage/643812d97078db084eba20d9" target="_blank" rel="noopener"><span class="k">Classifiers</span><span class="n">7 maternal-behaviour classifiers</span><span class="s">7.9 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/e3dyc/" target="_blank" rel="noopener"><span class="k">More</span><span class="n">Zipped models, 30/2 fps variants, SHAP &amp; permutation files</span></a>
       </div>
       <div class="simba-ds-foot"><b>Cite:</b> Lapp, H. E., Salazar, M. G. &amp; Champagne, F. A. (2023). Automated maternal behavior during early life in rodents (AMBER) pipeline. <em>Scientific Reports</em> (<a href="https://doi.org/10.1038/s41598-023-45495-4" target="_blank" rel="noopener">doi:10.1038/s41598-023-45495-4</a>).</div>
     </div>

     <div class="simba-ds-card">
       <div class="top"><span class="ico">🟫</span><h3>Mouse open field</h3></div>
       <div class="simba-ds-tags"><span>Mouse</span><span>1 animal</span><span>Videos</span><span>Pose model</span></div>
       <p class="d">A single C57BL/6J mouse in an open-field arena, filmed in RGB, with a DeepLabCut model. A good starting point for ROI and movement analyses.</p>
       <div class="simba-ds-list">
         <a class="simba-ds-row" href="https://osf.io/sr3ck/files/osfstorage/5e6913d04a60a500aabb5c79" target="_blank" rel="noopener"><span class="k">Videos</span><span class="n">Open field, C57BL/6J, RGB</span><span class="s">3.9 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/j9h26/" target="_blank" rel="noopener"><span class="k">Pose model</span><span class="n">DeepLabCut, black mouse, RGB</span><span class="s">~100 MB</span></a>
       </div>
     </div>

     <div class="simba-ds-card">
       <div class="top"><span class="ico">🔘</span><h3>Mouse operant chamber</h3></div>
       <div class="simba-ds-tags"><span>Mouse</span><span>Videos</span><span>Pose model</span></div>
       <p class="d">A C57BL/6J mouse in an operant chamber during social self-administration, filmed in RGB, with a DeepLabCut model.</p>
       <div class="simba-ds-list">
         <a class="simba-ds-row" href="https://osf.io/sr3ck/files/osfstorage/5e6913ec0cd06c00b90013b7" target="_blank" rel="noopener"><span class="k">Videos</span><span class="n">Operant social self-administration, RGB</span><span class="s">0.2 GB</span></a>
         <a class="simba-ds-row" href="https://osf.io/c9fm7/" target="_blank" rel="noopener"><span class="k">Pose model</span><span class="n">DeepLabCut, black mouse, RGB</span><span class="s">~100 MB</span></a>
       </div>
     </div>

   </div>

All files are hosted on the `SimBA OSF repository <https://osf.io/tmu6y/>`_ unless marked otherwise. Pose-estimation
models for other tools (YOLO, DeepPoseKit, Mask R-CNN) and their labelled images are in the
`tracking models <https://osf.io/sr3ck/>`_ project.
