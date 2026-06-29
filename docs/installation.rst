⚙️ Installation
===============================

SimBA is a Python package that you install with ``pip`` into a dedicated **Python 3.6** environment
(Python 3.10 is also supported if needed). There are two ways to get set up — pick the one that fits
how you like to work:

.. raw:: html

   <style>
     .simba-inst-grid { display:flex; flex-wrap:wrap; gap:22px; margin:26px 0 4px; }
     .simba-inst-card { flex:1 1 230px; position:relative; display:block; text-decoration:none !important;
        border:1px solid #e1e4e8; border-radius:12px; padding:26px 24px 22px; background:#fff;
        box-shadow:0 4px 14px rgba(0,0,0,.07); transition:transform .15s ease, box-shadow .15s ease; }
     .simba-inst-card:hover { transform:translateY(-4px); box-shadow:0 12px 28px rgba(33,86,122,.22); }
     .simba-inst-card--static:hover { transform:none; box-shadow:0 4px 14px rgba(0,0,0,.07); }
     .simba-inst-vids { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
     .simba-inst-vids a { text-decoration:none !important; font-size:13px; font-weight:600; color:#21567a !important;
        border:1px solid #cfe0ea; background:#f3f8fb; border-radius:20px; padding:5px 13px; transition:background .15s ease; }
     .simba-inst-vids a:hover { background:#e2eef5; }
     .simba-inst-card::before { content:""; position:absolute; top:0; left:0; right:0; height:5px;
        border-radius:12px 12px 0 0; background:linear-gradient(90deg,#21567a,#38a8d4); }
     .simba-inst-ico { font-size:34px; line-height:1; }
     .simba-inst-card h3 { margin:13px 0 7px; color:#21567a !important; font-size:20px; }
     .simba-inst-card p { margin:0; color:#444; font-size:14.5px; line-height:1.55; }
     .simba-inst-best { display:block; margin-top:13px; font-size:12.5px; color:#5a6b78; }
     .simba-inst-best b { color:#21567a; }
     .simba-inst-go { display:inline-block; margin-top:15px; font-weight:600; color:#21567a !important; font-size:14px; }
     .simba-inst-badge { position:absolute; top:15px; right:15px; background:#21567a; color:#fff; font-size:11px;
        font-weight:600; padding:3px 10px; border-radius:20px; letter-spacing:.3px; }
   </style>
   <div class="simba-inst-grid">
     <a class="simba-inst-card" href="pip_installation.html">
       <span class="simba-inst-badge">Recommended</span>
       <div class="simba-inst-ico">🐍</div>
       <h3>pip / conda &mdash; command line</h3>
       <p>Create a Python&nbsp;3.6 environment and run <code>pip install simba-uw-tf-dev</code> in a
          terminal. Covers the conda, main-Python, and venv routes.</p>
       <span class="simba-inst-best"><b>Best if</b> you're comfortable working in a terminal.</span>
       <span class="simba-inst-go">Open the pip / conda guide &rarr;</span>
     </a>
     <a class="simba-inst-card" href="anaconda_installation.html">
       <div class="simba-inst-ico">🖱️</div>
       <h3>Anaconda Navigator &mdash; GUI</h3>
       <p>Install through the Anaconda Navigator graphical interface &mdash; create the environment and
          install SimBA with point-and-click, no typed commands.</p>
       <span class="simba-inst-best"><b>Best if</b> you'd rather not use a terminal.</span>
       <span class="simba-inst-go">Open the Navigator guide &rarr;</span>
     </a>
     <div class="simba-inst-card simba-inst-card--static">
       <div class="simba-inst-ico">🎬</div>
       <h3>Video walkthroughs</h3>
       <p>Prefer to watch? Follow along with a step-by-step installation video for your chosen route.</p>
       <span class="simba-inst-best"><b>Best if</b> you'd like a guided, visual install.</span>
       <div class="simba-inst-vids">
         <a href="install_conda_video.html">conda &rarr;</a>
         <a href="install_anaconda_navigator_video.html">Anaconda Navigator &rarr;</a>
         <a href="install_venv_video.html">venv &rarr;</a>
       </div>
     </div>
   </div>
   <div style="height:26px;"></div>

.. note::
   Both routes also use **FFmpeg** for SimBA's video pre-processing, editing, and visualization tools.
   It is strongly recommended — see the FFmpeg install links on the :doc:`pip / conda page <pip_installation>`.

.. raw:: html

   <div class="simba-orbit-foot" style="margin:30px 0 8px;">
     <video autoplay loop muted playsinline preload="auto" aria-label="Running SimBA mouse">
       <source src="_static/img/mouse_run_simba_black_2.webm" type="video/webm">
     </video>
   </div>

.. toctree::
   :maxdepth: 1
   :hidden:

   pip_installation
   anaconda_installation
