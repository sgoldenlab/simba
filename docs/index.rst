SimBA
=====================================

.. The landing animation now plays as a lightweight WebM in the hero banner (see docs/_static/custom.js).

.. note::
   These docs are under active development. For detailed tutorials, code, and more extensive
   documentation, see the `SimBA GitHub repository <https://github.com/sgoldenlab/simba>`_.

.. raw:: html

   <style>
     .simba-btn-row { display:flex; flex-wrap:wrap; justify-content:center; gap:12px; margin:18px 0 4px; }
     .simba-btn-row a {
        display:inline-flex; align-items:center; text-decoration:none !important;
        background:#21567a; color:#fff !important; font-weight:600; font-size:14px;
        padding:10px 18px; border-radius:24px; box-shadow:0 2px 10px rgba(33,86,122,.25);
        transition:background .15s ease, transform .15s ease, box-shadow .15s ease;
     }
     .simba-btn-row a:hover { background:#19465f; transform:translateY(-2px); box-shadow:0 6px 18px rgba(33,86,122,.35); }
   </style>
   <div class="simba-btn-row">
     <a href="https://github.com/sgoldenlab/simba" title="SimBA source code on GitHub">💻&nbsp;&nbsp;GitHub</a>
     <a href="https://pypi.org/project/Simba-UW-tf-dev/" title="SimBA on the Python Package Index">📦&nbsp;&nbsp;PyPI</a>
     <a href="https://www.nature.com/articles/s41593-024-01649-9" title="SimBA in Nature Neuroscience">📄&nbsp;&nbsp;Paper</a>
     <a href="#how-to-cite-simba" title="How to cite SimBA (BibTeX)">📑&nbsp;&nbsp;Cite SimBA</a>
     <a href="https://app.gitter.im/#/room/#SimBA-Resource_community:gitter.im" title="Community support on Gitter">💬&nbsp;&nbsp;Gitter support</a>
     <a href="https://simba-uw-tf-dev.readthedocs.io/en/latest/overview_video_202510.html" title="High-level overview video for behavioral scientists">🎥&nbsp;&nbsp;Watch the overview</a>
   </div>

________________________________

🚀 INSTALLATION
------------------------

To install SimBA from PyPI, run the following (use **Python 3.6**, or 3.10 if necessary):

.. code-block:: bash

    pip install simba-uw-tf-dev

Then launch it by typing ``simba``. For step-by-step setup — conda, Anaconda Navigator, or
video walkthroughs — see the full installation guide:

.. raw:: html

   <div class="simba-btn-row" style="justify-content:center; margin-top:14px;">
     <a href="installation.html" title="pip / conda / Anaconda Navigator / video walkthroughs">⚙️&nbsp;&nbsp;Full installation guide</a>
     <a href="https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md" title="Installation guide on GitHub">📖&nbsp;&nbsp;Install guide on GitHub</a>
   </div>

____________________________________

📑 HOW TO CITE SIMBA
------------------------

.. include:: cite_simba.rst.inc

____________________________________

MORE INFORMATION
------------------------
Everything in one place — code, API, community, publications, and data:

.. raw:: html

   <div class="simba-pill-row">
     <a href="https://github.com/sgoldenlab/simba" title="Source code on GitHub">💻&nbsp;GitHub</a>
     <a href="https://simba-uw-tf-dev.readthedocs.io/en/latest/api.html" title="Full API reference">📖&nbsp;API docs</a>
     <a href="https://app.gitter.im/#/room/#SimBA-Resource_community:gitter.im" title="Community chat on Gitter">💬&nbsp;Gitter</a>
     <a href="https://www.biorxiv.org/content/10.1101/2020.04.19.049452v2" title="bioRxiv preprint">📄&nbsp;bioRxiv</a>
     <a href="https://www.nature.com/articles/s41593-024-01649-9" title="Nature Neuroscience paper">📰&nbsp;Nature Neuroscience</a>
     <a href="https://osf.io/tmu6y/" title="Open Science Framework data buckets">💾&nbsp;OSF data</a>
     <a href="https://pypi.org/project/Simba-UW-tf-dev/" title="Python Package Index">📦&nbsp;PyPI</a>
   </div>

________________________________________________

.. raw:: html

   <video class="simba-api-book" autoplay loop muted playsinline preload="metadata"
          poster="_static/img/book_simba_poster.jpg"
          style="float:right; width:min(300px,42%); height:auto; margin:0 0 12px 24px; border-radius:10px;"
          aria-label="A SimBA manual opening to a plate of the subject mouse">
     <source src="_static/img/book_simba.webm" type="video/webm">
     <source src="_static/img/book_simba.mp4" type="video/mp4">
   </video>

.. toctree::
   :maxdepth: 3
   :caption: API REFERENCE:

   api

.. toctree::
   :caption: NOTEBOOKS:

   notebooks

.. toctree::
   :maxdepth: 1
   :caption: USER GUIDE / TUTORIALS:

   installation
   tutorials

.. toctree::
   :maxdepth: 1
   :caption: WALKTHROUGHS:

   walkthroughs

.. toctree::
   :maxdepth: 1
   :caption: LABELLING TUTORIALS:

   labelling

.. toctree::
   :maxdepth: 1
   :caption: FAQ:

   ❓ FAQ <FAQ>

.. toctree::
   :maxdepth: 1
   :caption: GALLERY:

   visualization_gallery

.. toctree::
   :maxdepth: 1
   :caption: DOCS:

   📈 Presentations & Docs <docs/workflow>
   glossary
   qr_gallery
   download_stats
   published_studies

.. toctree::
   :maxdepth: 2
   :caption: ABOUT:

   👥 Credits <credits>
   links

.. toctree::
   :maxdepth: 2
   :caption: OTHER:

   🧩 Related Software <simba.related_software>
   📜 License <simba.license>
   📄 Third-Party Notices <simba.notice>

.. raw:: html

   <div class="simba-orbit-foot">
     <video autoplay loop muted playsinline preload="auto"
            aria-label="Rotating SimBA pose-estimation keypoints">
       <source src="_static/img/blackkp_idle_spin_alpha.webm" type="video/webm">
     </video>
   </div>



   


