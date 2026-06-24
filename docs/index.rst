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

If you use SimBA in your research, please cite:

.. raw:: html

   <div class="simba-cite">
     <p class="simba-cite-ref">Goodwin, N. L., Choong, J. J., Hwang, S., <em>et al.</em> (2024).
        Simple Behavioral Analysis (SimBA) as a platform for explainable machine learning in
        behavioral neuroscience. <em>Nature Neuroscience</em>, 27, 1411&ndash;1424.</p>
     <div class="simba-btn-row" style="justify-content:center; margin:8px 0 0;">
       <a href="https://www.nature.com/articles/s41593-024-01649-9" title="Read the paper in Nature Neuroscience">📄&nbsp;&nbsp;Read the paper</a>
       <a href="https://doi.org/10.1038/s41593-024-01649-9" title="DOI: 10.1038/s41593-024-01649-9">🔗&nbsp;&nbsp;DOI</a>
     </div>
   </div>

**BibTeX**

.. code-block:: bibtex

    @article{Goodwin_2024,
      title     = {Simple Behavioral Analysis (SimBA) as a platform for explainable machine learning in behavioral neuroscience},
      author    = {Goodwin, Nastacia L. and Choong, Jia J. and Hwang, Sophia and Pitts, Kayla and Bloom, Liana and Islam, Aasiya and Zhang, Yizhe Y. and Szelenyi, Eric R. and Tong, Xiaoyu and Newman, Emily L. and Miczek, Klaus and Wright, Hayden R. and McLaughlin, Ryan J. and Norville, Zane C. and Eshel, Neir and Heshmati, Mitra and Nilsson, Simon R. O. and Golden, Sam A.},
      journal   = {Nature Neuroscience},
      volume    = {27},
      number    = {7},
      pages     = {1411--1424},
      year      = {2024},
      doi       = {10.1038/s41593-024-01649-9},
      url       = {https://doi.org/10.1038/s41593-024-01649-9},
      publisher = {Springer Science and Business Media LLC}
    }

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
     <a href="https://www.nature.com/articles/s41593-024-01649-9" title="Nature Neuroscience paper">📰&nbsp;Nature paper</a>
     <a href="https://osf.io/tmu6y/" title="Open Science Framework data buckets">💾&nbsp;OSF data</a>
     <a href="https://pypi.org/project/Simba-UW-tf-dev/" title="Python Package Index">📦&nbsp;PyPI</a>
   </div>

________________________________________________

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

   FAQ

.. toctree::
   :maxdepth: 1
   :caption: GALLERY:

   visualization_gallery

.. toctree::
   :maxdepth: 1
   :caption: DOCS:

   docs/workflow
   glossary
   qr_gallery

.. toctree::
   :maxdepth: 2
   :caption: ABOUT:

   credits
   links

.. toctree::
   :maxdepth: 2
   :caption: OTHER:

   simba.related_software
   simba.license



   


