Related Software
================

A non-exhaustive map of the animal pose-estimation, tracking, and behavior-analysis ecosystem that
SimBA is commonly used alongside. It combines a curated list of software and home-cage monitoring
systems with the tools, models, datasets and hardware introduced by papers and preprints that cite
SimBA. Many of those do some or all of what SimBA does with a different approach, species or setup,
so they are worth comparing when choosing a tool; others handle one step of the pipeline, such as
pose estimation or tracking, and can be used alongside SimBA.

Search the table or filter it by section, type and source, and click a row for a fuller description
with code and paper links. Tools marked |simba-import| produce data SimBA reads directly -- pose
formats through its pose-import tools, and frame-wise annotations through its
:doc:`third-party label appenders <simba.third_party_label_appenders>`. Open-source tools whose
GitHub repository has not been updated in two years are tagged *no updates since* (or *archived*),
so check that they still run before relying on them. Home-cage monitoring systems are grouped by
sensing modality, since that determines what can be measured and whether individual animals can be
told apart in a socially housed cage. The table is updated weekly.

.. seealso::

   `The Behaviour Forum <https://www.thebehaviourforum.org/>`__ is a discussion venue for
   experimental methods in animal behaviour, and a good place to ask which of these tools
   suits a given experiment.

   Other catalogues of behavior tools:

   * `OpenBehavior <https://edspace.american.edu/openbehavior/>`__ -- open-source tools for behavioral
     neuroscience research, including hardware and software.
   * `Home Cage Monitoring Technologies Catalog <https://www.thebehaviourdatabase.org/catalogue.php>`__
     (The Behaviour Database) -- home-cage monitoring systems.
   * `Open Neuroscience <https://open-neuroscience.com/>`__ -- a community-curated list of open-source
     neuroscience projects, including behavior hardware and software.

   .. raw:: html
      :file: _generated/related_catalogues_new.html


.. |simba-import| raw:: html

   <span class="simba-tag simba-tag--simba">SimBA import</span>

.. raw:: html
   :file: _generated/related_tools_table.html

.. include:: _generated/related_tools_index.rst


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
