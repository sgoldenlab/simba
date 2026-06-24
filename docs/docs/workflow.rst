SimBA Presentations, Workflow, Documents
==========================================

.. raw:: html

   <div class="simba-workflow">

📈 Workflow Diagram
--------------------------
Visual overview of the SimBA analysis pipeline, from video input to classification output.

.. image:: ../_static/img/simba_workflow.png
   :alt: SimBA workflow diagram
   :width: 1000
   :align: center

:download:`📄 Download Workflow (PDF) <../_static/pdf/SimBA_workflow.pdf>`


📚 Presentations & Documents
----------------------------
Posters, slides, publications, and other SimBA resources. Click a thumbnail to open, or use the
download links on each card.

.. raw:: html

   <style>
     .simba-res-grid { display:flex; flex-wrap:wrap; gap:22px; margin:24px 0 6px; }
     .simba-res-card { position:relative; flex:1 1 290px; display:flex; flex-direction:column; border:1px solid #d6e1ea;
        border-radius:12px; overflow:hidden; background:#fff; box-shadow:0 6px 20px rgba(33,86,122,.13);
        transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
     .simba-res-card::before { content:""; position:absolute; top:0; left:0; right:0; height:5px; z-index:2;
        background:linear-gradient(90deg,#21567a,#38a8d4); }
     .simba-res-card:hover { transform:translateY(-6px); border-color:#9fc1d8; box-shadow:0 16px 36px rgba(33,86,122,.30); }
     .simba-res-thumb { display:block; height:158px; background-size:cover; background-position:center top;
        border-bottom:1px solid #eef1f3; }
     .simba-res-body { padding:15px 17px 17px; display:flex; flex-direction:column; flex:1; }
     .simba-res-body h3 { margin:0 0 6px !important; font-size:16px; color:#21567a !important; }
     .simba-res-body p { margin:0 0 13px; font-size:13px; color:#555; line-height:1.5; flex:1; }
     .simba-res-links { display:flex; flex-wrap:wrap; gap:7px; }
     .simba-res-links a { text-decoration:none !important; font-size:12.5px; font-weight:600;
        color:#21567a !important; border:1px solid #cfe0ea; background:#f3f8fb; border-radius:18px;
        padding:4px 11px; transition:background .15s ease; }
     .simba-res-links a:hover { background:#e2eef5; }
   </style>
   <div class="simba-res-grid">

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="../_static/pdf/simba_poster_sam_2.pdf" style="background-image:url('../_static/img/simba_sam_poster_2.webp')"></a>
       <div class="simba-res-body">
         <h3>🧠 Poster &ndash; GRC</h3>
         <p>Poster from the Gordon Research Conference summarizing SimBA&rsquo;s capabilities and applications.</p>
         <div class="simba-res-links"><a href="../_static/pdf/simba_poster_sam_2.pdf">🖼️ PDF</a></div>
       </div>
     </div>

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="https://osf.io/f9ws3/" style="background-image:url('../_static/img/explainability_slide.webp')"></a>
       <div class="simba-res-body">
         <h3>🔍 Explainability Slides &ndash; Winter Brain</h3>
         <p>Slides introducing SimBA&rsquo;s explainability features, presented at the Winter Brain Workshop.</p>
         <div class="simba-res-links"><a href="https://osf.io/f9ws3/">📊 PPTX</a></div>
       </div>
     </div>

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="https://osf.io/y9xj5/" style="background-image:url('../_static/img/goodwin_sfn.webp')"></a>
       <div class="simba-res-body">
         <h3>🧬 Slides &ndash; SfN</h3>
         <p>Society for Neuroscience presentation showcasing SimBA behavior-classification use cases.</p>
         <div class="simba-res-links"><a href="https://osf.io/y9xj5/">📊 PPTX</a></div>
       </div>
     </div>

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="https://www.biorxiv.org/content/10.1101/2020.04.19.049452v2.full.pdf" style="background-image:url('../_static/img/simba_biorxiv_header.webp')"></a>
       <div class="simba-res-body">
         <h3>📄 Preprint &ndash; bioRxiv</h3>
         <p>Original preprint detailing SimBA&rsquo;s architecture, methods, and early validation examples.</p>
         <div class="simba-res-links">
           <a href="https://www.biorxiv.org/content/10.1101/2020.04.19.049452v2.full.pdf">📄 PDF</a>
           <a href="../_static/pdf/Nilsson_etal_2024.pdf">💾 Backup</a>
         </div>
       </div>
     </div>

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="https://static1.squarespace.com/static/5b1b659871069912b3022368/t/666b3f0bae03e61fecdaab73/1718304536255/Goodwin+2024+NN.pdf" style="background-image:url('../_static/img/simba_paper_nn_header.webp')"></a>
       <div class="simba-res-body">
         <h3>📰 Paper &ndash; Nature Neuroscience</h3>
         <p>Peer-reviewed publication in <em>Nature Neuroscience</em> describing SimBA.</p>
         <div class="simba-res-links">
           <a href="https://static1.squarespace.com/static/5b1b659871069912b3022368/t/666b3f0bae03e61fecdaab73/1718304536255/Goodwin+2024+NN.pdf">🔗 PDF</a>
           <a href="../_static/pdf/Goodwin_etal_2024.pdf">💾 Backup</a>
         </div>
       </div>
     </div>

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="https://colab.research.google.com/drive/1x8oBKmSvndSakCsZvpITiNpQY-TDIsae" style="background-image:url('../_static/img/simba_api_example_nb.webp')"></a>
       <div class="simba-res-body">
         <h3>📓 API Example Notebook</h3>
         <p>Hands-on Google Colab notebook walking through the SimBA API and core machine-learning concepts.</p>
         <div class="simba-res-links"><a href="https://colab.research.google.com/drive/1x8oBKmSvndSakCsZvpITiNpQY-TDIsae">🔗 Open in Colab</a></div>
       </div>
     </div>

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="https://osf.io/re4x8" style="background-image:url('../_static/img/simba_qr_codes.webp')"></a>
       <div class="simba-res-body">
         <h3>𝄃𝄃𝄂𝄂𝄀𝄁 QR Codes</h3>
         <p>Printable QR codes linking to key SimBA resources &mdash; handy for posters and presentations.</p>
         <div class="simba-res-links"><a href="https://osf.io/re4x8">🔗 PDF</a></div>
       </div>
     </div>

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="https://osf.io/32vtd/files/ajdz6" style="background-image:url('../_static/img/qr_styled_thumb.webp')"></a>
       <div class="simba-res-body">
         <h3>𝄃𝄃𝄂𝄂𝄀𝄁 Styled QR Codes</h3>
         <p>Branded SimBA QR codes &mdash; neon, gradient, and animated designs. Printable PDF, or browse them all on the <a href="../qr_gallery.html">QR Codes page</a>.</p>
         <div class="simba-res-links"><a href="https://osf.io/32vtd/files/ajdz6">🔗 PDF</a> <a href="../qr_gallery.html">🔳 Gallery</a></div>
       </div>
     </div>

     <div class="simba-res-card">
       <a class="simba-res-thumb" href="https://simba-uw-tf-dev.readthedocs.io/en/latest/overview_video_202510.html" style="background-image:url('../_static/img/overview_video.webp')"></a>
       <div class="simba-res-body">
         <h3>🎥 Overview (2025/10)</h3>
         <p>High-level video overview of SimBA&rsquo;s purpose, use cases, and key features.</p>
         <div class="simba-res-links">
           <a href="https://osf.io/2uack">🔗 MP4</a>
           <a href="https://www.youtube.com/watch?v=oEr2-3Cuhb0">▶ YouTube</a>
           <a href="https://osf.io/32vtd/files/97zc6">📊 Slides</a>
         </div>
       </div>
     </div>

   </div>


🧪 SimBA Published Use Cases
----------------------------------------------------
Spreadsheet collection of real-world independent SimBA applications and validations.

.. warning::
   This list is not actively maintained and may be outdated.

.. raw:: html

   <div style="margin:14px 0 6px 0;height:650px;overflow:hidden;border:1px solid #e1e4e8;border-radius:10px;box-shadow:0 4px 16px rgba(0,0,0,.12);">
     <iframe
       src="https://docs.google.com/spreadsheets/d/169enc3Am2KQKifxj1F9KEKKLbftpMhBlw49zjl-egsY/htmlembed?gid=0"
       width="100%" height="694" frameborder="0"
       style="display:block;margin-top:-44px;border:0;"
       title="SimBA Published Use Cases (Google Sheet)"
       loading="lazy"></iframe>
   </div>

🔗 `Open full-screen in Google Sheets <https://docs.google.com/spreadsheets/d/169enc3Am2KQKifxj1F9KEKKLbftpMhBlw49zjl-egsY/edit?usp=sharing>`__

.. raw:: html

   </div>
