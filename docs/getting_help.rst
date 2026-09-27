🛟 Getting Help
==========================================================

Stuck? Most questions are answered faster by checking the resources below first. If you still
need help, ask on GitHub, Gitter, or by email.

.. raw:: html

   <style>
     .simba-help-h { font-size:17px; font-weight:700; color:#23272e; margin:30px 0 4px; }
     .simba-help-sub { font-size:13.5px; color:#6a7884; margin:0 0 14px; }
     .simba-help-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:18px; margin:0 0 8px; }
     @media (max-width:820px){ .simba-help-grid { grid-template-columns:1fr; } }
     .simba-help-card { display:flex; flex-direction:column; border:1px solid #e1e4e8; border-radius:14px;
        padding:20px 20px 18px; background:#fff; box-shadow:0 4px 16px rgba(0,0,0,.08);
        transition:transform .15s ease, box-shadow .15s ease; }
     .simba-help-card:hover { transform:translateY(-3px); box-shadow:0 12px 30px rgba(33,86,122,.16); }
     .simba-help-card .ico { font-size:26px; line-height:1; margin-bottom:10px; }
     .simba-help-card h3 { margin:0 0 6px !important; font-size:17px; color:#21567a !important; }
     .simba-help-card p { margin:0 0 6px; font-size:13.5px; color:#4b5563; line-height:1.5; }
     .simba-help-card .best { font-size:12.5px; color:#6a7884; }
     .simba-help-card .best b { color:#2f8f9d; }
     .simba-help-card a.go { margin-top:auto; align-self:flex-start; display:inline-flex; align-items:center;
        text-decoration:none !important; font-size:13px; font-weight:600; padding:7px 14px; border-radius:18px;
        background:#21567a; color:#fff !important; transition:background .15s ease; }
     .simba-help-card a.go:hover { background:#19465f; }
     .simba-help-card a.go.alt { background:#fff; color:#21567a !important; border:1px solid #cbd5e1; }
     .simba-help-card a.go.alt:hover { background:#eef5fa; }
     .simba-help-card .spacer { flex:1; min-height:10px; }
     .simba-help-check { border-left:4px solid #21567a; background:#f4f8fb; border-radius:0 12px 12px 0;
        padding:16px 20px 12px; margin:6px 0 10px; }
     .simba-help-check ul { margin:0 0 4px !important; padding-left:20px; }
     .simba-help-check li { font-size:14px; color:#2c3e50; margin:0 0 7px; line-height:1.5; list-style:disc; }
     .simba-help-check code { font-size:12.5px; }
     .simba-help-note { font-size:13px; color:#6a7884; margin:14px 0 0; }
   </style>

   <div class="simba-help-h">1 &middot; Check these first</div>
   <p class="simba-help-sub">Many common problems already have a written answer.</p>
   <div class="simba-help-grid">
     <div class="simba-help-card">
       <div class="ico">❓</div>
       <h3>FAQ</h3>
       <p>Answers to the most common installation, import and classifier questions.</p>
       <div class="spacer"></div>
       <a class="go" href="FAQ.html">Read the FAQ &rarr;</a>
     </div>
     <div class="simba-help-card">
       <div class="ico">📚</div>
       <h3>Tutorials &amp; walkthroughs</h3>
       <p>Step-by-step guides for every part of the SimBA workflow, from project setup to results.</p>
       <div class="spacer"></div>
       <a class="go" href="tutorials.html">Browse tutorials &rarr;</a>
     </div>
     <div class="simba-help-card">
       <div class="ico">🔬</div>
       <h3>Published studies</h3>
       <p>See how other labs used SimBA for similar species, behaviours and experimental set-ups.</p>
       <div class="spacer"></div>
       <a class="go" href="published_studies.html">Browse studies &rarr;</a>
     </div>
   </div>

   <div class="simba-help-h">2 &middot; Ask</div>
   <p class="simba-help-sub">Pick the channel that fits your question.</p>
   <div class="simba-help-grid">
     <div class="simba-help-card">
       <div class="ico">🐞</div>
       <h3>GitHub Issues</h3>
       <p>Report a bug or request a feature. Issues are public and searchable, so someone may already have hit the same problem.</p>
       <p class="best"><b>Best for:</b> errors, crashes, unexpected results, feature requests</p>
       <div class="spacer"></div>
       <a class="go" href="https://github.com/sgoldenlab/simba/issues/new/choose" target="_blank" rel="noopener">Open an issue &rarr;</a>
       <a class="go alt" style="margin-top:8px" href="https://github.com/sgoldenlab/simba/issues?q=is%3Aissue" target="_blank" rel="noopener">Search past issues</a>
     </div>
     <div class="simba-help-card">
       <div class="ico">💬</div>
       <h3>Gitter chat</h3>
       <p>Ask the community and the developers informal questions, and share tips.</p>
       <p class="best"><b>Best for:</b> quick &ldquo;how do I&hellip;?&rdquo; questions and advice on your analysis</p>
       <div class="spacer"></div>
       <a class="go" href="https://app.gitter.im/#/room/#SimBA-Resource_community:gitter.im" target="_blank" rel="noopener">Join the chat &rarr;</a>
     </div>
     <div class="simba-help-card">
       <div class="ico">✉️</div>
       <h3>Email</h3>
       <p>Contact the maintainer directly for things that shouldn&rsquo;t be public.</p>
       <p class="best"><b>Best for:</b> private data and collaborations</p>
       <div class="spacer"></div>
       <a class="go" href="mailto:sronilsson@gmail.com?subject=SimBA%20question">sronilsson@gmail.com &rarr;</a>
     </div>
   </div>

   <div class="simba-help-h">3 &middot; Get a faster answer</div>
   <p class="simba-help-sub">Include these in your question and it can usually be answered in one reply:</p>
   <div class="simba-help-check">
     <ul>
       <li><b>Your SimBA version</b> &mdash; run <code>pip show simba-uw-tf-dev</code></li>
       <li><b>Your operating system and Python version</b></li>
       <li><b>What you did</b> &mdash; which menu or function, and the settings you chose</li>
       <li><b>The full error message</b> &mdash; copy all of the text in the terminal window, not just the last line</li>
       <li><b>Screenshots</b> of the SimBA window, if the problem is visual</li>
       <li><b>Your <code>project_config.ini</code></b> and, if you can share it, a small sample of the data that fails</li>
     </ul>
   </div>
   <p class="simba-help-note">SimBA is maintained on a volunteer basis, so replies can take a few days. Thank you for your patience!</p>
