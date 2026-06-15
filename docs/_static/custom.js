window.dataLayer = window.dataLayer || [];
function gtag() { dataLayer.push(arguments); }
gtag('js', new Date());
gtag('config', 'G-PEKR9R5J47');

/* ------------------------------------------------------------------ *
 * DOI paper hovercard — hover a doi.org link to preview the paper.
 * Metadata fetched live from the CrossRef API (free, CORS-enabled).
 * ------------------------------------------------------------------ */
(function () {
  var cache = {};
  var card = null;
  var hideTimer = null;

  function makeCard() {
    var c = document.createElement('div');
    c.className = 'doi-hovercard';
    c.style.display = 'none';
    c.addEventListener('mouseenter', function () { clearTimeout(hideTimer); });
    c.addEventListener('mouseleave', hideCard);
    c.addEventListener('click', function (e) {
      var btn = e.target.closest && e.target.closest('.doi-hc-bib');
      if (btn) { e.preventDefault(); copyBibtex(btn); }
    });
    document.body.appendChild(c);
    return c;
  }
  function hideCard() {
    hideTimer = setTimeout(function () { if (card) card.style.display = 'none'; }, 200);
  }
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {          // fallback for non-secure (file://) contexts
      try {
        var ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.focus(); ta.select();
        var ok = document.execCommand('copy');
        document.body.removeChild(ta);
        if (ok) resolve(); else reject(new Error('copy failed'));
      } catch (err) { reject(err); }
    });
  }
  function copyBibtex(btn) {
    var orig = btn.getAttribute('data-label') || btn.textContent;
    btn.setAttribute('data-label', orig);
    var url = btn.getAttribute('data-bib');
    function flash(msg) { btn.textContent = msg; setTimeout(function () { btn.textContent = orig; }, 1500); }
    btn.textContent = 'Fetching…';
    fetch(url, { headers: { 'Accept': 'application/x-bibtex' } })   // DOI content negotiation -> BibTeX
      .then(function (r) { return r.ok ? r.text() : Promise.reject(r.status); })
      .then(function (bib) { return copyText(bib.trim()); })
      .then(function () { flash('Copied!'); })
      .catch(function () { flash('Failed'); });
  }
  function stripTags(s) {
    if (!s) return '';
    var d = document.createElement('div');
    d.innerHTML = s;
    return (d.textContent || '').trim();
  }
  function render(meta, url) {
    var m = meta.message || {};
    var title = (m.title && m.title[0]) || 'Untitled';
    var authors = (m.author || []).map(function (a) {
      return [a.given, a.family].filter(Boolean).join(' ');
    }).filter(Boolean);
    var authStr = authors.length > 4 ? authors.slice(0, 4).join(', ') + ', et al.' : authors.join(', ');
    var journal = (m['container-title'] && m['container-title'][0]) || '';
    var year = (m.issued && m.issued['date-parts'] && m.issued['date-parts'][0] && m.issued['date-parts'][0][0]) || '';
    var abs = stripTags(m.abstract || '');
    if (abs.length > 320) abs = abs.slice(0, 320) + '…';
    return '<div class="doi-hc-title">' + title + '</div>' +
           (authStr ? '<div class="doi-hc-authors">' + authStr + '</div>' : '') +
           '<div class="doi-hc-meta">' + [journal, year].filter(Boolean).join(' · ') + '</div>' +
           (abs ? '<div class="doi-hc-abstract">' + abs + '</div>' : '') +
           '<div class="doi-hc-foot"><span>CrossRef · doi.org</span>' +
           '<span class="doi-hc-actions">' +
           '<button type="button" class="doi-hc-bib" data-bib="' + url + '">Copy BibTeX</button>' +
           '<a class="doi-hc-open" href="' + url + '" target="_blank" rel="noopener">Open ↗</a>' +
           '</span></div>';
  }
  function position(link) {
    var r = link.getBoundingClientRect();
    var maxLeft = window.scrollX + document.documentElement.clientWidth - 380;
    card.style.left = Math.max(8, Math.min(r.left + window.scrollX, maxLeft)) + 'px';
    card.style.top = (r.bottom + window.scrollY + 8) + 'px';
  }
  function show(link) {
    var doi = decodeURIComponent(link.href.replace(/^https?:\/\/doi\.org\//, ''));
    if (!card) card = makeCard();
    clearTimeout(hideTimer);
    position(link);
    card.style.display = 'block';
    if (cache[doi]) { card.innerHTML = cache[doi]; return; }
    card.innerHTML = '<div class="doi-hc-loading">Loading paper details…</div>';
    fetch('https://api.crossref.org/works/' + encodeURIComponent(doi))
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (meta) { var html = render(meta, link.href); cache[doi] = html; card.innerHTML = html; position(link); })
      .catch(function () { card.innerHTML = '<div class="doi-hc-loading">Preview unavailable.</div>'; });
  }
  document.addEventListener('mouseover', function (e) {
    if (!e.target.closest) return;
    if (e.target.closest('.doi-hovercard')) return;   // ignore links inside the card (e.g. the Open button)
    var link = e.target.closest('a[href^="https://doi.org/"]');
    if (link) show(link);
  });
  document.addEventListener('mouseout', function (e) {
    if (!e.target.closest) return;
    if (e.target.closest('.doi-hovercard')) return;   // card manages its own hide via mouseleave
    var link = e.target.closest('a[href^="https://doi.org/"]');
    if (link) hideCard();
  });
})();

/* ------------------------------------------------------------------ *
 * Image lightbox — click a docstring figure to view it full-size.
 * Click the image to toggle zoom (full resolution); click the
 * backdrop, the × , or press ESC to close.
 * ------------------------------------------------------------------ */
(function () {
  var overlay = null, imgEl = null, capEl = null;
  var scale = 1, minScale = 1, maxScale = 8, tx = 0, ty = 0;
  var dragging = false, startX = 0, startY = 0, moved = false;

  function linkIsImageRef(a) {
    // docutils wraps content figures in <a class="image-reference" href="_images/x.png">.
    // That is not a real link, so treat such images as zoomable.
    if (!a) return false;
    if (a.classList && a.classList.contains('image-reference')) return true;
    var href = a.getAttribute('href') || '';
    return /\.(png|jpe?g|webp|gif|svg|bmp|tiff?)($|\?|#)/i.test(href);
  }
  function isZoomable(img) {
    if (!img) return false;
    if (img.closest('.simba-lightbox-overlay')) return false; // not the lightbox image itself
    if (!img.closest('[role="main"], .rst-content')) return false; // content only (skip sidebar/logo)
    var a = img.closest('a');
    if (a && !linkIsImageRef(a)) return false;                // skip only genuine (non-image) links
    var src = img.getAttribute('src') || '';
    if (/logo|favicon|badge|emoji|\bicon/i.test(src)) return false;
    if (img.naturalWidth && img.naturalWidth < 80) return false;   // skip tiny inline icons
    return true;
  }

  function clampScale(v) { return Math.max(minScale, Math.min(maxScale, v)); }
  function apply() {
    imgEl.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
    imgEl.classList.toggle('grabbable', scale > 1.01);
  }
  function reset() { scale = 1; tx = 0; ty = 0; apply(); }
  function zoomAt(clientX, clientY, factor) {
    var rect = imgEl.getBoundingClientRect();
    var cx = clientX - (rect.left + rect.width / 2);   // cursor offset from image centre
    var cy = clientY - (rect.top + rect.height / 2);
    var old = scale;
    scale = clampScale(scale * factor);
    var k = scale / old;
    tx -= cx * (k - 1);                                // keep point under cursor fixed
    ty -= cy * (k - 1);
    if (scale <= 1.0001) { tx = 0; ty = 0; }
    apply();
  }

  function build() {
    overlay = document.createElement('div');
    overlay.className = 'simba-lightbox-overlay';
    var close = document.createElement('div');
    close.className = 'simba-lightbox-close';
    close.innerHTML = '&times;';
    imgEl = document.createElement('img');
    imgEl.className = 'simba-lightbox-img';
    imgEl.draggable = false;
    capEl = document.createElement('div');
    capEl.className = 'simba-lightbox-caption';
    var hint = document.createElement('div');
    hint.className = 'simba-lightbox-hint';
    hint.textContent = 'scroll or + / − to zoom · drag to pan · double-click to reset · Esc to close';
    overlay.appendChild(close);
    overlay.appendChild(imgEl);
    overlay.appendChild(capEl);
    overlay.appendChild(hint);
    document.body.appendChild(overlay);

    close.addEventListener('click', hide);
    overlay.addEventListener('click', function (e) { if (e.target === overlay && !moved) hide(); });
    imgEl.addEventListener('wheel', function (e) {
      e.preventDefault();
      zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });
    imgEl.addEventListener('mousedown', function (e) {
      if (scale <= 1.01) return;
      e.preventDefault(); dragging = true; moved = false; startX = e.clientX - tx; startY = e.clientY - ty;
    });
    window.addEventListener('mousemove', function (e) {
      if (!dragging) return;
      tx = e.clientX - startX; ty = e.clientY - startY; moved = true; apply();
    });
    window.addEventListener('mouseup', function () { dragging = false; setTimeout(function () { moved = false; }, 0); });
    imgEl.addEventListener('dblclick', function (e) {
      e.preventDefault();
      if (scale > 1.01) reset(); else zoomAt(e.clientX, e.clientY, 2.5);
    });
    document.addEventListener('keydown', function (e) {
      if (!overlay || overlay.style.display === 'none') return;
      if (e.key === 'Escape') hide();
      else if (e.key === '+' || e.key === '=') zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1.3);
      else if (e.key === '-' || e.key === '_') zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1 / 1.3);
    });
  }

  function show(img) {
    if (!overlay) build();
    reset();
    imgEl.src = img.currentSrc || img.src;
    capEl.textContent = img.getAttribute('alt') || '';
    overlay.style.display = 'flex';
    requestAnimationFrame(function () { overlay.classList.add('visible'); });
  }
  function hide() {
    if (!overlay) return;
    overlay.classList.remove('visible');
    setTimeout(function () { overlay.style.display = 'none'; }, 150);
  }

  document.addEventListener('click', function (e) {
    var img = e.target.closest && e.target.closest('img');
    if (!isZoomable(img)) return;
    e.preventDefault();
    show(img);
  });
  window.addEventListener('load', function () {
    var imgs = document.querySelectorAll('[role="main"] img, .rst-content img');
    Array.prototype.forEach.call(imgs, function (img) {
      if (isZoomable(img)) img.classList.add('simba-zoomable');
    });
  });
})();

/* ------------------------------------------------------------------ *
 * Video lightbox — hover a sphinxcontrib video, click ⤢ to enlarge.
 * Opens the same source(s) in a big player with native controls.
 * ------------------------------------------------------------------ */
(function () {
  var overlay = null, holder = null;

  function build() {
    overlay = document.createElement('div');
    overlay.className = 'simba-lightbox-overlay simba-video-overlay';
    var close = document.createElement('div');
    close.className = 'simba-lightbox-close';
    close.innerHTML = '&times;';
    holder = document.createElement('div');
    holder.className = 'simba-video-holder';
    overlay.appendChild(close);
    overlay.appendChild(holder);
    document.body.appendChild(overlay);
    close.addEventListener('click', hide);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) hide(); });
    document.addEventListener('keydown', function (e) {
      if (overlay && overlay.style.display !== 'none' && e.key === 'Escape') hide();
    });
  }

  function show(srcVideo) {
    if (!overlay) build();
    holder.innerHTML = '';
    var v = document.createElement('video');
    v.className = 'simba-video-player';
    v.controls = true; v.autoplay = true; v.loop = srcVideo.loop; v.muted = srcVideo.muted; v.playsInline = true;
    var srcs = srcVideo.querySelectorAll('source');
    if (srcs.length) {
      Array.prototype.forEach.call(srcs, function (s) {
        var ns = document.createElement('source');
        ns.src = s.getAttribute('src'); ns.type = s.getAttribute('type') || '';
        v.appendChild(ns);
      });
    } else if (srcVideo.getAttribute('src') || srcVideo.src) {
      v.src = srcVideo.getAttribute('src') || srcVideo.src;   // carousel videos use a src attribute
    }
    holder.appendChild(v);
    overlay.style.display = 'flex';
    requestAnimationFrame(function () { overlay.classList.add('visible'); });
    if (v.play) { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
  }

  function hide() {
    if (!overlay) return;
    var v = holder.querySelector('video'); if (v) v.pause();
    overlay.classList.remove('visible');
    setTimeout(function () { overlay.style.display = 'none'; holder.innerHTML = ''; }, 150);
  }

  window.addEventListener('load', function () {
    var vids = document.querySelectorAll('.sphinx-contrib-video-container video');
    Array.prototype.forEach.call(vids, function (vid) {
      if (vid.parentNode.classList && vid.parentNode.classList.contains('simba-video-wrap')) return;
      var wrap = document.createElement('span');
      wrap.className = 'simba-video-wrap';
      vid.parentNode.insertBefore(wrap, vid);
      wrap.appendChild(vid);
      var btn = document.createElement('button');
      btn.className = 'simba-video-expand';
      btn.type = 'button';
      btn.title = 'Expand video';
      btn.innerHTML = '⤢';
      btn.addEventListener('click', function (e) { e.preventDefault(); e.stopPropagation(); show(vid); });
      wrap.appendChild(btn);
    });
  });
  window.SimbaVideoLightbox = show;     // exposed so the landing carousel can open videos large
})();

/* ------------------------------------------------------------------ *
 * Copy button on code blocks (covers the landing-page install command
 * and every example block). Strips >>> / ... prompts so code copies clean.
 * ------------------------------------------------------------------ */
(function () {
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      try {
        var ta = document.createElement('textarea'); ta.value = text;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.focus(); ta.select();
        var ok = document.execCommand('copy'); document.body.removeChild(ta);
        if (ok) resolve(); else reject(new Error('copy failed'));
      } catch (e) { reject(e); }
    });
  }
  function cleanCode(pre) {
    return pre.innerText.replace(/^\s*(>>>|\.\.\.)\s?/gm, '').replace(/\s+$/, '');
  }
  window.addEventListener('load', function () {
    var blocks = document.querySelectorAll('.rst-content div.highlight, [role="main"] div.highlight');
    Array.prototype.forEach.call(blocks, function (block) {
      if (block.querySelector('.simba-copy-btn')) return;
      var pre = block.querySelector('pre');
      if (!pre) return;
      var btn = document.createElement('button');
      btn.type = 'button'; btn.className = 'simba-copy-btn'; btn.textContent = 'Copy';
      function flash(m) { btn.textContent = m; btn.classList.add('copied'); setTimeout(function () { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1400); }
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        copyText(cleanCode(pre)).then(function () { flash('Copied!'); }).catch(function () { flash('Failed'); });
      });
      block.appendChild(btn);
    });
  });
})();

/* ------------------------------------------------------------------ *
 * Command palette — Ctrl/Cmd-K (or "/") to jump to any documented
 * function/class/method. Data is the Sphinx general index (genindex.html),
 * so results link straight to the exact object anchor.
 * ------------------------------------------------------------------ */
(function () {
  var ROOT = (function () {
    var el = document.getElementById('documentation_options');
    return (el && el.getAttribute('data-url_root')) || '';
  })();
  var entries = null, overlay = null, input = null, list = null, sel = -1, loading = false;

  function loadEntries(cb) {
    if (entries) return cb(entries);
    if (loading) return;
    loading = true;
    fetch(ROOT + 'genindex.html')
      .then(function (r) { return r.text(); })
      .then(function (html) {
        var doc = new DOMParser().parseFromString(html, 'text/html');
        var seen = {}, out = [];
        Array.prototype.forEach.call(doc.querySelectorAll('a[href]'), function (a) {
          var href = a.getAttribute('href');
          if (!href || href.indexOf('#') === -1) return;        // only object deep-links
          var label = a.textContent.replace(/\s+/g, ' ').trim();
          if (!label) return;
          var key = label + '|' + href;
          if (seen[key]) return; seen[key] = 1;
          out.push({ label: label, href: ROOT + href });
        });
        entries = out; cb(entries);
      })
      .catch(function () { entries = []; cb(entries); });
  }
  function escapeHtml(s) { return s.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function score(label, q) {
    var L = label.toLowerCase(); var i = L.indexOf(q);
    if (i === -1) return -1;
    return (i === 0 ? 1000 : (L[i - 1] === '.' || L[i - 1] === ' ' ? 500 : 100)) - i;
  }
  function build() {
    overlay = document.createElement('div'); overlay.className = 'simba-cmdk-overlay';
    var box = document.createElement('div'); box.className = 'simba-cmdk-box';
    input = document.createElement('input'); input.className = 'simba-cmdk-input';
    input.type = 'text'; input.placeholder = 'Jump to a function, class or method…';
    list = document.createElement('div'); list.className = 'simba-cmdk-list';
    box.appendChild(input); box.appendChild(list); overlay.appendChild(box);
    document.body.appendChild(overlay);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });
    input.addEventListener('input', function () { render(input.value); });
    input.addEventListener('keydown', onKey);
  }
  function open() {
    if (!overlay) build();
    overlay.classList.add('visible');
    input.value = ''; sel = -1;
    loadEntries(function () { render(''); });
    setTimeout(function () { input.focus(); }, 0);
  }
  function close() { if (overlay) overlay.classList.remove('visible'); }
  function render(q) {
    q = (q || '').trim().toLowerCase();
    var items;
    if (!q) items = (entries || []).slice(0, 40);
    else items = (entries || []).map(function (e) { return { e: e, s: score(e.label, q) }; })
      .filter(function (x) { return x.s >= 0; })
      .sort(function (a, b) { return b.s - a.s; })
      .slice(0, 40).map(function (x) { return x.e; });
    sel = items.length ? 0 : -1;
    list.innerHTML = items.map(function (e, i) {
      return '<a class="simba-cmdk-item' + (i === 0 ? ' active' : '') + '" href="' + e.href + '">' + escapeHtml(e.label) + '</a>';
    }).join('') || '<div class="simba-cmdk-empty">No matches</div>';
  }
  function itemEls() { return list.querySelectorAll('.simba-cmdk-item'); }
  function setActive(n) {
    var it = itemEls(); if (!it.length) return;
    sel = (n + it.length) % it.length;
    Array.prototype.forEach.call(it, function (el, i) { el.classList.toggle('active', i === sel); });
    it[sel].scrollIntoView({ block: 'nearest' });
  }
  function onKey(e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(sel + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(sel - 1); }
    else if (e.key === 'Enter') { e.preventDefault(); var it = itemEls(); if (it[sel]) window.location.href = it[sel].href; }
    else if (e.key === 'Escape') { close(); }
  }
  document.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase();
    var typing = tag === 'input' || tag === 'textarea' || e.target.isContentEditable;
    if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) { e.preventDefault(); open(); }
    else if (e.key === '/' && !typing) { e.preventDefault(); open(); }
  });
  window.addEventListener('load', function () {
    var base = location.pathname.split('/').pop();
    if (!(base === '' || base === 'index.html')) return;        // landing page only
    var main = document.querySelector('[role="main"]');
    if (!main || main.querySelector('.simba-cmdk-trigger')) return;
    var btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'simba-cmdk-trigger';
    btn.innerHTML = '<span>🔍&nbsp;&nbsp;Search the API — functions, classes, methods…</span><kbd>Ctrl K</kbd>';
    btn.addEventListener('click', open);
    main.insertBefore(btn, main.firstChild);
  });
})();

/* ------------------------------------------------------------------ *
 * Landing page: turn the top-level toctree into a card grid.
 * Non-destructive — inserts a grid near the top; the original TOC stays.
 * ------------------------------------------------------------------ */
(function () {
  function iconFor(title) {
    var m = title.match(/^([^\w\s(]+)\s*/u);          // leading emoji/symbol, if any
    if (m && m[1]) return { icon: m[1], text: (title.slice(m[0].length).trim() || title) };
    var t = title.toLowerCase();
    var map = [['license', '📜'], ['credit', '🙏'], ['software', '🧰'], ['presentation', '🎤'],
               ['faq', '❔'], ['api', '📖'], ['tutorial', '🎓'], ['install', '⚙️'], ['notebook', '📚'],
               ['walkthrough', '🚶'], ['label', '🏷️'], ['link', '🔗']];
    for (var i = 0; i < map.length; i++) if (t.indexOf(map[i][0]) !== -1) return { icon: map[i][1], text: title };
    return { icon: '📄', text: title };
  }
  window.addEventListener('load', function () {
    var base = location.pathname.split('/').pop();
    if (!(base === '' || base === 'index.html')) return;
    var main = document.querySelector('[role="main"]');
    if (!main || main.querySelector('.simba-cardgrid')) return;
    var seen = {}, cards = [];
    Array.prototype.forEach.call(main.querySelectorAll('.toctree-wrapper li.toctree-l1 > a'), function (a) {
      var href = a.getAttribute('href');
      if (!href || seen[href]) return;
      seen[href] = 1;
      var info = iconFor(a.textContent.trim());
      cards.push({ href: href, icon: info.icon, text: info.text });
    });
    if (cards.length < 2) return;
    var grid = document.createElement('div');
    grid.className = 'simba-cardgrid';
    grid.innerHTML = cards.map(function (c) {
      return '<a class="simba-card" href="' + c.href + '">' +
             '<span class="simba-card-ico">' + c.icon + '</span>' +
             '<span class="simba-card-txt">' + c.text.replace(/</g, '&lt;') + '</span></a>';
    }).join('');
    var h1 = main.querySelector('h1');
    if (h1) h1.insertAdjacentElement('afterend', grid);
    else main.insertBefore(grid, main.firstChild);
  });
})();

/* ------------------------------------------------------------------ *
 * Landing page: demo showcase carousel.
 * Edit SLIDES to choose which _static/img/*.webm demos to feature.
 * ------------------------------------------------------------------ */
(function () {
  var SLIDES = [
    'InteractiveROIModifier.webm', 'NetworkMixin.mp4', 'FreezingDetector.webm', 'SkeletonVideoCreator.mp4',
    'CirclingDetector.webm', 'BlobVisualizer.webm', 'GeometryPlotter_1.webm', 'pose_plotter_cuda.mp4',
    'HeatMapperLocationMultiprocess.webm', 'EgocentricalAligner.webm', 'DirectingOtherAnimalsVisualizerMultiprocess.mp4', 'CueLightVisualizer.webm',
    'DirectingROIVisualizer.webm', 'BlobLocationComputer.webm', 'get_convex_hull_cuda.mp4', 'LightDarkBoxPlotter.webm',
    'MitraFeatureExtractor.webm', 'ROIRuler.webm', 'YOLOPoseVisualizer.webm', 'YOLOVisualizer.webm',
    'YoloModelComparator.webm', 'YoloInference_1.webm', 'video_bg_subtraction.webm', 'make_gantt_plot.webm',
    'open_field.webm', 'roi_blurbox.webm', 'sam_example.webm', 'outside_roi_example.mp4',
    'inset_overlay_video.webm', 'overlay_video_progressbar.webm', 'GetPixelsPerMillimeterInterface.webm', 'brightness_intensity.mp4',
    'crossfade_two_videos.webm', 'superimpose_elapsed_time.webm', 'watermark_video.webm', 'video_to_bw.webm',
    'rotate_video.mp4', 'multiframe_union.webm', 'bg_removed_ex_1_clipped.webm', 'bg_remover_example_1.webm',
    'blob_quick_check.webm', 'change_single_video_fps.mp4', 'create_average_frm_1.webm', 'cue_light_example_2.webm',
    'egocentric_nb_1.webm', 'example_annotation.webm', 'flip_videos.webm', 'geometry_example_7_1.webm',
    'get_video_slic.webm', 'label_change_size_img.webm', 'playback_speed.webm', 'reverse_videos.webm',
    'slice_imgs_gpu.webm', 'smoothing_example_2.webm', 'superimpose_freetext.webm', 'superimpose_video_names.webm',
    'T1.webm', 'ts_example.webm', 'vertical_concat.webm', 'video_bg_substraction_mp.webm'
  ];
  var GAP = 16, FRAC = 0.62;        // slide width = 62% of viewport -> neighbours peek on both sides
  function caption(fn) {
    return fn.replace(/\.(webm|mp4)$/, '').replace(/_/g, ' ')
             .replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2');
  }
  window.addEventListener('load', function () {
    var base = location.pathname.split('/').pop();
    if (!(base === '' || base === 'index.html')) return;
    var main = document.querySelector('[role="main"]');
    if (!main || main.querySelector('.simba-carousel')) return;

    var wrap = document.createElement('div'); wrap.className = 'simba-carousel';
    var viewport = document.createElement('div'); viewport.className = 'simba-carousel-viewport';
    var track = document.createElement('div'); track.className = 'simba-carousel-track';
    var cap = document.createElement('div'); cap.className = 'simba-carousel-cap';
    var prev = document.createElement('button'); prev.type = 'button'; prev.className = 'simba-carousel-nav prev'; prev.innerHTML = '‹';
    var next = document.createElement('button'); next.type = 'button'; next.className = 'simba-carousel-nav next'; next.innerHTML = '›';
    var counter = document.createElement('div'); counter.className = 'simba-carousel-counter';
    var hint = document.createElement('div'); hint.className = 'simba-carousel-hint';
    hint.innerHTML = 'Use <b>&#8249;</b> <b>&#8250;</b> keys to browse &nbsp;·&nbsp; click a video to enlarge';
    var progress = document.createElement('div'); progress.className = 'simba-carousel-progress';
    var bar = document.createElement('i'); progress.appendChild(bar);
    wrap.tabIndex = 0;                                    // focusable -> arrow-key navigation
    viewport.appendChild(track);
    wrap.appendChild(viewport); wrap.appendChild(cap); wrap.appendChild(prev); wrap.appendChild(next);
    wrap.appendChild(counter); wrap.appendChild(hint); wrap.appendChild(progress);

    var slides = SLIDES.map(function (fn, i) {
      var slide = document.createElement('div'); slide.className = 'simba-carousel-slide';
      var v = document.createElement('video');
      v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'none';
      v.setAttribute('data-src', '_static/img/' + fn);   // lazy: real src is set only near the active slide
      slide.appendChild(v);
      slide.addEventListener('click', function () { if (suppressClick) return; if (i === idx) openVid(v); else go(i); });
      track.appendChild(slide);
      return { slide: slide, v: v };
    });
    function openVid(video) { if (window.SimbaVideoLightbox) window.SimbaVideoLightbox(video); }
    function loadAround(n) {                              // load active +/- 2 only
      for (var d = -2; d <= 2; d++) {
        var v = slides[(n + d + slides.length) % slides.length].v;
        if (!v.src && v.getAttribute('data-src')) v.src = v.getAttribute('data-src');
      }
    }

    var idx = 0, timer = null, slideW = 0, suppressClick = false;
    function position(animate) {
      track.style.transition = (animate === false) ? 'none' : 'transform .45s ease';
      var activeCenter = idx * (slideW + GAP) + slideW / 2;
      track.style.transform = 'translateX(' + (viewport.clientWidth / 2 - activeCenter) + 'px)';
      loadAround(idx);
      slides.forEach(function (s, i) {
        var on = i === idx;
        s.slide.classList.toggle('active', on);
        if (on) { try { s.v.currentTime = 0; } catch (e) {} var p = s.v.play(); if (p && p.catch) p.catch(function () {}); }
        else s.v.pause();
      });
      cap.textContent = caption(SLIDES[idx]);
      counter.textContent = (idx + 1) + ' / ' + slides.length;
    }
    function layout() {
      slideW = Math.max(220, Math.round(viewport.clientWidth * FRAC));
      slides.forEach(function (s) { s.slide.style.width = slideW + 'px'; });
      position(false);
    }
    var INTERVAL = 7000, hovered = false;
    function startBar() {
      bar.style.transition = 'none'; bar.style.width = '0%';
      void bar.offsetWidth;                              // reflow so the next transition animates
      bar.style.transition = 'width ' + INTERVAL + 'ms linear'; bar.style.width = '100%';
    }
    function freezeBar() {
      var w = getComputedStyle(bar).width;
      bar.style.transition = 'none'; bar.style.width = w;
    }
    function schedule() {
      clearTimeout(timer); startBar();
      timer = setTimeout(function () { show(idx + 1); }, INTERVAL);
    }
    function show(n) { idx = (n + slides.length) % slides.length; position(true); schedule(); }
    function go(n) { show(n); }
    prev.addEventListener('click', function (e) { e.stopPropagation(); go(idx - 1); });
    next.addEventListener('click', function (e) { e.stopPropagation(); go(idx + 1); });
    wrap.addEventListener('mouseenter', function () { hovered = true; clearTimeout(timer); freezeBar(); });
    wrap.addEventListener('mouseleave', function () { hovered = false; schedule(); });
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      if (!(hovered || wrap.contains(document.activeElement))) return;   // only when in focus/hover
      var tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
      e.preventDefault();
      go(e.key === 'ArrowLeft' ? idx - 1 : idx + 1);
    });
    var touchX = 0, touchY = 0, swiping = false;
    viewport.addEventListener('touchstart', function (e) {
      var t = e.touches[0]; touchX = t.clientX; touchY = t.clientY; swiping = true;
      clearTimeout(timer); freezeBar();
    }, { passive: true });
    viewport.addEventListener('touchend', function (e) {
      if (!swiping) return; swiping = false;
      var t = e.changedTouches[0], dx = t.clientX - touchX, dy = t.clientY - touchY;
      if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy)) {     // horizontal swipe
        suppressClick = true; setTimeout(function () { suppressClick = false; }, 350);
        go(dx < 0 ? idx + 1 : idx - 1);
      } else {
        schedule();                                              // tap / vertical scroll -> resume autoplay
      }
    }, { passive: true });
    window.addEventListener('resize', layout);

    var inst = main.querySelector('#installation');
    var mi = main.querySelector('#more-information');
    var miList = mi && mi.querySelector('ul');
    if (inst) {
      inst.insertAdjacentElement('afterend', wrap);     // just below the INSTALLATION section
    } else if (miList) {
      miList.insertAdjacentElement('afterend', wrap);
    } else {
      var grid = main.querySelector('.simba-cardgrid');
      var h1 = main.querySelector('h1');
      if (grid) grid.parentNode.insertBefore(wrap, grid);
      else if (h1) h1.insertAdjacentElement('afterend', wrap);
      else main.insertBefore(wrap, main.firstChild);
    }
    layout(); schedule();
    setTimeout(layout, 400);          // re-center once videos report dimensions
  });
})();

/* ------------------------------------------------------------------ *
 * Copy LaTeX on display-math blocks.
 * Uses the MathJax v3 math list (the raw TeX is gone from the DOM after
 * typesetting, but MathJax keeps the source string per math item).
 * ------------------------------------------------------------------ */
(function () {
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      try {
        var ta = document.createElement('textarea'); ta.value = text;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.focus(); ta.select();
        var ok = document.execCommand('copy'); document.body.removeChild(ta);
        if (ok) resolve(); else reject(new Error('copy failed'));
      } catch (e) { reject(e); }
    });
  }
  function addButtons() {
    if (!(window.MathJax && MathJax.startup && MathJax.startup.document)) return;
    var list;
    try { list = Array.from(MathJax.startup.document.math); } catch (e) { return; }
    list.forEach(function (item) {
      if (!item.display) return;                          // display blocks only
      var root = item.typesetRoot;
      var box = root && root.closest ? root.closest('div.math') : null;
      if (!box) return;
      var wrap = (box.parentNode && box.parentNode.classList && box.parentNode.classList.contains('simba-math-wrap')) ? box.parentNode : null;
      if (!wrap) { wrap = document.createElement('div'); wrap.className = 'simba-math-wrap'; box.parentNode.insertBefore(wrap, box); wrap.appendChild(box); }
      if (wrap.querySelector('.simba-tex-btn')) return;
      var tex = (item.math || '').trim();
      if (!tex) return;
      var btn = document.createElement('button'); btn.type = 'button'; btn.className = 'simba-tex-btn'; btn.textContent = 'Copy LaTeX';
      function flash(m) { btn.textContent = m; btn.classList.add('copied'); setTimeout(function () { btn.textContent = 'Copy LaTeX'; btn.classList.remove('copied'); }, 1400); }
      btn.addEventListener('click', function (e) { e.preventDefault(); copyText(tex).then(function () { flash('Copied!'); }).catch(function () { flash('Failed'); }); });
      wrap.appendChild(btn);
    });
  }
  window.addEventListener('load', function () {
    if (window.MathJax && MathJax.startup && MathJax.startup.promise) MathJax.startup.promise.then(addButtons).catch(function () {});
    else addButtons();
  });
})();

/* ------------------------------------------------------------------ *
 * Fast smooth-scroll for same-page anchor jumps (fixed ~300ms,
 * distance-independent — avoids the slow native scroll-behavior).
 * ------------------------------------------------------------------ */
(function () {
  var DURATION = 300, OFFSET = 20;
  function ease(t) { return 1 - Math.pow(1 - t, 3); }   // easeOutCubic
  function scrollToEl(el) {
    var startY = window.pageYOffset;
    var targetY = el.getBoundingClientRect().top + startY - OFFSET;
    var diff = targetY - startY;
    if (Math.abs(diff) < 2) return;
    var start = null;
    function step(ts) {
      if (start === null) start = ts;
      var p = Math.min((ts - start) / DURATION, 1);
      window.scrollTo(0, startY + diff * ease(p));
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[href]');
    if (!a) return;
    var href = a.getAttribute('href');
    if (!href || href.charAt(0) !== '#' || href.length < 2) return;
    var id = decodeURIComponent(href.slice(1));
    var el = document.getElementById(id);
    if (!el) { try { el = document.querySelector('a[name="' + (window.CSS ? CSS.escape(id) : id) + '"]'); } catch (x) {} }
    if (!el) return;
    e.preventDefault();
    scrollToEl(el);
    if (history.pushState) history.pushState(null, '', href);
  });
})();