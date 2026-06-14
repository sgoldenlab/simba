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
    Array.prototype.forEach.call(srcVideo.querySelectorAll('source'), function (s) {
      var ns = document.createElement('source');
      ns.src = s.getAttribute('src'); ns.type = s.getAttribute('type') || '';
      v.appendChild(ns);
    });
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
})();