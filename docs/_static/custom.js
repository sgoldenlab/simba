window.dataLayer = window.dataLayer || [];
function gtag() { dataLayer.push(arguments); }
gtag('js', new Date());
gtag('config', 'G-PEKR9R5J47');

/* Tag the landing page early (runs in <head>) so CSS can hide the
 * redundant page H1 immediately — the hero banner is the splash. */
(function () {
  var base = location.pathname.split('/').pop();
  if (base === '' || base === 'index.html') document.documentElement.classList.add('simba-landing');
})();

/* ------------------------------------------------------------------ *
 * Splash screen — full-viewport intro overlay, shown ONCE per browser
 * session. Runs at parse time (in <head>) so it paints before page
 * content; dismisses after a short minimum, on window load, or on any
 * click / keypress. sessionStorage suppresses it for the rest of the
 * session, so it never delays repeat navigation.
 * ------------------------------------------------------------------ */
(function () {
  try { if (sessionStorage.getItem('simbaSplashSeen')) return; } catch (e) { /* private mode: still show once */ }

  var ROOT = (function () {
    var el = document.getElementById('documentation_options');   // precedes custom.js in <head>
    return (el && el.getAttribute('data-url_root')) || './';
  })();

  var root = document.documentElement;
  root.classList.add('simba-splash-on');

  var letters = 'SimBA'.split('').map(function (ch, i) {
    return '<span style="animation-delay:' + (0.15 + i * 0.09).toFixed(2) + 's">' + ch + '</span>';
  }).join('');

  var overlay = document.createElement('div');
  overlay.className = 'simba-splash';
  overlay.setAttribute('role', 'img');
  overlay.setAttribute('aria-label', 'SimBA — Simple Behavioral Analysis');
  overlay.innerHTML =
    '<div class="simba-splash-inner">' +
      '<div class="simba-splash-word">' + letters + '</div>' +
      '<div class="simba-splash-underline"></div>' +
      '<video class="simba-splash-mouse" autoplay loop muted playsinline preload="auto">' +
        '<source src="' + ROOT + '_static/img/mouse_run.webm" type="video/webm">' +   // mesh mouse + baked keypoints
      '</video>' +
      '<div class="simba-splash-cap">Simple Behavioral Analysis</div>' +
    '</div>' +
    '<div class="simba-splash-skip">click anywhere to skip</div>';
  root.appendChild(overlay);   // body does not exist yet at parse time; documentElement is fine for position:fixed


  var shownAt = (window.performance && performance.now) ? performance.now() : (+new Date());
  var MIN_MS = 3400, MAX_MS = 5200, done = false;   // hold long enough to enjoy the assembled scene + running mouse

  function dismiss() {
    if (done) return;
    done = true;
    try { sessionStorage.setItem('simbaSplashSeen', '1'); } catch (e) {}
    overlay.classList.add('leaving');
    root.classList.remove('simba-splash-on');
    setTimeout(function () { if (overlay.parentNode) overlay.parentNode.removeChild(overlay); }, 600);
  }
  function dismissAfterMin() {
    var elapsed = ((window.performance && performance.now) ? performance.now() : (+new Date())) - shownAt;
    setTimeout(dismiss, Math.max(0, MIN_MS - elapsed));
  }

  window.addEventListener('load', dismissAfterMin);
  setTimeout(dismiss, MAX_MS);                       // hard cap if load never fires
  overlay.addEventListener('click', dismiss);        // let users skip it
  window.addEventListener('keydown', dismiss, { once: true });
})();

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

  function parseSearchIndex(text) {
    var i = text.indexOf('{'), j = text.lastIndexOf('}');
    if (i === -1 || j === -1) return null;
    try { return JSON.parse(text.slice(i, j + 1)); } catch (e) { return null; }
  }
  function entriesFromGenindex(html) {
    var doc = new DOMParser().parseFromString(html, 'text/html'), out = [];
    Array.prototype.forEach.call(doc.querySelectorAll('a[href]'), function (a) {
      var href = a.getAttribute('href');
      if (!href || href.indexOf('#') === -1) return;            // only object deep-links
      var label = a.textContent.replace(/\s+/g, ' ').trim();
      if (!label) return;
      out.push({ label: label, href: ROOT + href, kind: 'API' });
    });
    return out;
  }
  function entriesFromSearchIndex(idx) {
    var out = [], docs = idx.docnames || [], titles = idx.titles || [], at = idx.alltitles;
    if (at && Object.keys(at).length) {                         // Sphinx >=5: page titles + section headings
      Object.keys(at).forEach(function (title) {
        (at[title] || []).forEach(function (pair) {
          var di = pair[0], anchor = pair[1];
          if (typeof di !== 'number' || !docs[di]) return;
          var href = ROOT + docs[di] + '.html' + (anchor ? ('#' + String(anchor).replace(/^#/, '')) : '');
          out.push({ label: title, href: href, kind: anchor ? 'section' : 'page' });
        });
      });
    } else {                                                    // fallback: page titles only
      for (var d = 0; d < docs.length; d++) out.push({ label: titles[d] || docs[d], href: ROOT + docs[d] + '.html', kind: 'page' });
    }
    return out;
  }
  function loadEntries(cb) {
    if (entries) return cb(entries);
    if (loading) return;
    loading = true;
    Promise.all([
      fetch(ROOT + 'genindex.html').then(function (r) { return r.text(); }).catch(function () { return ''; }),
      fetch(ROOT + 'searchindex.js').then(function (r) { return r.text(); }).catch(function () { return ''; })
    ]).then(function (res) {
      var out = [];
      if (res[0]) out = out.concat(entriesFromGenindex(res[0]));
      var idx = res[1] ? parseSearchIndex(res[1]) : null;
      if (idx) out = out.concat(entriesFromSearchIndex(idx));
      var seen = {}, dedup = [];
      out.forEach(function (e) { var k = e.label + '|' + e.href; if (seen[k]) return; seen[k] = 1; dedup.push(e); });
      entries = dedup; cb(entries);
    }).catch(function () { entries = []; cb(entries); });
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
    input.type = 'text'; input.placeholder = 'Jump to a page, section, function or class…';
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
      var kind = e.kind ? '<span class="simba-cmdk-kind simba-cmdk-kind-' + e.kind + '">' + e.kind + '</span>' : '';
      return '<a class="simba-cmdk-item' + (i === 0 ? ' active' : '') + '" href="' + e.href + '"><span class="simba-cmdk-label">' + escapeHtml(e.label) + '</span>' + kind + '</a>';
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
    btn.innerHTML = '<span>🔍&nbsp;&nbsp;Search the docs — pages, sections, functions…</span><kbd>Ctrl K</kbd>';
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
  var CURATED = [
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
  // Merge: curated openers first, then every other embedded video (window.SIMBA_DEMOS_ALL,
  // generated by conf.py from all ".. video::" refs, newest first). Falls back to just the
  // curated list when the manifest is absent (e.g. local preview without a build).
  var _all = window.SIMBA_DEMOS_ALL || [];
  var _extra = _all.filter(function (f) { return CURATED.indexOf(f) === -1; });
  var SLIDES = CURATED.concat(_extra);
  window.SIMBA_DEMOS = SLIDES;      // shared with the "explore all demos" wall below the carousel
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
      var frac = viewport.clientWidth <= 700 ? 0.82 : FRAC;   // wider active slide on phones
      slideW = Math.max(200, Math.round(viewport.clientWidth * frac));
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
 * Landing page: "explore all demos" scrolling wall (hybrid — sits just
 * below the hero carousel). Three rows of tiles drift in alternating
 * directions; reuses the carousel's demo list. Only tiles near the
 * viewport load & play (IntersectionObserver), the row pauses on hover,
 * and clicking a tile opens it large via the shared video lightbox.
 * prefers-reduced-motion -> a static, manually-scrollable strip.
 * ------------------------------------------------------------------ */
(function () {
  var SPEED = 3.4;                    // seconds of drift per tile (lower = faster)
  function caption(fn) {
    return fn.replace(/\.(webm|mp4)$/, '').replace(/_/g, ' ')
             .replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2');
  }
  window.addEventListener('load', function () {
    var base = location.pathname.split('/').pop();
    if (!(base === '' || base === 'index.html')) return;
    var full = (window.SIMBA_DEMOS || []).slice();
    if (full.length < 2) return;
    var main = document.querySelector('[role="main"]');
    if (!main || main.querySelector('.simba-wall')) return;

    // Show every embedded video. Fast load comes from lazy-loading (tiles fetch/play only near the
    // viewport) + pausing the whole wall off-screen -- not from limiting the count. Fewer rows on
    // phones, and a static strip for reduced-motion / Save-Data.
    var conn = navigator.connection || {};
    var reduced = (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) || !!conn.saveData;
    var mobile = window.innerWidth <= 700;
    var ROWS = mobile ? 2 : 3;
    var list = full;

    var wall = document.createElement('div'); wall.className = 'simba-wall' + (reduced ? ' static' : '');
    var rowsWrap = document.createElement('div'); rowsWrap.className = 'simba-wall-rows';
    wall.appendChild(rowsWrap);

    var vids = [];
    var io = ('IntersectionObserver' in window) ? new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) {
          if (!v.src && v.getAttribute('data-src')) v.src = v.getAttribute('data-src');
          if (!reduced) { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
        } else { v.pause(); }
      });
    }, { rootMargin: '120px', threshold: 0.15 }) : null;

    var per = Math.ceil(list.length / ROWS);
    for (var r = 0; r < ROWS; r++) {
      var chunk = list.slice(r * per, (r + 1) * per);
      if (!chunk.length) continue;
      var row = document.createElement('div'); row.className = 'simba-wall-row' + (r % 2 ? ' rev' : '');
      var track = document.createElement('div'); track.className = 'simba-wall-track';
      chunk.concat(chunk).forEach(function (fn) {          // duplicate chunk so translateX(-50%) loops seamlessly
        var tile = document.createElement('div'); tile.className = 'simba-wall-tile';
        var v = document.createElement('video');
        v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'none';
        if (!reduced) v.autoplay = true;                  // reliable in-view playback once the src is set
        v.setAttribute('data-src', '_static/img/' + fn);
        var cap = document.createElement('div'); cap.className = 'cap'; cap.textContent = caption(fn);
        tile.appendChild(v); tile.appendChild(cap);
        tile.addEventListener('click', function () {       // give it a real src before enlarging, so it always plays
          if (!v.src && v.getAttribute('data-src')) v.src = v.getAttribute('data-src');
          if (window.SimbaVideoLightbox) window.SimbaVideoLightbox(v);
        });
        track.appendChild(tile); vids.push(v);
        if (io) io.observe(v);
      });
      if (!reduced) track.style.animationDuration = (chunk.length * SPEED) + 's';
      row.appendChild(track);
      rowsWrap.appendChild(row);
    }

    var carousel = main.querySelector('.simba-carousel');
    if (carousel) carousel.insertAdjacentElement('afterend', wall);
    else main.appendChild(wall);

    // Pause the whole wall (drift + playback) when it is off-screen or the tab is hidden -> saves CPU/battery/data.
    var active = true;
    function setActive(on) {
      if (on === active) return; active = on;
      wall.classList.toggle('offscreen', !on);
      vids.forEach(function (v) {
        if (!v.src) return;
        if (on && !reduced) { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
        else v.pause();
      });
    }
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) { setActive(es[0].isIntersecting); }, { threshold: 0.01 }).observe(wall);
    }
    document.addEventListener('visibilitychange', function () { setActive(!document.hidden); });
  });
})();

/* --- Idle SimBA mouse at the top of the auto-generated A–Z index (genindex) --- */
(function () {
  window.addEventListener('load', function () {
    if (location.pathname.split('/').pop() !== 'genindex.html') return;
    var main = document.querySelector('[role="main"]');
    if (!main || main.querySelector('.simba-genindex-mouse')) return;
    var ROOT = (function () {
      var el = document.getElementById('documentation_options');
      return (el && el.getAttribute('data-url_root')) || '';
    })();
    var wrap = document.createElement('div');
    wrap.className = 'simba-orbit-foot simba-genindex-mouse';
    wrap.style.margin = '4px 0 22px';
    wrap.innerHTML =
      '<video autoplay loop muted playsinline preload="auto" aria-label="Idle SimBA mouse">' +
        '<source src="' + ROOT + '_static/img/mouse_idle_simba.webm" type="video/webm">' +
      '</video>';
    main.insertBefore(wrap, main.firstChild);
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

/* ------------------------------------------------------------------ *
 * Permalink "copy link": click the ¶ on a heading or function/class
 * signature to copy its full URL to the clipboard.
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
  function flash(el, msg) {
    var t = document.createElement('span');
    t.className = 'simba-permalink-toast';
    t.textContent = msg;
    el.appendChild(t);
    requestAnimationFrame(function () { t.classList.add('show'); });
    setTimeout(function () {
      t.classList.remove('show');
      setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, 220);
    }, 1100);
  }
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a.headerlink');
    if (!a) return;
    var href = a.getAttribute('href') || '';
    var url = location.origin + location.pathname + href;   // absolute, shareable
    copyText(url).then(function () { flash(a, 'Link copied!'); }).catch(function () {});
    // (the smooth-scroll handler already updates the hash / scrolls)
  });
  function flashBtn(btn) {
    var o = btn.getAttribute('data-icon') || btn.innerHTML;
    btn.setAttribute('data-icon', o);
    btn.innerHTML = 'Copied!'; btn.classList.add('copied');
    setTimeout(function () { btn.innerHTML = o; btn.classList.remove('copied'); }, 1200);
  }
  window.addEventListener('load', function () {
    Array.prototype.forEach.call(document.querySelectorAll('a.headerlink'), function (a) {
      a.title = 'Copy link to clipboard';
    });
    // visible "copy link" button next to [source] on function/class signatures
    Array.prototype.forEach.call(document.querySelectorAll('.rst-content dl.py > dt[id]'), function (dt) {
      if (dt.querySelector('.simba-copylink')) return;
      var hl = dt.querySelector('a.headerlink');
      var href = hl ? hl.getAttribute('href') : '#' + dt.id;
      var btn = document.createElement('button');
      btn.type = 'button'; btn.className = 'simba-copylink'; btn.title = 'Copy link to this entry';
      btn.innerHTML = '🔗 link';
      btn.addEventListener('click', function (e) {
        e.preventDefault(); e.stopPropagation();
        copyText(location.origin + location.pathname + href).then(function () { flashBtn(btn); }).catch(function () {});
        if (history.pushState) history.pushState(null, '', href);
      });
      dt.appendChild(btn);
    });
  });
})();

/* ------------------------------------------------------------------ *
 * Back-to-top button (appears after scrolling; fast smooth scroll up)
 * ------------------------------------------------------------------ */
(function () {
  var btn = null;
  function toTop() {
    var startY = window.pageYOffset, start = null, D = 340;
    function ease(t) { return 1 - Math.pow(1 - t, 3); }
    function step(ts) {
      if (start === null) start = ts;
      var p = Math.min((ts - start) / D, 1);
      window.scrollTo(0, startY * (1 - ease(p)));
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  function ensure() {
    if (btn) return btn;
    btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'simba-top-btn';
    btn.title = 'Back to top';
    btn.setAttribute('aria-label', 'Back to top');
    btn.addEventListener('click', toTop);
    document.body.appendChild(btn);
    return btn;
  }
  function onScroll() {
    ensure();
    if (window.pageYOffset > 450) btn.classList.add('show');
    else btn.classList.remove('show');
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('load', onScroll);
})();

/* ------------------------------------------------------------------ *
 * "On this page" right-rail TOC + scroll-spy (API pages only).
 * Lists the page's function/class/method anchors, highlights the one
 * currently in view, hides on narrow screens.
 * ------------------------------------------------------------------ */
(function () {
  var MIN_ENTRIES = 3;
  window.addEventListener('load', function () {
    var dts = document.querySelectorAll('.rst-content dl.py > dt[id]');
    var targets = [];
    Array.prototype.forEach.call(dts, function (dt) { if (dt.id) targets.push(dt); });
    if (targets.length < MIN_ENTRIES) return;

    var nav = document.createElement('nav');
    nav.className = 'simba-toc';
    nav.setAttribute('aria-label', 'On this page');
    var title = document.createElement('div');
    title.className = 'simba-toc-title';
    title.textContent = 'On this page';
    nav.appendChild(title);

    var filter = document.createElement('input');
    filter.className = 'simba-toc-filter';
    filter.type = 'text';
    filter.placeholder = 'Filter methods…';
    filter.setAttribute('aria-label', 'Filter methods on this page');
    nav.appendChild(filter);

    var links = {};
    targets.forEach(function (dt) {
      var nameEl = dt.querySelector('.sig-name, .descname, code.descname');
      var name = nameEl ? nameEl.textContent.trim() : dt.id.split('.').pop();
      var a = document.createElement('a');
      a.href = '#' + dt.id;
      a.textContent = name;
      a.title = name;
      nav.appendChild(a);
      links[dt.id] = a;
    });
    document.body.appendChild(nav);

    function applyFilter() {
      var q = filter.value.trim().toLowerCase();
      targets.forEach(function (dt) {
        var a = links[dt.id];
        a.style.display = (!q || a.textContent.toLowerCase().indexOf(q) !== -1) ? '' : 'none';
      });
    }
    filter.addEventListener('input', applyFilter);
    filter.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { filter.value = ''; applyFilter(); filter.blur(); }
      else if (e.key === 'Enter') {                       // jump to first visible match
        for (var i = 0; i < targets.length; i++) {
          var a = links[targets[i].id];
          if (a.style.display !== 'none') { a.click(); break; }
        }
      }
    });

    var current = null;
    function setActive(id) {
      if (id === current) return;
      if (current && links[current]) links[current].classList.remove('active');
      current = id;
      var a = links[id];
      if (!a) return;
      a.classList.add('active');
      // keep the active item visible within the rail (without scrolling the page)
      var lt = a.offsetTop, lb = lt + a.offsetHeight;
      if (lt < nav.scrollTop) nav.scrollTop = lt - 8;
      else if (lb > nav.scrollTop + nav.clientHeight) nav.scrollTop = lb - nav.clientHeight + 8;
    }
    function spy() {
      var active = targets[0].id;
      for (var i = 0; i < targets.length; i++) {
        if (targets[i].getBoundingClientRect().top <= 130) active = targets[i].id;
        else break;
      }
      setActive(active);
    }
    var ticking = false;
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; requestAnimationFrame(function () { spy(); ticking = false; }); }
    }, { passive: true });
    spy();
  });
})();

/* ------------------------------------------------------------------ *
 * Landing-page hero banner: animated navy gradient, logo, rotating
 * tagline, and CTA buttons.
 * ------------------------------------------------------------------ */
(function () {
  window.addEventListener('load', function () {
    var base = location.pathname.split('/').pop();
    if (!(base === '' || base === 'index.html')) return;
    var main = document.querySelector('[role="main"]');
    if (!main || main.querySelector('.simba-hero-banner')) return;
    var ROOT = (function () {
      var el = document.getElementById('documentation_options');
      return (el && el.getAttribute('data-url_root')) || '';
    })();

    var hero = document.createElement('div');
    hero.className = 'simba-hero-banner';
    hero.innerHTML =
      '<video class="simba-hero-vid" autoplay loop muted playsinline preload="auto">' +
        '<source src="' + ROOT + '_static/img/landing_1.webm" type="video/webm">' +
      '</video>' +
      '<div class="simba-hero-cta"></div>';
    // lift the hero up to the content wrapper so it can span the full width
    var wrap = document.querySelector('.wy-nav-content-wrap') || main;
    wrap.insertBefore(hero, wrap.firstChild);

    // hide the now-duplicate in-content landing gif
    var dup = main.querySelector('img[src*="landing_1"]');
    if (dup) { var w = dup.closest('a') || dup; if (!w.closest('.simba-hero-banner')) w.style.display = 'none'; }

    // hide the redundant page title (the hero banner is the splash)
    var pageH1 = main.querySelector('h1');
    if (pageH1 && !pageH1.closest('.simba-hero-banner')) pageH1.style.display = 'none';

    // hero already shows the section buttons -> hide the duplicate card grid below
    var grid = main.querySelector('.simba-cardgrid');
    if (grid) grid.style.display = 'none';

    // CTA buttons mirror the section links shown below (the toctree),
    // minus the stale RST tutorials/docs (those stay in the sidebar, not featured here).
    var SKIP = /tutorial|walkthrough|labell?ing/i;
    var cta = hero.querySelector('.simba-hero-cta'), seen = {};
    Array.prototype.forEach.call(document.querySelectorAll('.toctree-wrapper li.toctree-l1 > a'), function (a) {
      var href = a.getAttribute('href');
      if (!href || seen[href]) return;
      if (SKIP.test(href) || SKIP.test(a.textContent)) return;
      seen[href] = 1;
      var b = document.createElement('a');
      b.className = 'simba-hero-btn';
      b.href = href;
      b.textContent = a.textContent.trim();
      cta.appendChild(b);
    });
    var gh = document.createElement('a');
    gh.className = 'simba-hero-btn'; gh.href = 'https://github.com/sgoldenlab/simba';
    gh.target = '_blank'; gh.rel = 'noopener'; gh.textContent = '★ GitHub';
    cta.appendChild(gh);

    requestAnimationFrame(function () { hero.classList.add('in'); });
  });
})();

/* ------------------------------------------------------------------ *
 * Sidebar search box: clearer placeholder + Ctrl-K hint.
 * ------------------------------------------------------------------ */
(function () {
  window.addEventListener('load', function () {
    var inp = document.querySelector('.wy-side-nav-search input[type="text"]');
    if (inp) inp.setAttribute('placeholder', 'Search docs…  (Ctrl K)');
  });
})();

/* ------------------------------------------------------------------ *
 * Sidebar extras: prominent "Install SimBA" CTA + pinned quick-links footer.
 * ------------------------------------------------------------------ */
(function () {
  var ROOT = (function () {
    var el = document.getElementById('documentation_options');
    return (el && el.getAttribute('data-url_root')) || './';
  })();
  window.addEventListener('load', function () {
    var search = document.querySelector('.wy-side-nav-search');
    if (search && !document.querySelector('.simba-install-cta')) {
      var cta = document.createElement('a');
      cta.className = 'simba-install-cta';
      cta.href = ROOT + 'installation.html';
      cta.innerHTML = '⚙️&nbsp;&nbsp;Install SimBA';
      search.insertAdjacentElement('afterend', cta);
    }
    var scroll = document.querySelector('.wy-side-scroll');
    if (scroll && !document.querySelector('.simba-side-links')) {
      var f = document.createElement('div');
      f.className = 'simba-side-links';
      f.innerHTML =
        '<a href="https://github.com/sgoldenlab/simba" target="_blank" rel="noopener">GitHub</a>' +
        '<a href="https://pypi.org/project/Simba-UW-tf-dev/" target="_blank" rel="noopener">PyPI</a>' +
        '<a href="https://app.gitter.im/#/room/#SimBA-Resource_community:gitter.im" target="_blank" rel="noopener">Gitter</a>' +
        '<a href="https://www.nature.com/articles/s41593-024-01649-9" target="_blank" rel="noopener">Paper</a>';
      scroll.appendChild(f);
    }
  });
})();

/* ------------------------------------------------------------------ *
 * Notebooks page: give each tile a short subtitle under its title.
 * Keyed by notebook filename so the toctree (and sidebar nav) stay intact.
 * ------------------------------------------------------------------ */
(function () {
  var SUB = {
    'CLI Example 1': 'Import pose data and run a classifier end-to-end.',
    'shap_example_1': 'Compute SHAP explainability scores on a single core.',
    'shap_example_2': 'Compute SHAP scores in parallel across cores.',
    'shap_log_3': 'GPU-accelerated SHAP score computation.',
    'outlier_correction': 'Correct movement & location pose outliers.',
    'third_party_append': 'Append annotations from BORIS, Solomon, etc.',
    'advanced_smoothing_interpolation': 'Fine-grained smoothing & interpolation control.',
    'advanced_outlier_correction': 'Custom, parameterized outlier correction.',
    'kleinberg_gridsearch': 'Grid-search Kleinberg bout-smoothing settings.',
    'train_models': 'Train a random-forest classifier from code.',
    'path_plots': 'Draw animal movement paths.',
    'classifier_validation_videos': 'Render clips to validate classifier hits.',
    'gantt_plots': 'Gantt charts of classified bouts over time.',
    'distance_plotter': 'Plot distances between body-parts.',
    'clf_results_plotting': 'Overlay classification results on video.',
    'Visualize pose-estimation': 'Plot tracked body-part locations.',
    'roi_feature_visualizer': 'Visualize ROI-based features.',
    'probability_plots': 'Plot classifier probability over time.',
    'hatmap_location_plotter': 'Heatmaps of where animals spend time.',
    'heatmap_clf': 'Heatmaps of where behaviors occur.',
    'pose_plotter_gpu': 'GPU-accelerated pose-estimation plotting.',
    'geometry_example_1': 'Key-point movement statistics in a grid.',
    'geometry_example_2': 'Hull movement statistics in a grid.',
    'geometry_ex_3': 'Build & analyze animal paths as geometries.',
    'geometry_example_3': 'Slice animal videos by location (CPU).',
    'geometry_example_5': 'Slice animal shapes from frames.',
    'geometry_example_6': 'ROI and path statistics from geometries.',
    'geometry_example_7': 'Further geometry computation examples.',
    'yolo_ex_1': 'Detect animals with YOLO bounding boxes.',
    'yolo_ex_2': 'YOLO detection — further example.',
    'bg_remove': 'Remove static background from videos.',
    'egocentric_align': 'Egocentrically align data and video.',
    'blob_tracking': 'Track animals as blobs (no pose model).',
    'blob_tracking_vis': 'Visualize blob-tracking results.',
    'import_sleap_h5': 'Import SLEAP .h5 tracking files.',
    'multiclass': 'Build multi-class behavior classifiers.',
    'define_rois': 'Define ROIs programmatically.'
  };
  window.addEventListener('load', function () {
    var root = document.querySelector('.simba-nb');
    if (!root) return;
    Array.prototype.forEach.call(root.querySelectorAll('.toctree-l1 > a.reference.internal'), function (a) {
      if (a.querySelector('.simba-nb-txt')) return;                 // already processed
      var base = (a.getAttribute('href') || '').split('/').pop().replace(/\.html.*$/, '');
      try { base = decodeURIComponent(base); } catch (e) {}
      var title = a.textContent.trim();
      var sub = SUB[base];
      a.textContent = '';
      var wrap = document.createElement('span'); wrap.className = 'simba-nb-txt';
      var t = document.createElement('span'); t.className = 'simba-nb-title'; t.textContent = title;
      wrap.appendChild(t);
      if (sub) { var s = document.createElement('span'); s.className = 'simba-nb-sub-card'; s.textContent = sub; wrap.appendChild(s); }
      a.appendChild(wrap);
    });
  });
})();

/* ------------------------------------------------------------------ *
 * Global lazy-loading of media (page-load speedup).
 *
 * Docs pages stack many large .webp/.png images and .webm/.mp4 clips;
 * downloading the off-screen ones blocks first paint. Native
 * loading="lazy" defers images that are NOT in the initial viewport
 * (in-viewport ones — logo, hero — still load immediately, so LCP is
 * unaffected). For <video> we set preload="none" so the demo clips
 * fetch no data until the user hits play.
 *
 * The attribute must be set BEFORE the browser begins fetching, so a
 * MutationObserver tags nodes as they stream into the DOM — a
 * DOMContentLoaded-only pass would run too late for images already in
 * flight. A final sweep + disconnect runs once the DOM is parsed.
 * ------------------------------------------------------------------ */
(function () {
  function tagImg(img) {
    if (!img.hasAttribute('loading')) img.setAttribute('loading', 'lazy');
    if (!img.hasAttribute('decoding')) img.setAttribute('decoding', 'async');
  }
  function tagVideo(v) {
    // Don't override autoplay clips; otherwise defer the download.
    if (!v.autoplay && !v.hasAttribute('preload')) v.setAttribute('preload', 'none');
  }
  function scan(node) {
    if (node.nodeType !== 1) return;                         // elements only
    if (node.tagName === 'IMG') tagImg(node);
    else if (node.tagName === 'VIDEO') tagVideo(node);
    if (node.querySelectorAll) {
      Array.prototype.forEach.call(node.querySelectorAll('img'), tagImg);
      Array.prototype.forEach.call(node.querySelectorAll('video'), tagVideo);
    }
  }
  var mo = new MutationObserver(function (muts) {
    for (var i = 0; i < muts.length; i++) {
      var added = muts[i].addedNodes;
      for (var j = 0; j < added.length; j++) scan(added[j]);
    }
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener('DOMContentLoaded', function () {
    scan(document.documentElement);
    mo.disconnect();
  });
})();

/* --- Credits page: spin a person card on click, open their link in a new tab --- */
(function () {
  document.addEventListener('click', function (e) {
    var card = e.target.closest && e.target.closest('a.simba-person');
    if (!card) return;
    // let modified clicks (new tab, etc.) behave normally
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    if (card.classList.contains('simba-spin')) return;
    // Spin the card FIRST (page stays in front so the flip is visible), then
    // open the link in a new tab. The 0.7s delay is within the browser's
    // user-activation window, so window.open is not treated as a popup.
    e.preventDefault();
    var href = card.getAttribute('href');
    card.classList.add('simba-spin');
    setTimeout(function () {
      card.classList.remove('simba-spin');
      if (href) window.open(href, '_blank', 'noopener');
    }, 700);
  });
})();
