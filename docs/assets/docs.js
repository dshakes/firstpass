/* Firstpass site — progressive enhancement only. Page renders fully without JS.
   (The pre-paint theme set lives inline in each <head>; this handles the toggle
   click, copy buttons, sidebar drawer, TOC scrollspy, install tabs, terminal.) */
(function () {
  'use strict';

  /* ---- theme toggle -------------------------------------------------------- */
  var root = document.documentElement;
  function currentTheme() {
    return root.getAttribute('data-theme') ||
      (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  function setTheme(t) {
    root.setAttribute('data-theme', t);
    try { localStorage.setItem('fp-theme', t); } catch (e) {}
    document.querySelectorAll('.theme-toggle').forEach(function (b) {
      b.setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
      b.setAttribute('aria-pressed', String(t === 'dark'));
    });
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', t === 'dark' ? '#08090b' : '#ffffff');
  }
  document.querySelectorAll('.theme-toggle').forEach(function (btn) {
    btn.addEventListener('click', function () {
      setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
    });
  });
  // reflect OS changes only when the user hasn't pinned a choice
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
      var saved; try { saved = localStorage.getItem('fp-theme'); } catch (x) {}
      if (!saved) setTheme(e.matches ? 'dark' : 'light');
    });
  }

  /* ---- copy buttons -------------------------------------------------------- */
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var block = btn.closest('.codeblock');
      var pre = block && block.querySelector('pre');
      if (!pre) return;
      var text = pre.innerText;
      var done = function () {
        var prev = btn.querySelector('.lbl');
        var old = prev ? prev.textContent : '';
        btn.classList.add('ok');
        if (prev) prev.textContent = 'Copied';
        setTimeout(function () { btn.classList.remove('ok'); if (prev) prev.textContent = old || 'Copy'; }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
    });
  });

  /* ---- top-nav mobile drawer (landing) ------------------------------------- */
  var menuBtn = document.querySelector('.menu-btn');
  var mobileNav = document.querySelector('.mobile-nav');
  if (menuBtn && mobileNav) {
    menuBtn.addEventListener('click', function () {
      var open = mobileNav.classList.toggle('open');
      menuBtn.setAttribute('aria-expanded', String(open));
    });
  }

  /* ---- docs sidebar drawer ------------------------------------------------- */
  var sbToggle = document.querySelector('.sidebar-toggle');
  var sidebar = document.querySelector('.sidebar');
  var scrim = document.querySelector('.scrim');
  function closeSidebar() {
    if (sidebar) sidebar.classList.remove('open');
    if (scrim) scrim.classList.remove('show');
    if (sbToggle) sbToggle.setAttribute('aria-expanded', 'false');
  }
  if (sbToggle && sidebar) {
    sbToggle.addEventListener('click', function () {
      var open = sidebar.classList.toggle('open');
      if (scrim) scrim.classList.toggle('show', open);
      sbToggle.setAttribute('aria-expanded', String(open));
    });
    if (scrim) scrim.addEventListener('click', closeSidebar);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeSidebar(); });
    sidebar.querySelectorAll('a').forEach(function (a) { a.addEventListener('click', closeSidebar); });
  }

  /* ---- TOC scrollspy ------------------------------------------------------- */
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc a[href^="#"]'));
  if (tocLinks.length && 'IntersectionObserver' in window) {
    var map = {};
    tocLinks.forEach(function (l) { map[l.getAttribute('href').slice(1)] = l; });
    var heads = tocLinks.map(function (l) { return document.getElementById(l.getAttribute('href').slice(1)); }).filter(Boolean);
    var visible = new Set();
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) visible.add(en.target.id); else visible.delete(en.target.id);
      });
      var firstId = null;
      for (var i = 0; i < heads.length; i++) { if (visible.has(heads[i].id)) { firstId = heads[i].id; break; } }
      if (firstId) {
        tocLinks.forEach(function (l) { l.classList.remove('active'); });
        if (map[firstId]) map[firstId].classList.add('active');
      }
    }, { rootMargin: '-80px 0px -70% 0px', threshold: 0 });
    heads.forEach(function (h) { obs.observe(h); });
  }

  /* ---- install tabs (landing) ---------------------------------------------- */
  document.querySelectorAll('[data-tabs]').forEach(function (group) {
    var tabs = group.querySelectorAll('.tab');
    var panels = group.querySelectorAll('.tabpanel');
    tabs.forEach(function (tab, i) {
      tab.addEventListener('click', function () {
        tabs.forEach(function (t) { t.setAttribute('aria-selected', 'false'); });
        panels.forEach(function (p) { p.classList.remove('active'); });
        tab.setAttribute('aria-selected', 'true');
        if (panels[i]) panels[i].classList.add('active');
      });
    });
  });

  /* ---- landing terminal: reveal lines (respects reduced motion) ------------ */
  var stream = document.getElementById('stream');
  if (stream) {
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var lines = Array.prototype.slice.call(stream.querySelectorAll('.ln'));
    if (reduce) {
      lines.forEach(function (l) { l.style.opacity = 1; });
    } else {
      lines.forEach(function (l) { l.style.opacity = 0; });
      var i = 0;
      (function step() {
        if (i >= lines.length) return;
        var l = lines[i++];
        l.classList.add('anim');
        setTimeout(step, 90 + Math.random() * 260);
      })();
    }
  }
})();
