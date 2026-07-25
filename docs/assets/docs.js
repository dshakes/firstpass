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



  /* ---- scroll reveal (landing) ---------------------------------------------
     A reveal that hides content in CSS and un-hides it from script can strand
     the page blank, so this is built to fail open in every direction:

       - nothing is armed unless the tab is actually visible. rAF and
         IntersectionObserver are both throttled in background tabs, so arming
         a hidden tab is what leaves cmd-clicked pages blank;
       - elements already on screen are revealed synchronously, not in a frame
         callback that a hidden tab will never run;
       - a failsafe timer strips the hiding class from anything still unshown,
         so the worst case is "no animation", never "no content";
       - reduced motion never arms at all.                                    */
  function armReveal() {
    if (document.visibilityState !== 'visible') return false;
    var targets = document.querySelectorAll(
      '.sec .card, .sec .figure, .sec .vs, .sec .statcol, .sec .install, .rig'
    );
    if (!targets.length) return true;
    var list = Array.prototype.slice.call(targets);
    var show = function (el) { el.classList.add('shown'); };

    if (!('IntersectionObserver' in window)) { return true; }   // leave content as-is

    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        show(en.target);
        obs.unobserve(en.target);                                // one-shot; never re-hides
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 });

    list.forEach(function (el, i) {
      el.classList.add('js-reveal');
      el.style.transitionDelay = Math.min(i % 4, 3) * 55 + 'ms';
      // Synchronous, because anything already on screen must not wait on a frame.
      if (el.getBoundingClientRect().top < window.innerHeight) show(el);
      else obs.observe(el);
    });

    // Failsafe: whatever has not been revealed within 4s gets un-hidden outright.
    setTimeout(function () {
      list.forEach(function (el) {
        if (!el.classList.contains('shown')) el.classList.remove('js-reveal');
      });
    }, 4000);
    return true;
  }

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduceMotion && !armReveal()) {
    // Backgrounded at load — wait until the tab is actually looked at.
    document.addEventListener('visibilitychange', function once() {
      if (armReveal()) document.removeEventListener('visibilitychange', once);
    });
  }

  /* ---- config builder ------------------------------------------------------
     The generated file is the ONLY source of these configs: `firstpass onboard`
     renders them and `cargo test -p firstpass-proxy presets` writes them to
     assets/ladder-presets.js, so the page cannot drift from the binary. This
     file keeps the wiring and none of the format.
     Degrades to the copyable default if the presets asset is missing.        */
  var fpb = document.getElementById('fpb');
  if (fpb) {
    var PRESETS = window.FP_LADDER_PRESETS || null;
    var WHY = {
      json: 'A schema gate is the cheapest proof there is — it parses the response against your shape, or it does not. No judge, no extra tokens. Edit <code>required</code> to your fields.',
      code: 'Your own test suite is the gate. This is the shape behind the published ≤10% bound on 974 MBPP tasks — the cheap model ships only when the tests actually pass. Point <code>cmd</code> at your real runner.',
      prose: 'Prose has no parser, so a separate model grades the output against a rubric. The runner enforces maker ≠ checker, which is why the judge model sits outside the ladder — it costs tokens per call, so budget it.',
      mixed: 'No single check fits mixed traffic, so the same model is sampled k times and their agreement becomes the confidence score. Here maker = checker is deliberate. Most expensive gate — reach for it only when the others do not fit.'
    };
    var sel = { provider: 'anthropic', shape: 'json', mode: 'observe' };
    var out = document.getElementById('fpb-out');
    var why = document.getElementById('fpb-why');

    function render() {
      var toml = PRESETS && PRESETS[sel.provider] && PRESETS[sel.provider][sel.shape]
        ? PRESETS[sel.provider][sel.shape][sel.mode]
        : null;
      out.textContent = toml || (
        '# Preset data failed to load. `firstpass onboard` asks these same three\n' +
        '# questions on a terminal and writes the file for you:\n' +
        '#   uvx firstpass onboard --apply --provider ' + sel.provider +
        ' --shape ' + sel.shape + ' --mode ' + sel.mode + '\n');
      why.innerHTML = '<b>Why these gates:</b> ' + WHY[sel.shape] +
        (sel.mode === 'observe'
          ? ' <b>Observe mode</b> changes no behavior — run a session, then read <code>firstpass savings</code>.'
          : ' <b>Enforce mode</b> serves from the cheap rung the moment a gate passes.');
    }

    fpb.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-k]');
      if (!b) return;
      sel[b.dataset.k] = b.dataset.v;
      fpb.querySelectorAll('button[data-k="' + b.dataset.k + '"]').forEach(function (o) {
        o.setAttribute('aria-pressed', String(o === b));
      });
      render();
    });
    render();
  }

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
