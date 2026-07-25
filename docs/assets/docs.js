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
     The docs used to hand you a TOML blob to retype. Pick provider, workload and
     mode; a COMPLETE runnable file writes itself — including the [[provider]] and
     [[gate]] blocks the selection actually requires. Only `anthropic` and `openai`
     are built-in providers, and only `non-empty`/`json-valid` are built-in gates,
     so anything else must be declared or the config will not parse.
     Guarded on the element, so every page without a builder is unaffected.     */
  var fpb = document.getElementById('fpb');
  if (fpb) {
    var PROVIDERS = {
      anthropic: { ladder: ['anthropic/claude-haiku-4-5', 'anthropic/claude-sonnet-5'],
                   judge: 'anthropic/claude-opus-4-8', block: null },
      openai:    { ladder: ['openai/gpt-4.1-mini', 'openai/gpt-5.5'],
                   judge: 'anthropic/claude-haiku-4-5', block: null },
      google:    { ladder: ['google/gemini-3.1-flash', 'google/gemini-3.1-pro'],
                   judge: 'anthropic/claude-haiku-4-5',
                   block: '[[provider]]                # only anthropic + openai are built in\n' +
                          'id          = "google"\ndialect     = "gemini"\n' +
                          'base_url    = "https://generativelanguage.googleapis.com"\n' +
                          'api_key_env = "GEMINI_API_KEY"\n' },
      local:     { ladder: ['ollama/qwen2.5-coder:7b', 'anthropic/claude-sonnet-5'],
                   judge: 'anthropic/claude-opus-4-8',
                   block: '[[provider]]                # local rung; escalates to a frontier model\n' +
                          'id       = "ollama"\ndialect  = "openai"\n' +
                          'base_url = "http://localhost:11434"   # keyless\n' }
    };
    var WORK = {
      json: { gates: ['json-valid', 'extract-shape'],
              gate: function () {
                return '[[gate]]\nid         = "extract-shape"\n' +
                       'schema     = { type = "object", required = ["id", "total"] }\n' +
                       'on_abstain = "fail_closed"\n'; },
              why: 'A schema gate is the cheapest proof there is — it parses the response against your shape, or it does not. No judge, no extra tokens. Edit <code>required</code> to your fields.' },
      code: { gates: ['json-valid', 'unit-tests'],
              gate: function () {
                return '[[gate]]                    # any command reading the candidate as JSON on stdin\n' +
                       'id  = "unit-tests"\ncmd = ["your-test-runner", "--from-stdin"]\n'; },
              why: 'Your own test suite is the gate. This is the shape behind the published ≤10% bound on 974 MBPP tasks — the cheap model ships only when the tests actually pass. Point <code>cmd</code> at your real runner.' },
      prose:{ gates: ['non-empty', 'judge'],
              gate: function (p) {
                return '[[gate]]                    # judge sits OUTSIDE the ladder: maker ≠ checker\n' +
                       'id    = "judge"\njudge = { model = "' + p.judge + '", threshold = 0.7, ' +
                       'rubric = "The response fully and correctly resolves the request, with no errors." }\n'; },
              why: 'Prose has no parser, so a separate model grades the output against a rubric. The runner enforces maker ≠ checker, which is why the judge model sits outside the ladder — it costs tokens per call, so budget it.' },
      mixed:{ gates: ['non-empty', 'uncertainty'],
              gate: function (p) {
                return '[[gate]]                    # k samples; agreement becomes the score\n' +
                       'id          = "uncertainty"\nconsistency = { model = "' + p.ladder[0] +
                       '", k = 3, threshold = 0.6 }\n'; },
              why: 'No single check fits mixed traffic, so the same model is sampled k times and their agreement becomes the confidence score. Here maker = checker is deliberate. Most expensive gate — reach for it only when the others do not fit.' }
    };
    var sel = { provider: 'anthropic', work: 'json', mode: 'observe' };
    var out = document.getElementById('fpb-out');
    var why = document.getElementById('fpb-why');

    function render() {
      var p = PROVIDERS[sel.provider], w = WORK[sel.work];
      var q = function (a) { return a.map(function (x) { return '"' + x + '"'; }).join(', '); };
      var toml =
        '# firstpass.toml — generated below. Save it, then:\n' +
        '#   FIRSTPASS_MODE=' + sel.mode + ' FIRSTPASS_CONFIG=./firstpass.toml firstpass up\n\n' +
        (p.block ? p.block + '\n' : '') +
        '[[route]]                   # routes match top to bottom; first match wins\n' +
        'match  = {}                 # everything\n' +
        'mode   = "' + sel.mode + '"\n' +
        'ladder = [' + q(p.ladder) + ']\n' +
        'gates  = [' + q(w.gates) + ']\n\n' +
        w.gate(p);
      if (sel.mode === 'enforce') {
        toml += '\n[escalation]\nmax_rungs_per_request = 2   # one rung up, never a runaway\n';
      } else {
        toml += '\n# observe: every request is forwarded unchanged and a receipt is written off\n' +
                '# the hot path. Nothing routes differently until mode = "enforce".\n';
      }
      out.textContent = toml;
      why.innerHTML = '<b>Why these gates:</b> ' + w.why +
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
