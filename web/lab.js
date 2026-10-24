/**
 * 04 · Try it — upload a recording, follow the job, read the result; generate without a recording.
 *
 * Talks to the API of `python whistle.py serve` (same origin). Without it (a static host, or web/serve.py) the
 * upload is disabled and the examples open their pre-rendered results (data/examples.json, from
 * `python whistle.py export-web`). Every value that came from a file name is inserted as text, never as markup.
 *
 * Sound: the server renders each image to a WAV at the whistle's own centre pitch (sonify.py). "Slow ×2" and the
 * pitch shift re-synthesise the same contour here, with the same rules (phase-continuous tone, 2.67 ms ramps,
 * 20 ms of silence either side, peak −3 dBFS), so they need no round trip.
 */
(() => {
  const $ = s => document.querySelector(s);
  const h = (tag, props = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? '' : v);
    }
    for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
    return e;
  };
  const kHz = f => (f / 1000).toFixed(2);

  /* SHALLOWS, the palette of infer.py: quiet = pale sea foam, loud = deep water. The PNGs are plain levels (0-255)
     and the page colours them, so the overview and every whistle picture share one look. */
  const SHALLOWS = [[0, [238, 247, 248]], [0.18, [200, 226, 233]], [0.4, [127, 182, 201]], [0.65, [42, 110, 150]], [0.85, [13, 58, 92]], [1, [4, 26, 46]]];
  const LUT = (() => {
    const t = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const v = i / 255; let k = 1;
      while (k < SHALLOWS.length - 1 && v > SHALLOWS[k][0]) k++;
      const [a, ca] = SHALLOWS[k - 1], [b, cb] = SHALLOWS[k], w = (v - a) / (b - a);
      for (let c = 0; c < 3; c++) t[i * 3 + c] = ca[c] + (cb[c] - ca[c]) * w;
    }
    return t;
  })();
  function paint(canvas, url) {
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
      const c = canvas.getContext('2d');
      c.drawImage(img, 0, 0);
      const d = c.getImageData(0, 0, canvas.width, canvas.height), p = d.data;
      for (let i = 0; i < p.length; i += 4) { const v = p[i] * 3; p[i] = LUT[v]; p[i + 1] = LUT[v + 1]; p[i + 2] = LUT[v + 2]; p[i + 3] = 255; }
      c.putImageData(d, 0, 0);
    };
    img.src = url;
    return canvas;
  }
  const shot = (url, label) => paint(h('canvas', { class: 'shot', width: 128, height: 128, role: 'img', 'aria-label': label }), url);

  const STAGES = { queued: 'Waiting for the previous job', reading: 'Reading the file', detecting: 'Finding whistles',
                   model: 'Running the model', sound: 'Turning images into sound', overview: 'Drawing the overview',
                   done: 'Done' };
  const S = { api: false, status: null, file: null, result: null, base: null, sel: 0, slow: false, shift: 0, examples: [] };

  /* ------------------------------------------------------------------ sound */
  const player = new Audio();
  let ctx = null, node = null, playing = null;
  const stop = () => {
    player.pause();
    if (node) { try { node.stop(); } catch (_) {} node = null; }
    if (playing) { playing.setAttribute('aria-pressed', 'false'); playing = null; }
  };
  player.addEventListener('ended', stop);
  function synth(sound, slow, shift) {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
    const fs = ctx.sampleRate, c = sound.contour, hop = c.hop_s * slow, n = c.offset_hz.length;
    const nTone = Math.round(n * hop * fs), pad = Math.round(0.02 * fs), ramp = Math.min(Math.round(0.0026667 * fs), nTone >> 1);
    const buf = ctx.createBuffer(1, nTone + 2 * pad, fs), x = buf.getChannelData(0);
    let ph = 0, peak = 0;
    for (let i = 0; i < nTone; i++) {
      const u = (i + 0.5) / fs / hop - 0.5, k = Math.max(0, Math.min(n - 1, Math.floor(u))), k1 = Math.min(n - 1, k + 1);
      const w = Math.max(0, Math.min(1, u - k)), f = Math.max(20, sound.fc_hz + shift + c.offset_hz[k] * (1 - w) + c.offset_hz[k1] * w);
      ph += 2 * Math.PI * f / fs;
      let v = Math.cos(ph);
      if (i < ramp) v *= 0.5 * (1 - Math.cos(Math.PI * i / ramp));
      else if (i >= nTone - ramp) v *= 0.5 * (1 - Math.cos(Math.PI * (nTone - 1 - i) / ramp));
      x[pad + i] = v; peak = Math.max(peak, Math.abs(v));
    }
    const g = 0.7079 / Math.max(peak, 1e-9);
    for (let i = 0; i < x.length; i++) x[i] *= g;
    return buf;
  }
  function play(btn, url, sound, model) {
    if (playing === btn) { stop(); return; }
    stop();
    playing = btn; btn.setAttribute('aria-pressed', 'true');
    if (model && sound && sound.contour && (S.slow || S.shift)) {
      const buf = synth(sound, S.slow ? 2 : 1, S.shift);
      if (ctx.state === 'suspended') ctx.resume();
      node = ctx.createBufferSource(); node.buffer = buf; node.connect(ctx.destination);
      node.onended = () => { if (playing === btn) stop(); };
      node.start();
    } else { player.src = url; player.play().catch(stop); }
  }
  const playBtn = (label, url, sound, model = true) => {
    const ok = !!url;
    const b = h('button', { class: 'play', type: 'button', 'aria-pressed': 'false', disabled: !ok,
                            title: ok ? null : 'Sound not rendered for this whistle (limit per recording)' },
                h('span', { class: 'ic', 'aria-hidden': 'true' }), label);
    if (ok) b.addEventListener('click', () => play(b, url, sound, model));
    return b;
  };
  const fileUrl = (name, base = S.base) => name ? new URL(name, base).href : null;

  /* ------------------------------------------------------------------ status and examples */
  async function init() {
    try {
      const r = await fetch('api/status');
      if (!r.ok) throw new Error(r.status);
      S.status = await r.json(); S.api = true;
    } catch (_) { S.api = false; }
    // the status line speaks to visitors; the exact file paths stay in its tooltip (and in model-status)
    const st = $('#lab-status'), txt = st.querySelector('.txt');
    if (!S.api) {
      txt.textContent = 'This copy of the page runs without its analysis server, so here are recordings we analysed earlier.';
      st.title = 'Live analysis: python whistle.py serve';
      $('#drop').classList.add('off'); $('#file').disabled = true;
    } else {
      const m = S.status.model;
      txt.textContent = (m.ready.same_type ? 'The model is ready. Every whistle you bring gets a recreation and new whistles of its family.'
        : m.ready.recreate ? 'The model is ready to recreate whistles; its map of whistle families is still being drawn.'
        : 'The model is still in training. Finding whistles and hearing them works now; the recreations and new whistles appear here once it is done.')
        + (S.status.matches_training_data === false ? ' (Note: the preprocessing settings differ from the training data.)' : '');
      st.title = m.message;
      st.classList.toggle('ok', m.ready.recreate);
      $('#nsamples').value = S.status.limits.n_samples;
      $('#drop-n').textContent = `${S.status.limits.formats.map(f => f.slice(1).toUpperCase()).join(', ')} · ${S.status.limits.min_sample_rate_hz / 1000} kHz or more · up to ${S.status.limits.max_duration_s} s · ${S.status.limits.max_upload_mb} MB`;
    }
    setupGenerate();
    try {
      const ex = await (await fetch('data/examples.json')).json();
      S.examples = ex.examples || [];
      $('#ex-row').replaceChildren(...S.examples.map((e, i) => exampleCard(e, i)));
      $('#examples').hidden = !S.examples.length;
    } catch (_) { /* no examples exported */ }
  }

  /* an example as a card: a small whistle drawn in its own style, the encounter, when, how many whistles */
  function exampleCard(e, i) {
    const d = e.source_file.split('_')[1] || '', when = d ? new Date(`${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}T00:00:00Z`) : null;
    const f = [u => 0.5 * Math.sin(3.3 * Math.PI * u) + 0.7 * (u - 0.45), u => -0.8 + 1.6 * Math.pow(u, 0.8), u => -0.6 + 1.3 * Math.sin(Math.PI * u)][i % 3];
    let dpath = '';
    for (let k = 0; k <= 60; k++) { const u = k / 60; dpath += `${k ? 'L' : 'M'}${(8 + u * 284).toFixed(1)} ${(27 - f(u) * 20).toFixed(1)}`; }
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'), p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    svg.setAttribute('viewBox', '0 0 300 54'); svg.setAttribute('aria-hidden', 'true'); p.setAttribute('d', dpath); svg.append(p);
    return h('button', { class: 'ex', type: 'button', onclick: () => openExample(e) }, svg,
             h('b', {}, `Encounter ${e.encounter}`),
             h('span', {}, `${when ? when.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }) : ''} · off Hawai‘i`),
             h('span', {}, `${e.seconds} s · ${e.n_whistles} whistles`), h('em', {}, 'Listen →'));
  }

  /* the arrival from the film: a whistle line draws itself across, then the heading rises (CSS .anim / .in) */
  /* The film ends under a tide: four swells roll in over it, back ones first, each rising as the page comes up and
     rolling sideways with time and with the scroll. The front swell is the page's own water; its crest is a whistle
     that draws itself as it arrives. Runs only while the tide is on screen. */
  const SWELLS = [   // rest height (fraction of the space above the page edge), amplitude, wavelengths, speeds, lead, colour
    { y: 0.18, a: 0.055, k: [1.1, 2.3], v: [0.018, -0.011], ph: 0.4, lead: 0.0, c: [52, 124, 166, 0.2], hl: 0.26 },
    { y: 0.36, a: 0.06, k: [0.8, 1.9], v: [-0.014, 0.02], ph: 2.1, lead: 0.12, c: [34, 98, 140, 0.32], hl: 0.26 },
    { y: 0.55, a: 0.05, k: [1.3, 2.7], v: [0.022, -0.016], ph: 4.0, lead: 0.24, c: [20, 78, 118, 0.55], hl: 0.32 },
    { y: 0.74, a: 0.0, k: [1.6, 3.1], v: [0.012, -0.02], ph: 1.3, lead: 0.36, c: [12, 63, 102, 1], hl: 0 }  // front: the whistle; the page's --sea0
  ];
  function setupTide() {
    const lab = $('#try'), svg = $('#tide'), cv = $('#tide-c'), cx = cv.getContext('2d');
    const crest = svg.querySelectorAll('.crest');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches, TAU = Math.PI * 2, n = 160;
    const sm = (a, b, v) => { const t = Math.min(1, Math.max(0, (v - a) / (b - a))); return t * t * (3 - 2 * t); };
    let W = 0, H = 0, E = 0, on = false, raf = 0, last = -1e9, paint = [];
    // the swells are soft-edged colour: one canvas pixel per CSS pixel is plenty, and it keeps the fill cost low on an
    // integrated GPU; only the whistle crest (svg) stays sharp
    const measure = () => {
      W = svg.clientWidth; H = svg.clientHeight; E = H * 70 / 98;     // E: the page's top edge inside the box
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      cv.width = W; cv.height = H;
      // each swell fades out below the page's edge, into the page's own water
      const fade = (rgb, a) => { const g = cx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, `rgba(${rgb},${a})`);
                                 g.addColorStop(E / H, `rgba(${rgb},${a})`); g.addColorStop(1, `rgba(${rgb},0)`); return g; };
      paint = SWELLS.map(w => ({ fill: fade(w.c.slice(0, 3).join(), w.c[3]), line: w.hl && fade('196,230,244', w.hl) }));
    };
    const draw = now => {
      const t = reduced ? 0 : now / 1000, r = lab.getBoundingClientRect(), vh = innerHeight;
      const s = Math.min(1.2, Math.max(0, (vh - r.top) / vh));        // 0: the page edge at the bottom of the screen, 1: at the top
      cx.clearRect(0, 0, W, H); cx.lineWidth = 1.2;
      SWELLS.forEach((w, i) => {
        const rise = sm(w.lead, w.lead + 0.62, s), base = E * (1 - rise * (1 - w.y)) + 4, front = i === 3;
        const amp = E * (front ? 0.11 : w.a) * (0.3 + 0.7 * rise), slide = s * 0.35 * (i % 2 ? -1 : 1);
        const y0 = front ? Math.min(base, E - amp * 1.32) : base;    // the front swell never dips below the page's edge
        const pts = [];
        for (let j = 0; j <= n; j++) {
          const u = j / n;
          let y;
          if (front) {    // the whistle: rising loops, gently breathing
            y = -Math.sin(3.3 * Math.PI * u + 0.4 + 0.25 * Math.sin(t * 0.21)) * (1 - 0.2 * u) * 0.9 + 0.55 * (u - 0.5)
                + 0.12 * Math.sin(TAU * (w.k[1] * u + w.v[1] * t * 4) + w.ph);
          } else {
            y = 0.62 * Math.sin(TAU * (w.k[0] * u + w.v[0] * t * 4 + slide) + w.ph) + 0.38 * Math.sin(TAU * (w.k[1] * u + w.v[1] * t * 4 - slide) + 2 * w.ph);
          }
          pts.push(u * W, y0 + amp * y);
        }
        cx.beginPath(); cx.moveTo(pts[0], pts[1]);
        for (let j = 2; j < pts.length; j += 2) cx.lineTo(pts[j], pts[j + 1]);
        if (paint[i].line) { cx.strokeStyle = paint[i].line; cx.stroke(); }
        cx.lineTo(W, H); cx.lineTo(0, H); cx.closePath(); cx.fillStyle = paint[i].fill; cx.fill();
        if (front) {
          let d = '';
          for (let j = 0; j < pts.length; j += 2) d += `${j ? 'L' : 'M'}${pts[j].toFixed(1)} ${pts[j + 1].toFixed(1)}`;
          crest.forEach(p => p.setAttribute('d', d));
        }
      });
    };
    // the swells drift slowly: 60 redraws a second are enough even on a 240 Hz screen (the scroll itself stays native)
    const loop = now => { if (now - last > 15) { last = now; draw(now); } raf = on && !reduced ? requestAnimationFrame(loop) : 0; };
    measure(); draw(performance.now());
    addEventListener('resize', () => { measure(); draw(performance.now()); });
    addEventListener('scroll', () => { if (!raf) requestAnimationFrame(draw); }, { passive: true });
    new IntersectionObserver(es => {
      on = es[es.length - 1].isIntersecting;
      if (on && !raf && !reduced) raf = requestAnimationFrame(loop);
    }).observe(svg);
    lab.classList.add('anim');
    const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { lab.classList.add('in'); io.disconnect(); } },
                                        { rootMargin: '0px 0px -35% 0px' });
    io.observe(lab);
  }

  /* The water of the page: light rays fanning down from the surface, the surface's shimmer near the top and marine snow
     at two depths, all in one fragment shader. It renders at about 1000 px across, 30 times a second, and is stretched
     to the screen: soft light needs no more, and the GPU composites one layer. Dims as you go deeper; pauses off screen. */
  const SEA_FS = `precision mediump float;
uniform vec2 R; uniform float T, D;
float h1(float n) { return fract(sin(n * 127.1) * 43758.5453); }
vec2 h2(vec2 p) { return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453); }
float n1(float x) { float i = floor(x), f = fract(x); return mix(h1(i), h1(i + 1.0), f * f * (3.0 - 2.0 * f)); }
float snow(vec2 p, float cells, float speed, float r, float t) {
  vec2 q = p * cells + vec2(0.3 * sin(t * 0.13 + p.y * 3.0), -t * speed);   // sinking, with a slow sideways sway
  vec2 c = floor(q), f = fract(q), h = h2(c);
  if (h.x > 0.42) return 0.0;                                                // most cells are empty water
  float d = length(f - (0.2 + 0.6 * h2(c + 7.0)));
  return smoothstep(r, r * 0.25, d) * (0.35 + 0.65 * h.y);
}
void main() {
  vec2 uv = gl_FragCoord.xy / R; uv.y = 1.0 - uv.y;                        // y down from the top of the view
  float a = R.x / R.y; vec2 p = vec2(uv.x * a, uv.y); float t = T;
  // rays: brightness varying with the angle from a light source above the view, drifting and breathing
  vec2 d = p - vec2(0.52 * a + 0.08 * sin(t * 0.05), -0.55);
  float ang = atan(d.x, d.y);
  float rays = pow(n1(ang * 11.0 + t * 0.045), 3.0) * 0.65 + pow(n1(ang * 23.0 - t * 0.07 + 5.0), 4.0) * 0.45;
  rays *= smoothstep(-0.02, 0.2, uv.y) * (1.0 - smoothstep(0.2, 0.95, uv.y)) * (0.8 + 0.2 * sin(t * 0.3 + ang * 7.0));
  // the surface's shimmer: interfering ripples, bright where they cancel, strongest at the top
  vec2 w = p * 7.0 + vec2(sin(p.y * 4.0 + t * 0.4), cos(p.x * 3.0 - t * 0.3)) * 0.35;
  float v = sin(w.x + t * 0.5) + sin(0.6 * w.x + 0.8 * w.y - t * 0.37) + sin(-0.5 * w.x + 0.9 * w.y + t * 0.29);
  float shimmer = pow(1.0 - smoothstep(0.0, 0.55, abs(v)), 3.0) * (1.0 - smoothstep(0.0, 0.45, uv.y));
  float s = snow(p, 26.0, 0.012, 0.09, t) * 0.55 + snow(p + 3.7, 11.0, 0.022, 0.07, t) * 0.8;
  float L = rays * 0.19 * (1.0 - 0.75 * D) + shimmer * 0.05 * (1.0 - 0.8 * D) + s * 0.3 * (1.0 - 0.4 * D);
  gl_FragColor = vec4(vec3(0.81, 0.93, 0.97) * L, L);
}`;
  function setupSea() {
    const lab = $('#try'), view = $('#sea-view'), cv = $('#sea-gl');
    const gl = cv.getContext('webgl', { alpha: true, antialias: false, depth: false, powerPreference: 'low-power' });
    const sh = (type, src) => { const o = gl.createShader(type); gl.shaderSource(o, src); gl.compileShader(o); return o; };
    let prog = null;
    if (gl) {
      prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, 'attribute vec2 v; void main() { gl_Position = vec4(v, 0.0, 1.0); }'));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, SEA_FS));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) prog = null;
    }
    if (!prog) { cv.remove(); view.classList.add('static'); return; }
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const uR = gl.getUniformLocation(prog, 'R'), uT = gl.getUniformLocation(prog, 'T'), uD = gl.getUniformLocation(prog, 'D');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const size = () => { const k = Math.min(1, 1000 / view.clientWidth); cv.width = Math.round(view.clientWidth * k); cv.height = Math.round(view.clientHeight * k);
                         gl.viewport(0, 0, cv.width, cv.height); gl.uniform2f(uR, cv.width, cv.height); };
    let on = false, raf = 0, last = -1e9;
    const draw = now => {
      gl.uniform1f(uT, reduced ? 20 : now / 1000);
      gl.uniform1f(uD, Math.min(1, Math.max(0, -lab.getBoundingClientRect().top / (innerHeight * 2.4))));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    const loop = now => { if (now - last > 32) { last = now; draw(now); } raf = on && !reduced ? requestAnimationFrame(loop) : 0; };
    size(); draw(performance.now());
    addEventListener('resize', () => { size(); draw(performance.now()); });
    if (reduced) addEventListener('scroll', () => requestAnimationFrame(draw), { passive: true });
    new IntersectionObserver(es => { on = es[es.length - 1].isIntersecting; if (on && !raf) raf = requestAnimationFrame(loop); }).observe(lab);
  }

  async function openExample(e) {
    if (S.api) return submit(fetch('api/analyze-example', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                         body: JSON.stringify({ example: e.id, n_samples: +$('#nsamples').value }) }));
    showResult(e.result);
  }

  /* ------------------------------------------------------------------ upload and jobs */
  function pick(f) {
    S.file = f || null;
    $('#drop-t').replaceChildren(...(f ? [h('b', {}, f.name), ` · ${(f.size / 1048576).toFixed(1)} MB`]
                                       : ['Drop a recording here, or ', h('u', {}, 'choose a file')]));
    $('#run').disabled = !f || !S.api;
  }
  function bindUpload() {
    const d = $('#drop'), inp = $('#file');
    inp.addEventListener('change', () => pick(inp.files[0]));
    ['dragenter', 'dragover'].forEach(ev => d.addEventListener(ev, e => { e.preventDefault(); if (S.api) d.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => d.addEventListener(ev, e => { e.preventDefault(); d.classList.remove('over'); }));
    d.addEventListener('drop', e => { if (S.api && e.dataTransfer.files.length) pick(e.dataTransfer.files[0]); });
    $('#run').addEventListener('click', () => {
      if (!S.file) return;
      const fd = new FormData();
      fd.append('file', S.file); fd.append('channel', $('#channel').value || '0'); fd.append('n_samples', $('#nsamples').value || '0');
      submit(fetch('api/analyze', { method: 'POST', body: fd }));
    });
  }
  /* a job's progress line sits next to the control that started it (#job for analyses, #gen-job for generation) */
  function jobLine(el, text, frac, err) {
    el.hidden = false; el.classList.toggle('err', !!err);
    el.querySelector('span').textContent = text;
    el.querySelector('i').style.width = frac == null ? '' : `${Math.round(frac * 100)}%`;
    el.classList.toggle('busy', frac == null && !err);
  }
  async function submit(req, onDone = showResult, el = $('#job'), btn = $('#run')) {
    btn.disabled = true; jobLine(el, el.id === 'job' ? 'Uploading…' : 'Starting…', null);
    try {
      const r = await req, body = await r.json();
      if (!r.ok) throw new Error(body.detail || r.statusText);
      const id = body.job;
      for (;;) {
        await new Promise(res => setTimeout(res, 400));
        const j = await (await fetch(`api/jobs/${id}`)).json();
        if (j.state === 'error') throw new Error(j.error);
        if (j.state === 'done') { el.hidden = true; await onDone(j.result.replace(/^\//, '')); break; }
        const stage = STAGES[j.state === 'queued' ? 'queued' : j.stage] || 'Working';
        jobLine(el, stage + (j.frac != null && j.stage === 'sound' ? ` · ${Math.round(j.frac * 100)}%` : '') + '…', j.frac);
      }
    } catch (e) { jobLine(el, String(e.message || e), null, true); }
    btn.disabled = btn.id === 'run' ? !S.file || !S.api : !(S.api && S.status.model.ready.generate);
  }

  /* ------------------------------------------------------------------ one analysis */
  async function showResult(url) {
    const r = await (await fetch(url)).json();
    S.result = r; S.base = new URL(url, location.href).href; S.sel = 0;
    const inp = r.input, ws = r.whistles;
    const rej = r.rejected.length ? ` · ${r.rejected.length} not usable (too long or too wide for the model's images)` : '';
    $('#sum').replaceChildren(
      h('b', {}, inp.name), ` · ${inp.duration_s.toFixed(1)} s · ${inp.fs_in_hz / 1000} kHz · channel ${inp.channel_index} of ${inp.n_channels} · `,
      h('b', {}, `${ws.length} whistle${ws.length === 1 ? '' : 's'}`), rej,
      ...inp.warnings.map(w => h('span', { class: 'warn' }, ' · ' + w)),
      S.api ? h('a', { class: 'dl', href: S.base.replace(/\/result\.json$/, '.zip') }, 'download all') : null);
    drawOverview(r);
    drawList(r);
    select(ws.length ? 0 : -1);
    const res = $('#results'); res.hidden = false;
    res.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function drawOverview(r) {
    const o = r.overview, t1 = o.t1_s, fl = o.f_lo_hz / 1000, fh = o.f_hi_hz / 1000;
    paint($('#ov-img'), fileUrl(o.png));
    const svg = $('#ov-svg'), ns = 'http://www.w3.org/2000/svg';
    svg.setAttribute('viewBox', `0 0 ${t1} ${fh - fl}`);
    svg.replaceChildren();
    const rect = (x0, x1, f0, f1, cls) => {
      const e = document.createElementNS(ns, 'rect');
      e.setAttribute('x', x0); e.setAttribute('width', Math.max(0.02, x1 - x0));
      e.setAttribute('y', fh - f1 / 1000 - 0.3); e.setAttribute('height', (f1 - f0) / 1000 + 0.6);
      e.setAttribute('class', cls);
      svg.append(e); return e;
    };
    for (const q of r.rejected) {
      const e = rect(q.start_s, q.end_s, q.f_min_hz, q.f_max_hz, 'rej');
      const t = document.createElementNS(ns, 'title'); t.textContent = `not usable: ${q.detail}`; e.append(t);
    }
    r.whistles.forEach(w => {
      const e = rect(w.start_s, w.end_s, w.f_min_hz, w.f_max_hz, 'w');
      e.dataset.i = w.i;
      const t = document.createElementNS(ns, 'title'); t.textContent = `whistle ${w.i + 1}`; e.append(t);
      e.addEventListener('click', () => {                 // a box is the whistle: select it and hear it as recorded
        select(w.i);
        const b = $('#detail .play');
        if (b && !b.disabled) b.click();
      });
    });
    const step = [0.5, 1, 2, 5, 10, 20, 30][[0.5, 1, 2, 5, 10, 20, 30].findIndex(s => t1 / s <= 10)] || 30;
    const tAx = $('#ov-t'); tAx.replaceChildren();
    for (let t = 0; t <= t1 + 1e-6; t += step) tAx.append(h('span', { style: `left:${(t / t1) * 100}%` }, `${+t.toFixed(1)} s`));
    const fAx = $('.ov-f'); fAx.replaceChildren();
    for (let f = Math.ceil(fl / 5) * 5; f <= fh; f += 5) fAx.append(h('span', { style: `bottom:${((f - fl) / (fh - fl)) * 100}%` }, `${f} kHz`));
  }

  function drawList(r) {
    const tb = $('#wtable tbody'); tb.replaceChildren();
    if (!r.whistles.length) { tb.append(h('tr', {}, h('td', { colspan: 6, class: 'none' }, 'No whistles found in this recording.'))); return; }
    for (const w of r.whistles) {
      const tr = h('tr', { tabindex: 0, 'aria-selected': 'false', onclick: () => select(w.i),
                           onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(w.i); }
                                             if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); select(Math.max(0, Math.min(r.whistles.length - 1, w.i + (e.key === 'ArrowDown' ? 1 : -1))), true); } } },
                   h('td', {}, w.i + 1), h('td', {}, `${w.start_s.toFixed(2)} s`), h('td', {}, `${Math.round(w.duration_ms)} ms`),
                   h('td', {}, `${kHz(w.fc_hz)} kHz`), h('td', {}, `${kHz(w.f_min_hz)}–${kHz(w.f_max_hz)}`),
                   h('td', {}, w.type ? `#${w.type.component}` : '—'));
      tb.append(tr);
    }
  }

  function select(i, focus) {
    const r = S.result; S.sel = i; stop();
    document.querySelectorAll('#wtable tbody tr').forEach((tr, k) => { tr.setAttribute('aria-selected', k === i ? 'true' : 'false'); if (k === i && focus) tr.focus(); });
    document.querySelectorAll('#ov-svg rect.w').forEach(e => e.classList.toggle('sel', +e.dataset.i === i));
    const d = $('#detail');
    if (i < 0) { d.replaceChildren(h('p', { class: 'lab-note' }, 'Nothing to show: no whistle in this recording could be turned into a model image.')); return; }
    const w = r.whistles[i], m = r.model;
    const img = (name, alt) => shot(fileUrl(name), alt);
    const rec = h('div', { class: 'card' }, h('h4', {}, `Whistle ${i + 1} · as recorded`), img(w.input.png, 'the whistle as the model sees it'),
                  h('div', { class: 'plays' }, playBtn('Recording', w.recording && fileUrl(w.recording.wav), null, false),
                    playBtn('Its shape as a tone', w.input.sound && fileUrl(w.input.sound.wav), w.input.sound)),
                  h('p', { class: 'meta' }, `${w.start_s.toFixed(2)}–${w.end_s.toFixed(2)} s · ${Math.round(w.duration_ms)} ms`, h('br'),
                    `centre ${kHz(w.fc_hz)} kHz · ${kHz(w.f_min_hz)}–${kHz(w.f_max_hz)} kHz`));
    const pending = (title, text) => h('div', { class: 'card pending' }, h('h4', {}, title), h('p', {}, text));
    let recon;
    if (w.recreation) {
      const rc = w.recreation;
      recon = h('div', { class: 'card' }, h('h4', {}, 'Recreated by the model'), img(rc.png, 'model recreation'),
                h('div', { class: 'plays' }, playBtn('Recreation', rc.sound && fileUrl(rc.sound.wav), rc.sound)),
                h('p', { class: 'meta' }, rc.contour_error_hz != null ? `Its shape stays within ${Math.round(rc.contour_error_hz)} Hz of the original on average` : 'No clear shape in the recreation',
                  h('br'), `PSNR ${rc.psnr_db} dB`));
    } else recon = pending('Recreated by the model', 'Appears once the trained model is in place (see the note above).');
    let same;
    if (w.samples && w.samples.length) {
      const gmm = r.samples_method === 'gmm';
      same = h('div', { class: 'card' }, h('h4', {}, gmm ? `New whistles · family ${w.type.component}` : 'Close variations'),
               h('div', { class: 'thumbs' }, w.samples.map((s, j) => h('figure', {}, shot(fileUrl(s.png), `new whistle ${j + 1}`),
                 playBtn(`New ${j + 1}`, s.sound && fileUrl(s.sound.wav), s.sound)))),
               h('p', { class: 'meta' }, gmm ? `Drawn from the same island on the model's map (sure of the family: ${Math.round(w.type.posterior * 100)}%)`
                 : 'Drawn from the encoder\'s own spread around this whistle: a standard VAE step, not the paper\'s generation method (fit the Gaussian mixture for that).'));
    } else same = pending('New whistles of the same family', m.ready.recreate ? 'Set "New whistles per whistle" above 0.'
                                                                           : 'Appear once the trained model and its map of whistle families are in place.');
    const opts = h('div', { class: 'popts' },
      h('label', {}, h('input', { type: 'checkbox', checked: S.slow, onchange: e => { S.slow = e.target.checked; stop(); } }), 'Slow ×2 (same pitch)'),
      h('label', {}, 'Pitch shift ', h('input', { type: 'range', min: -3000, max: 3000, step: 100, value: S.shift,
        oninput: e => { S.shift = +e.target.value; e.target.nextSibling.textContent = `${S.shift > 0 ? '+' : ''}${S.shift} Hz`; stop(); } }),
        h('span', {}, `${S.shift > 0 ? '+' : ''}${S.shift} Hz`)),
      h('span', { class: 'lab-note' }, 'Applies to the tones, not the recording.'));
    d.replaceChildren(h('div', { class: 'cards' }, rec, recon, same), opts);
  }

  /* ------------------------------------------------------------------ generation */
  function setupGenerate() {
    const ready = S.api && S.status.model.ready.generate, note = $('#gen-note'), sel = $('#gen-cond');
    if (S.api) for (const c of S.status.conditions) sel.append(h('option', { value: c.no, disabled: !S.status.model.ready.conditional }, `${c.no}. ${c.label}`));
    $('#gen-run').disabled = !ready;
    note.textContent = !S.api ? 'Imagining new whistles needs the analysis server (python whistle.py serve).'
      : !ready ? 'Available once the model has finished training.'
      : !S.status.model.ready.conditional ? 'Asking for a certain kind becomes available once the model has sampled every family.' : '';
    note.title = S.api ? S.status.model.message : '';
    $('#gen-run').addEventListener('click', () => {
      const fc = $('#gen-fc').value;
      submit(fetch('api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n: +$('#gen-n').value || 8, seed: +$('#gen-seed').value || 0,
                               condition: sel.value ? +sel.value : null, fc_hz: fc ? +fc : null }) }),
             showGenerated, $('#gen-job'), $('#gen-run'));
    });
  }
  async function showGenerated(url) {
    const r = await (await fetch(url)).json();
    const base = new URL(url, location.href).href;      // its own base: the analysis above keeps S.base
    const grid = $('#gen-grid');
    const zero = 65, hz = 93.75, ms = 1000 * 1024 / 192000;
    grid.replaceChildren(...r.items.map(it => {
      const f = it.features, fr = f.f_min != null ? `${kHz(it.fc_hz + (f.f_min - zero) * hz)}–${kHz(it.fc_hz + (f.f_max - zero) * hz)} kHz` : '';
      return h('figure', {}, shot(fileUrl(it.png, base), `generated whistle ${it.i + 1}`),
               playBtn(`Whistle ${it.i + 1}`, it.sound && fileUrl(it.sound.wav, base), it.sound),
               h('figcaption', {}, f.L != null ? `${Math.round(f.L * ms)} ms · ${fr}` : 'empty image', it.component != null ? ` · k ${it.component}` : ''));
    }));
    $('#gen-note').textContent = r.condition ? `${r.condition.label}: ${r.n} accepted (acceptance ${(r.acceptance_rate * 100).toPrecision(2)}%). Centre pitch ${r.fc}.`
      : `${r.n} whistles from the mixture. Centre pitch ${r.fc}.`;
  }

  addEventListener('DOMContentLoaded', () => { bindUpload(); setupSea(); setupTide(); init(); });
})();
