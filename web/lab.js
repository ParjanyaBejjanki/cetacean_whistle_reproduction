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
  function setupArrival() {
    const lab = $('#try'), svg = $('#lab-wave'), paths = svg.querySelectorAll('path');
    const draw = () => {
      const w = svg.clientWidth, hgt = svg.clientHeight;
      svg.setAttribute('viewBox', `0 0 ${w} ${hgt}`);
      let d = '';
      for (let k = 0; k <= 240; k++) {
        const u = k / 240, y = 0.5 - 0.22 * Math.sin(3.3 * Math.PI * u + 0.4) * (1 - 0.2 * u) - 0.32 * (u - 0.5);
        d += `${k ? 'L' : 'M'}${(u * w).toFixed(1)} ${(y * hgt).toFixed(1)}`;
      }
      paths.forEach(p => p.setAttribute('d', d));
    };
    draw(); addEventListener('resize', draw);
    lab.classList.add('anim');
    const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { lab.classList.add('in'); io.disconnect(); } },
                                        { rootMargin: '0px 0px -35% 0px' });
    io.observe(lab);
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

  addEventListener('DOMContentLoaded', () => { bindUpload(); setupArrival(); init(); });
})();
