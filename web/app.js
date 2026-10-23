/**
 * Cetacean Acoustic Lab — scroll is the editor.
 *
 * Every visual decision is a pure function of `pos` (section + progress):
 * which shot is on screen, how dark the water is, which words are visible,
 * what state the research instrument is in. Footage plays at its own speed;
 * it is never time-scrubbed. Text lives in one fixed layer and cross-fades in
 * place, so sections only set how long each moment lasts.
 *
 * Video: two 4K files. At most two clips play at once. On Windows the
 * browser decodes video on the same GPU thread that draws the page, so any
 * decoder work off screen (preroll, seeks) costs frames on screen. Hence:
 * every later clip is prerolled while the opening frame holds still, before
 * anyone scrolls; afterwards the only seeks happen while a shot is fully
 * invisible (rewinding for a fresh entrance). The depth shade is
 * folded into plane opacity, and empty full-screen layers are hidden, so a
 * scroll frame composites as little as possible.
 */
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const sm = (a, b, v) => { if (b <= a) return v >= b ? 1 : 0; const t = clamp((v - a) / (b - a)); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;
const $ = s => document.querySelector(s);
const curve = (K, p) => {
  if (p <= K[0][0]) return K[0][1];
  for (let k = 1; k < K.length; k++) if (p <= K[k][0]) return lerp(K[k - 1][1], K[k][1], sm(K[k - 1][0], K[k][0], p));
  return K[K.length - 1][1];
};
const win = (w, p) => sm(w[0], w[1], p) * (1 - sm(w[2], w[3], p));

class Film {
  constructor() {
    document.body.classList.add('cam-on');
    try {
      this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.secs = [...document.querySelectorAll('.sec')];
      this.blocks = [...document.querySelectorAll('[data-t]')].map(el => ({ el, t: window.TEXT[el.dataset.t], o: -1 }));
      this.veilEl = $('#veil'); this.bar = $('#progress-bar');
      this.links = [...document.querySelectorAll('[data-stop]')];
      this.cv = $('#sig'); this.cx = this.cv.getContext('2d');
      this.target = this.cur = this.cam1 = scrollY; this.run = false; this.playT = null; this.last = 0;
      this.buildShots(); this.prepare();
      try { this.engine = new CetaceanAudioEngine(); } catch (e) { console.error('[film] audio engine failed:', e); this.engine = null; }
      this.bindListen();
      try { this.buildSignal(); } catch (e) { console.error('[film] signal build failed:', e); this.cont = null; }
      this.measure(); this.bind(); this.read(); this.wake();
      // canvas text uses the web fonts: redraw once they are ready
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { this.dirty = true; this.wake(); });
    } catch (e) {
      console.error('[film] boot failed, static fallback:', e);
      document.body.classList.remove('cam-on');
    }
  }

  buildShots() {
    const R = this.reduced, cam = $('#camera');
    this.shots = (window.SHOTS || []).map((m, k) => {
      const el = document.createElement('div');
      el.className = 'plane'; el.dataset.shot = m.id;
      const s = { m, el, w: 0, wanted: false, warm: false, op: -1, tf: '' };
      if (m.image) {
        const img = new Image(); img.decoding = 'async'; img.src = m.image; img.alt = '';
        // decode now, off the main thread, not on the first frame it is shown
        if (img.decode) img.decode().catch(() => {});
        el.appendChild(img); s.media = img;
      } else {
        const v = document.createElement('video');
        v.muted = true; v.playsInline = true; v.setAttribute('muted', ''); v.setAttribute('playsinline', '');
        v.disablePictureInPicture = true; v.loop = !!m.loop; v.poster = m.poster;
        // only the opening shot loads at once; the rest wait for warm()
        v.preload = !R && k === 0 ? 'auto' : 'none';
        v.src = m.file;
        if (m.start) v.addEventListener('loadedmetadata', () => { if (!s.wanted) v.currentTime = m.start; }, { once: true });
        v.addEventListener('error', () => { s.dead = true; console.error('[film] video failed:', m.file); });
        el.appendChild(v); s.media = s.video = v;
      }
      s.media.style.objectPosition = `${m.pan[0]}% ${m.pan[1]}%`;
      s.w0 = m.windows[0][0]; s.w1 = m.windows[m.windows.length - 1][3];
      cam.appendChild(el);
      return s;
    });
  }

  /* The opening frame holds still until every later clip has prerolled (and
     sought to its entry point), so their decoders spin up before anyone
     scrolls, never in the middle of a scroll. */
  prepare() {
    const first = this.shots[0], later = this.shots.filter((s, k) => k > 0 && s.video);
    if (this.reduced || !first || !first.video || !later.length) { this.ready = true; return; }
    let left = later.length;
    const go = () => { if (this.ready) return; this.ready = true; this.dirty = true; this.wake(); };
    for (const s of later) {
      const done = () => { if (s.prepped) return; s.prepped = true; if (--left === 0) go(); };
      this.warm(s);
      s.video.addEventListener(s.m.start ? 'seeked' : 'canplay', done, { once: true });
      s.video.addEventListener('error', done, { once: true });
    }
    setTimeout(go, 1500);   // never hold the opening frame longer than this
  }
  warm(s) { s.warm = true; s.video.preload = 'auto'; }

  measure() {
    this.vh = innerHeight; this.W = innerWidth; this.H = innerHeight;
    this.max = Math.max(1, document.documentElement.scrollHeight - innerHeight);
    const sy = scrollY;
    this.top = this.secs.map(s => s.getBoundingClientRect().top + sy);
    this.h = this.secs.map(s => s.offsetHeight);
    const lab = $('#try');
    this.labTop = lab ? lab.getBoundingClientRect().top + sy : Infinity;
    const d = Math.min(2, devicePixelRatio || 1);
    this.cv.width = this.W * d; this.cv.height = this.H * d; this.cx.setTransform(d, 0, 0, d, 0, 0);
    // The instrument sits right of the research column, clear of the index.
    const col = $('.block-top'), gut = this.W * .075;
    const colRight = col ? gut + Math.min(col.offsetWidth || this.W * .34, this.W * .34) : this.W * .42;
    this.x0 = Math.max(this.W * .47, colRight + 72); this.x1 = this.W - gut;
    this.y0 = this.H * .3; this.y1 = this.H * .78;
    this.fs = Math.round(clamp(this.W * .0055, 12, 15));   // plot type scales with the display
    this.dirty = true;
  }
  span(i) { return i < this.secs.length - 1 ? this.h[i] : Math.max(1, this.h[i] - this.vh); }
  posAt(y) {
    let i = 0; for (let k = 0; k < this.secs.length; k++) if (y >= this.top[k] - 1) i = k;
    return i + clamp((y - this.top[i]) / this.span(i));
  }
  yAt(pos) {
    const i = clamp(Math.floor(pos), 0, this.secs.length - 1);
    return this.top[i] + (pos - i) * this.span(i);
  }

  read() { this.target = scrollY; this.bar.style.transform = `scaleX(${clamp(this.target / this.max).toFixed(4)})`; }
  bind() {
    addEventListener('scroll', () => { this.read(); this.wake(); }, { passive: true });
    addEventListener('resize', () => { this.measure(); this.read(); this.wake(); });
    // An index link moves focus too, like an in-page anchor: a block takes it
    // once it is on screen (hidden blocks cannot), so Tab then reaches its controls.
    this.links.forEach(a => a.addEventListener('click', e => {
      e.preventDefault();
      const p = (window.STOPS || {})[a.dataset.stop], el = document.getElementById(a.dataset.stop);
      const y = a.dataset.stop === 'try' ? this.labTop : p != null ? this.yAt(p) : null;
      if (y == null) return;
      this.pendingFocus = null;
      if (el && this.blocks.some(b => b.el === el)) this.pendingFocus = el;
      else if (el) el.focus({ preventScroll: true });
      this.glideTo(y); this.dirty = true; this.wake();
    }));
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.shots.forEach(s => s.video && s.video.pause());
      this.shots.forEach(s => s.wanted = false); this.wake();
    });
    const kick = () => { this.shots.forEach(s => s.wanted = false); this.wake(); };
    ['pointerdown', 'keydown', 'wheel'].forEach(ev => addEventListener(ev, kick, { once: true, passive: true }));
  }
  wake() { if (!this.run) { this.run = true; this.last = 0; requestAnimationFrame(t => this.tick(t)); } }

  glideTo(y) {
    if (this.reduced) { scrollTo(0, y); return; }
    const from = scrollY, dist = y - from;
    if (Math.abs(dist) < 2) return;
    if (this.glideCancel) this.glideCancel();
    const my = (this.seq = (this.seq || 0) + 1); let raf = 0;
    const t0 = performance.now(), dur = Math.min(1700, 600 + Math.abs(dist) / 5);
    const evs = ['wheel', 'keydown', 'pointerdown'], off = () => evs.forEach(ev => removeEventListener(ev, cancel));
    const cancel = () => { if (this.seq !== my) return; cancelAnimationFrame(raf); this.seq++; this.glideCancel = null; off(); };
    this.glideCancel = cancel;
    evs.forEach(ev => addEventListener(ev, cancel, { passive: true }));
    const step = now => {
      if (this.seq !== my) return;
      const t = Math.min(1, (now - t0) / dur), e = t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      scrollTo(0, from + dist * e);
      if (t < 1) raf = requestAnimationFrame(step); else { this.glideCancel = null; off(); }
    };
    raf = requestAnimationFrame(step);
  }

  tick(now) {
    const dt = this.last ? Math.min(.05, (now - this.last) / 1000) : .016; this.last = now;
    if (this.reduced) this.cur = this.target;
    else {
      const f = 1 - Math.pow(1 - .14, dt * 60);
      this.cam1 += (this.target - this.cam1) * f;
      this.cur += (this.cam1 - this.cur) * f;
      if (Math.abs(this.target - this.cur) < .3) this.cur = this.cam1 = this.target;
    }
    this.render(now);
    const moving = Math.abs(this.target - this.cur) > .3 || this.playT != null || this.dirty;
    this.dirty = false;
    if (moving && !document.hidden) requestAnimationFrame(t => this.tick(t)); else this.run = false;
  }

  render(now) {
    const pos = this.pos = this.posAt(this.cur), R = this.reduced;

    /* ---- parked: the Try-it page covers the screen, so the film stops drawing and decoding ---- */
    const parked = this.target >= this.labTop - 1;
    if (parked !== this.parked) {
      this.parked = parked;
      $('.world').style.visibility = parked ? 'hidden' : '';
      this.shots.forEach(s => { if (s.video) s.video.pause(); s.wanted = false; });
    }
    this.markIndex(pos);
    if (parked) return;

    /* ---- shots, and the water's depth ----
       The depth shade is the water's own blue laid over the frame, and the
       page behind the planes is that same blue. So instead of a full-screen
       shade layer, each plane's opacity is scaled to give identical pixels:
       top plane o' = (1 - a)·o; below it, o' = (1 - a)·o·Π(1 - o_above) / Π(1 - o'_above). */
    const a = curve(window.SHADE, pos);
    let cover = 1, coverEff = 1;
    for (let k = this.shots.length - 1; k >= 0; k--) {
      const s = this.shots[k], m = s.m;
      const o = s.w = s.dead ? 0 : Math.max(...m.windows.map(w => win(w, pos)));
      const e = coverEff > 1e-4 ? clamp((1 - a) * o * cover / coverEff) : 0;
      cover *= 1 - o; coverEff *= 1 - e;
      const on = e > .002;
      if (on !== s.vis) { s.vis = on; s.el.style.visibility = on ? 'visible' : 'hidden'; }
      if (!on) {
        if (s.op !== 0) { s.el.style.opacity = '0'; s.op = 0; }
        // fresh entrance next time: rewind while invisible, only when the
        // viewer has gone back upstream of the shot, and only a loaded clip
        // that has moved (a seek wakes the decoder)
        const v = s.video;
        if (v && s.warm && pos < s.w0 && !s.rewound && (m.restart || m.start) && v.readyState >= 1
            && Math.abs(v.currentTime - (m.start || 0)) > .05) {
          try { v.pause(); v.currentTime = m.start || 0; } catch (_) {}
          s.rewound = true; s.wanted = false;
        }
        continue;
      }
      s.rewound = false;
      const lo = Math.max(0, s.w0), hi = Math.min(4, s.w1);
      const p = clamp((pos - lo) / Math.max(.001, hi - lo));
      const sc = lerp(m.push[0], m.push[1], p);
      const sh = m.shift || [0, 0];
      const op = e.toFixed(3), tf = R ? 'none' : `translate3d(${sh[0]}%,${sh[1]}%,0) scale(${sc.toFixed(4)})`;
      if (op !== s.op) { s.el.style.opacity = op; s.op = op; }
      if (tf !== s.tf) { s.el.style.transform = tf; s.tf = tf; }
    }
    this.playback(pos);

    /* ---- words ---- */
    let low = 0;
    for (const b of this.blocks) {
      const o = win(b.t, pos);
      if (b.el.classList.contains('block-low')) low = Math.max(low, o);
      if (Math.abs(o - b.o) >= .001) {
        b.o = o;
        const on = o > .002;
        b.el.style.visibility = on ? 'visible' : 'hidden';
        b.el.style.opacity = o.toFixed(3);
        if (!R) b.el.style.transform = `translate3d(0,${((1 - o) * 10).toFixed(1)}px,0)`;
      }
      if (this.pendingFocus === b.el && o > .5) { b.el.focus({ preventScroll: true }); this.pendingFocus = null; }
    }
    // an empty full-screen layer still costs a blend every frame: hide it
    const vo = (low * .9).toFixed(3);
    if (vo !== this.vo) { this.vo = vo; this.veilEl.style.opacity = vo; this.veilEl.style.visibility = +vo > .002 ? 'visible' : 'hidden'; }

    /* ---- the research instrument ---- */
    this.drawSignal(pos);
  }

  /* index: the research beat on screen (timings from TEXT), or the Try-it page once it fills half the screen */
  markIndex(pos) {
    const T = window.TEXT, order = ['signal', 'model', 'generate'];
    let cur = null;
    order.forEach((id, k) => { const end = order[k + 1] ? T[order[k + 1]][0] : T[id][3] + .09; if (pos >= T[id][0] - .03 && pos < end) cur = id; });
    if (this.target >= this.labTop - this.vh / 2) cur = 'try';
    const on = this.target >= this.labTop - 110;
    if (on !== this.labOn) { this.labOn = on; document.body.classList.toggle('lab-on', on); }
    if (cur !== this.cur_) {
      this.cur_ = cur;
      this.links.forEach(a => a.dataset.stop === cur ? a.setAttribute('aria-current', 'step') : a.removeAttribute('aria-current'));
    }
  }

  /* play / pause only; at most two decoders; preload ahead */
  playback(pos) {
    if (this.reduced) return;
    for (const s of this.shots) if (s.video && !s.dead && !s.warm && pos > s.w0 - .7 && pos < s.w1 + .3) this.warm(s);
    if (!this.ready) return;   // the opening frame holds until prepare() is done
    const want = new Set(this.shots.filter(s => s.video && s.w > .002 && !s.dead).sort((a, b) => b.w - a.w).slice(0, 2));
    for (const s of this.shots) {
      const v = s.video; if (!v || s.dead) continue;
      const on = want.has(s);
      if (on && (!s.wanted || (v.paused && !(s.m.once && v.ended)))) {
        if (!(s.m.once && v.ended)) { const p = v.play(); if (p && p.catch) p.catch(() => {}); }
        s.wanted = true;
      } else if (!on && s.wanted) { s.wanted = false; v.pause(); }
    }
  }

  /* ---- one object, four states, from the engine's own whistle contours ---- */
  buildSignal() {
    const E = this.engine, D = 1.8, N = this.N = 220;
    if (!E) throw new Error('no engine');
    this.cont = [];
    for (let id = 1; id <= 8; id++) {
      const a = new Float32Array(N + 1);
      for (let i = 0; i <= N; i++) a[i] = E.getFrequencyAtTime(D * i / N, D, id, 9500, 8000);
      this.cont.push(a);
    }
    this.wave = new Float32Array(N + 1);
    for (let i = 0, p = 0; i <= N; i++) { p += this.cont[0][i] / 1000 * .2; this.wave[i] = Math.sin(p) * Math.sin(Math.PI * i / N); }
    this.C = [[.2, .55], [.42, .28], [.66, .62], [.3, .8], [.78, .3], [.52, .52], [.86, .72], [.14, .25]];
  }
  drawSignal(pos) {
    if (!this.cont) return;
    const V = window.VIZ, cx = this.cx, A = win(V.on, pos);
    // the canvas is a full-screen layer: hidden whenever nothing is drawn on it
    if (A < .004) { if (this.cvOn) { cx.clearRect(0, 0, this.W, this.H); this.cv.style.visibility = 'hidden'; this.cvOn = false; } return; }
    if (!this.cvOn) { this.cv.style.visibility = 'visible'; this.cvOn = true; }
    const rev = sm(...V.draw, pos), m1 = sm(...V.pitch, pos), app = sm(...V.appear, pos), m2 = sm(...V.latent, pos), m3 = sm(...V.decode, pos);
    const { x0, x1, y0, y1, N } = this, w = x1 - x0, h = y1 - y0, yc = (y0 + y1) / 2;
    const yF = f => y1 - (f - 3000) / 12000 * h;
    cx.clearRect(0, 0, this.W, this.H); cx.lineJoin = 'round'; cx.lineCap = 'round';

    // frequency axis: only where pitch is meaningful (not under the pressure wave)
    const ax = Math.max(sm(.5, 1, m1) * (1 - m2), m3) * A;
    if (ax > .01) {
      cx.font = `400 ${this.fs}px "IBM Plex Mono", monospace`; cx.textAlign = 'right';
      cx.lineWidth = 1; cx.setLineDash([1, 6]);
      [[5000, '5 kHz'], [10000, '10 kHz'], [15000, '15 kHz']].forEach(([f, l]) => {
        cx.strokeStyle = `rgba(238,244,246,${.22 * ax})`; cx.beginPath(); cx.moveTo(x0, yF(f)); cx.lineTo(x1, yF(f)); cx.stroke();
        cx.fillStyle = `rgba(188,207,216,${.8 * ax})`; cx.fillText(l, x0 - 14, yF(f) + 4);
      });
      cx.setLineDash([]);
      cx.textAlign = 'right'; cx.fillStyle = `rgba(188,207,216,${.6 * ax})`; cx.fillText('time  →', x1, y1 + 26);
    }
    // plot title; every state is a hand-made sketch until the trained model's outputs replace it
    cx.font = `400 ${this.fs}px "IBM Plex Mono", monospace`; cx.textAlign = 'left';
    // one title at a time: the outgoing one fades out before the next fades in (they share a spot)
    const [title, tw] = [['PRESSURE · TIME', 1 - m1], ['PITCH · TIME', m1 * (1 - m2)], ['LATENT SPACE · CLUSTERS', m2 * (1 - m3)], ['DECODED CONTOUR', m3]]
      .reduce((b, t) => t[1] > b[1] ? t : b), ta = sm(.5, 1, tw) * A;
    if (ta > .02) { cx.fillStyle = `rgba(169,207,224,${ta * .95})`; cx.fillText(title, x0, y0 - 34); }
    cx.fillStyle = `rgba(188,207,216,${.55 * A})`; cx.fillText('illustration · not model output', x0, y1 + 26);

    const dots = [];
    for (const id of [3, 4, 5, 6, 7, 8, 1, 2]) {
      const c = this.cont[id - 1], C = this.C[id - 1], cX = x0 + C[0] * w, cY = y0 + C[1] * h;
      let al = id === 1 ? lerp(.95, .35, m2) : id === 2 ? lerp(app * .3, .95, m2) : app * lerp(.32, .2, m3);
      al *= A; if (al < .01) continue;
      const col = id === 2 ? m2 * (1 - m3) : m2, cnt = id === 1 ? Math.floor(N * rev) : N;
      const pts = [];
      for (let k = 0; k <= cnt; k++) {
        const px = x0 + k / N * w, py = yF(c[k]);
        const yy = id === 1 ? lerp(yc + this.wave[k] * h * .28, py, m1) : py;
        pts.push(lerp(px, cX, col), lerp(yy, cY, col));
      }
      if (pts.length < 4) continue;
      const path = () => { cx.beginPath(); for (let k = 0; k < pts.length; k += 2) k ? cx.lineTo(pts[k], pts[k + 1]) : cx.moveTo(pts[0], pts[1]); };
      if (col < .98) { path(); cx.strokeStyle = `rgba(214,236,244,${al})`; cx.lineWidth = id === 1 || id === 2 ? 1.6 : 1; cx.stroke(); }
      if (col > .02) dots.push([cX, cY, al, col, id]);
    }
    cx.font = `400 ${this.fs - 2}px "IBM Plex Mono", monospace`; cx.textAlign = 'left';
    for (const [X, Y, al, col, id] of dots) {
      // the decoded contour stands alone: the other clusters leave the plot
      const sel = id === 2, f = sel ? 1 : 1 - m3;
      if (f < .01) continue;
      cx.strokeStyle = `rgba(214,236,244,${.22 * col * A * f})`; cx.lineWidth = 1;
      cx.beginPath(); cx.arc(X, Y, (sel ? 32 : 22) * col, 0, 6.283); cx.stroke();
      cx.fillStyle = `rgba(238,248,252,${al * col * f})`; cx.beginPath(); cx.arc(X, Y, sel ? 3.4 : 2.4, 0, 6.283); cx.fill();
      cx.fillStyle = `rgba(188,207,216,${.55 * col * A * (1 - m3)})`; cx.fillText('k' + id, X + 8, Y - 8);
    }
    if (this.playT != null && m3 > .4) {
      const px = x0 + this.playT * w, py = yF(this.cont[1][Math.round(this.playT * N)]);
      cx.strokeStyle = `rgba(238,244,246,${.35 * m3 * A})`; cx.beginPath(); cx.moveTo(px, y0 - 8); cx.lineTo(px, y1 + 8); cx.stroke();
      cx.fillStyle = `rgba(238,248,252,${m3 * A})`; cx.beginPath(); cx.arc(px, py, 4, 0, 6.283); cx.fill();
    }
  }

  /* exactly one tone; user-gated, stoppable, never automatic */
  bindListen() {
    const btn = $('#btn-listen'); if (!btn) return;
    const lbl = btn.querySelector('.lbl'), fill = btn.querySelector('.ring .fill');
    const note = btn.parentElement && btn.parentElement.querySelector('.note');
    const setFill = f => { if (fill) fill.style.strokeDashoffset = (106.8 * (1 - clamp(f))).toFixed(1); };
    const E = this.engine;
    if (!E) { btn.disabled = true; lbl.textContent = 'Unavailable'; if (note) note.textContent = 'Audio engine unavailable'; return; }
    const ui = st => {
      btn.setAttribute('aria-pressed', st === 'playing' ? 'true' : 'false');
      if (st === 'playing') btn.removeAttribute('aria-busy'); else btn.setAttribute('aria-busy', st === 'busy' ? 'true' : 'false');
      lbl.textContent = st === 'playing' ? 'Stop' : st === 'busy' ? 'Preparing…' : 'Hear it';
      if (st !== 'playing') { this.playT = null; setFill(0); this.dirty = true; this.wake(); }
    };
    E.onPlaybackProgress = t => { this.playT = clamp(t / 1.8); setFill(t / 1.8); this.wake(); };
    E.onPlaybackEnd = () => ui('idle');
    let ready = false, busy = false;
    btn.addEventListener('click', async () => {
      if (busy) return;
      if (E.isPlaying) { E.stop(); ui('idle'); return; }
      busy = true; ui('busy');
      try {
        if (!ready) { await E.synthesizeWhistleBuffer({ componentId: 2, duration: 1.8 }); ready = true; }
        E.play(); ui('playing');
      } catch (e) { ui('idle'); }
      busy = false;
    });
  }
}
addEventListener('DOMContentLoaded', () => {
  try { window.film = new Film(); }
  catch (e) { console.error('[film] fatal:', e); document.body.classList.remove('cam-on'); }
});
