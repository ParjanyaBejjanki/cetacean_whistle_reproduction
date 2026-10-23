/**
 * Listen → Learn → Imagine as one 3D scene on the #sig canvas: canvas 2D with its own perspective projection, no
 * library. Like the rest of the film, every frame is a pure function of the scroll position (timings: shots.js VIZ).
 * Hand-made to show the idea, not the trained model's output; the corner says so.
 *
 *   rise, tip      A sound landscape: time runs across, pitch into the distance, loudness up. The whistle is the
 *                  glowing ridge. The camera tips over until it looks straight down: the ridge is now a line, the
 *                  whistle's shape (a spectrogram seen from above is exactly that picture).
 *   crowd, gather  Other whistles appear around it; each folds into a point; look-alikes drift together into islands
 *                  while the camera orbits, so the map reads as a space.
 *   dive, unfold   The camera sinks into the first island; a new point appears and unrolls into a new whistle of
 *                  the same family: the one "Hear it" plays (newWhistleHz).
 */
(() => {
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const sm = (a, b, v) => { const t = clamp((v - a) / (b - a)); return t * t * (3 - 2 * t); };
  const lerp = (a, b, t) => a + (b - a) * t;
  const ph = (w, p) => sm(w[0], w[1], p);
  const rng = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };

  /* pitch shapes p(u) in about [-1, 1] for u in [0, 1]; v = {a: amplitude, s: shift} makes a family member */
  const FAMILIES = [
    (u, v) => 0.5 * v.a * Math.sin(3.3 * Math.PI * u + v.s) * (1 - 0.2 * u) + 0.8 * (u - 0.45) * (0.8 + 0.2 * v.a),  // rising loops
    (u, v) => v.a * (-0.85 + 1.7 * Math.pow(u, 0.8 + 0.25 * v.s)),                   // rising
    (u, v) => v.a * (0.8 - 1.6 * Math.pow(u, 0.9)) + 0.12 * Math.sin(6 * u + v.s),    // falling
    (u, v) => v.a * (-0.7 + 1.45 * Math.sin(Math.PI * u)) + 0.08 * v.s,              // arch
    (u, v) => 0.55 * v.a * Math.sin(3 * Math.PI * u + v.s) + 0.25 * u - 0.1          // wave
  ];
  const ISLANDS = [[-0.78, 0.36, -0.12], [0.72, 0.48, 0.36], [-0.18, -0.52, 0.46], [0.82, -0.36, -0.42], [0.02, 0.1, -0.85]];
  const MAIN = { f: 0, v: { a: 1, s: 0 } };              // the whistle we follow
  const NEW = { f: 0, v: { a: 0.82, s: 0.95 } };          // the new whistle of the same family
  const INK = '234,246,251', PALE = '169,207,224', WATER = '4,24,44';

  class ResearchScene {
    constructor() {
      const r = rng(20261024);
      this.R = 30; this.C = 100;                          // landscape rows (pitch) x columns (time)
      this.noise = [];
      for (let i = 0; i < this.R; i++) {                  // smoothed random floor, one row per pitch band
        const raw = Array.from({ length: this.C + 6 }, () => r());
        this.noise.push(Array.from({ length: this.C }, (_, c) => (raw[c] + raw[c + 1] * 2 + raw[c + 2] * 3 + raw[c + 3] * 2 + raw[c + 4]) / 9));
      }
      this.clicks = [0.18, 0.47, 0.83].map(u => -1.3 + 2.6 * u);   // a few broadband clicks, as in real recordings
      this.members = [];
      const gauss = () => (r() + r() + r() - 1.5) * 0.9;
      ISLANDS.forEach((q, f) => {
        for (let j = 0; j < 9; j++) {
          const a = r() * Math.PI * 2, e = Math.acos(2 * r() - 1), rad = 0.55 + 0.65 * r();
          this.members.push({ f, v: { a: 0.75 + 0.45 * r(), s: (r() - 0.5) * 1.4 }, order: r(),
                              p0: [rad * Math.sin(e) * Math.cos(a) * 1.25, rad * Math.cos(e) * 0.8, rad * Math.sin(e) * Math.sin(a)],
                              p1: [q[0] + gauss() * 0.13, q[1] + gauss() * 0.1, q[2] + gauss() * 0.13] });
        }
      });
      this.members.push({ main: true, f: 0, v: MAIN.v, order: 0, p0: [-0.05, 0.05, 0.1],
                          p1: [ISLANDS[0][0] + 0.05, ISLANDS[0][1] + 0.03, ISLANDS[0][2]] });
      this.N = 64;                                        // points per drawn contour
    }

    resize(W, H, box) {
      this.W = W; this.H = H; this.box = box;
      this.ox = (box.x0 + box.x1) / 2; this.oy = (box.y0 + box.y1) / 2;
      this.S = Math.min((box.x1 - box.x0) / 2.75, (box.y1 - box.y0) / 1.45);
      this.fs = Math.round(clamp(W * 0.0068, 12, 16));
    }

    /** the new whistle as frequencies (Hz) for the audio engine: n points over its duration */
    newWhistleHz(n) { return Array.from({ length: n }, (_, i) => 9200 + 2600 * FAMILIES[NEW.f](i / (n - 1), NEW.v)); }

    /* camera: yaw about the vertical axis, then pitch (tilt down); t = point the camera looks at */
    proj(x, y, z, c) {
      x -= c.t[0]; y -= c.t[1]; z -= c.t[2];
      const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw), cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const y2 = y * cp + z1 * sp, z2 = -y * sp + z1 * cp;
      const k = c.f / (c.dist + z2);
      return [this.ox + x1 * k * this.S, this.oy - y2 * k * this.S, k, z2];
    }

    line(cx, pts, alpha, width = 1.7, glow = true) {
      if (alpha < 0.01 || pts.length < 2) return;
      const path = () => { cx.beginPath(); cx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) cx.lineTo(pts[i][0], pts[i][1]); };
      if (glow) { path(); cx.strokeStyle = `rgba(${PALE},${0.16 * alpha})`; cx.lineWidth = width * 4.5; cx.stroke(); }
      path(); cx.strokeStyle = `rgba(${INK},${0.96 * alpha})`; cx.lineWidth = width; cx.stroke();
    }

    label(cx, text, x, y, alpha, align = 'left') {
      if (alpha < 0.02) return;
      cx.textAlign = align; cx.fillStyle = `rgba(214,236,244,${0.88 * alpha})`; cx.fillText(text, x, y);
    }

    dot(cx, x, y, r, alpha) {
      if (alpha < 0.01) return;
      cx.fillStyle = `rgba(${INK},${alpha})`; cx.beginPath(); cx.arc(x, y, r, 0, 6.2832); cx.fill();
    }

    /* ------------------------------------------------------------------ Listen: the landscape */
    camA(tip) { return { yaw: lerp(-0.34, 0, tip), pitch: lerp(0.44, Math.PI / 2, tip), dist: lerp(3.05, 2.7, tip), f: 2.55, t: [0, 0, 0] }; }

    ridge(tip, rise) {                                   // the whistle on the landscape, in world space
      const out = [];
      for (let i = 0; i < this.N; i++) {
        const u = i / (this.N - 1), x = -0.95 + 1.9 * u;
        const env = Math.pow(Math.sin(Math.PI * clamp(u * 0.98 + 0.01)), 0.35);
        out.push([x, 0.55 * env * rise * (1 - tip), 0.72 * FAMILIES[MAIN.f](u, MAIN.v)]);
      }
      return out;
    }

    drawLandscape(cx, A, rise, tip) {
      const c = this.camA(tip), hs = rise * (1 - tip), { R, C } = this;
      for (let i = R - 1; i >= 0; i--) {                  // far rows first: each row hides what is behind it
        const z = -1 + 2 * i / (R - 1), top = [], base = [];
        for (let j = 0; j < C; j++) {
          const x = -1.3 + 2.6 * j / (C - 1), u = (x + 0.95) / 1.9;
          let h = 0.045 * this.noise[i][j];
          for (const xc of this.clicks) h += 0.07 * Math.exp(-(((x - xc) / 0.012) ** 2));
          if (u > 0 && u < 1) {
            const zc = 0.72 * FAMILIES[MAIN.f](u, MAIN.v), env = Math.pow(Math.sin(Math.PI * u), 0.35);
            h += 0.55 * env * Math.exp(-(((z - zc) / 0.075) ** 2));
          }
          top.push(this.proj(x, h * hs, z, c));
          base.push(this.proj(x, 0, z, c));
        }
        const near = 1 - i / (R - 1), a = A * rise * (1 - tip) * (0.2 + 0.45 * near);
        cx.beginPath(); cx.moveTo(top[0][0], top[0][1]);
        for (let j = 1; j < C; j++) cx.lineTo(top[j][0], top[j][1]);
        for (let j = C - 1; j >= 0; j--) cx.lineTo(base[j][0], base[j][1]);
        cx.closePath(); cx.fillStyle = `rgba(${WATER},${0.9 * A * rise * (1 - tip * 0.7)})`; cx.fill();
        if (a > 0.01) {
          cx.beginPath(); cx.moveTo(top[0][0], top[0][1]);
          for (let j = 1; j < C; j++) cx.lineTo(top[j][0], top[j][1]);
          cx.strokeStyle = `rgba(${PALE},${a})`; cx.lineWidth = 1; cx.stroke();
        }
      }
      const pts = this.ridge(tip, rise).map(p => this.proj(p[0], p[1], p[2], c));
      this.line(cx, pts, A * rise, 1.9);
      // plain-language labels: axes while it is a landscape, "its shape" once it is a line
      const o = A * rise * (1 - sm(0.3, 0.75, tip)), q = A * sm(0.65, 1, tip);
      const t1 = this.proj(0.95, 0, -1.16, c), p1 = this.proj(1.42, 0, 1.0, c), peak = pts[Math.round(this.N * 0.13)];
      this.label(cx, 'time →', t1[0], t1[1] + 18, o, 'right');
      this.label(cx, 'pitch', p1[0] + 10, p1[1], o);
      this.label(cx, 'loudness ↑', peak[0] - 14, peak[1] - 26, o * sm(0.5, 1, rise), 'right');
      this.label(cx, 'a whistle', pts[Math.round(this.N * 0.62)][0] + 18, pts[Math.round(this.N * 0.62)][1] - 30, o * sm(0.6, 1, rise));
      const tl = this.proj(1.02, 0, -1.08, c), pl = this.proj(-1.38, 0, 0.92, c);
      this.label(cx, 'time →', tl[0], tl[1] + 22, q, 'right');
      this.label(cx, 'pitch ↑', pl[0], pl[1], q);
      this.label(cx, 'its shape', pts[this.N - 1][0] + 16, pts[this.N - 1][1] + 5, q);
      return pts;
    }

    /* ------------------------------------------------------------------ Learn and Imagine: the map of shapes */
    camB(gather, dive) {
      const q = ISLANDS[0];
      return { yaw: lerp(-0.5, 0.42, gather) + 0.3 * dive, pitch: lerp(0.24, 0.16, dive), dist: lerp(3.5, 1.75, dive),
               f: 2.55, t: [q[0] * dive, q[1] * dive, q[2] * dive] };
    }

    contourAt(sx, sy, k, m, scale) {                    // a member drawn small, facing the viewer, around (sx, sy)
      const L = 0.36 * this.S * k * scale, Hh = 0.12 * this.S * k * scale, f = FAMILIES[m.f], out = [];
      for (let i = 0; i < this.N; i += 3) { const u = i / (this.N - 1); out.push([sx + (u - 0.5) * L, sy - f(u, m.v) * Hh]); }
      return out;
    }

    drawMap(cx, A, P, flatLine) {
      const crowd = ph(P.crowd, P.pos), gather = ph(P.gather, P.pos), dive = ph(P.dive, P.pos), unfold = ph(P.unfold, P.pos);
      const c = this.camB(gather, dive), g = sm(0, 1, gather);
      // islands: soft light where look-alikes gather
      ISLANDS.forEach((q, f) => {
        const [x, y, k] = this.proj(q[0], q[1], q[2], c), r = 0.36 * this.S * k;
        const a = A * 0.2 * sm(0.35, 1, gather) * (f === 0 ? 1 - 0.5 * unfold : 1 - 0.9 * dive);
        if (a < 0.005) return;
        const gr = cx.createRadialGradient(x, y, 0, x, y, r);
        gr.addColorStop(0, `rgba(${PALE},${a})`); gr.addColorStop(1, `rgba(${PALE},0)`);
        cx.fillStyle = gr; cx.beginPath(); cx.arc(x, y, r, 0, 6.2832); cx.fill();
      });
      // every whistle: appears (crowd), folds into a point and drifts to its island (gather)
      const items = this.members.map(m => {
        const t = sm(0, 1, g), p = [0, 1, 2].map(i => lerp(m.p0[i], m.p1[i], t));
        const s = this.proj(p[0], p[1], p[2], c);
        return { m, s };
      }).sort((a, b) => b.s[3] - a.s[3]);
      const fold = 1 - sm(0.05, 0.6, gather), pointA = sm(0.25, 0.7, gather);
      for (const { m, s } of items) {
        const appear = m.main ? 1 : sm(m.order * 0.7, m.order * 0.7 + 0.3, crowd);
        // diving in: the other islands fall away; once the new whistle unrolls, the map steps back behind it
        const fade = (m.main || m.f === 0 ? 1 : 1 - 0.92 * dive) * (1 - 0.8 * unfold);
        const depth = clamp(1.25 - s[3] * 0.4, 0.45, 1.2), a = A * appear * fade * depth;
        s[2] = Math.min(s[2], 1.6);                        // points passing the camera stay points, not blobs
        if (a < 0.01) continue;
        if (m.main) {
          // our whistle: from the flat line of Listen to its place among the others, then to its island
          const own = this.contourAt(s[0], s[1], s[2], m, 1.5 * fold + 0.0001);
          const into = sm(0, 1, crowd), pts = own.map((q, i) => {
            const fl = flatLine[Math.min(flatLine.length - 1, i * 3)];
            return [lerp(fl[0], q[0], into), lerp(fl[1], q[1], into)];
          });
          if (fold > 0.02) this.line(cx, pts, a, 1.8);
          this.dot(cx, s[0], s[1], 2 + 2.2 * s[2], a * pointA);
          if (pointA > 0.05) {
            cx.strokeStyle = `rgba(${INK},${0.5 * a * pointA * (1 - dive)})`; cx.lineWidth = 1;
            cx.beginPath(); cx.arc(s[0], s[1], 9 + 5 * s[2], 0, 6.2832); cx.stroke();
          }
        } else {
          if (fold > 0.02) this.line(cx, this.contourAt(s[0], s[1], s[2], m, fold), a * 0.75, 1.1, false);
          this.dot(cx, s[0], s[1], 1.4 + 1.6 * s[2], a * pointA * 0.85);
        }
      }
      const isl = this.proj(ISLANDS[0][0], ISLANDS[0][1], ISLANDS[0][2], c);
      this.label(cx, 'every whistle becomes a point', this.ox, this.box.y0 + 8, A * sm(0.05, 0.3, gather) * (1 - sm(0.6, 0.9, gather)), 'center');
      this.label(cx, 'look-alikes gather into islands', isl[0], isl[1] - 0.42 * this.S * isl[2], A * sm(0.65, 1, gather) * (1 - sm(0, 0.4, dive)), 'center');

      // Imagine: a new point in the island, unrolling into a new whistle
      if (dive > 0.3) {
        const q = [ISLANDS[0][0] + 0.12, ISLANDS[0][1] - 0.08, ISLANDS[0][2] + 0.04];
        const [x, y, k] = this.proj(q[0], q[1], q[2], c), born = sm(0.3, 0.7, dive);
        for (let i = 0; i < 3; i++) {                        // rings spreading out from the new point
          const rr = sm(0.4, 1, dive) * (18 + 26 * i) * k * (1 - unfold * 0.5);
          if (rr > 1) { cx.strokeStyle = `rgba(${PALE},${A * 0.5 * (1 - i / 3) * (1 - unfold)})`; cx.lineWidth = 1;
                        cx.beginPath(); cx.arc(x, y, rr, 0, 6.2832); cx.stroke(); }
        }
        this.dot(cx, x, y, 3.5 * born + 1, A * born * (1 - 0.7 * unfold));
        this.label(cx, 'a new spot…', x + 22, y - 18, A * sm(0.55, 0.9, dive) * (1 - sm(0.1, 0.4, unfold)));
        if (unfold > 0.001) {
          const b = this.box, X0 = b.x0 + 0.05 * (b.x1 - b.x0), X1 = b.x1 - 0.02 * (b.x1 - b.x0);
          const Y = (b.y0 + b.y1) / 2 + 0.04 * (b.y1 - b.y0), Hh = 0.27 * (b.y1 - b.y0), f = FAMILIES[NEW.f];
          const pts = [];
          for (let i = 0; i < this.N; i++) {
            const u = i / (this.N - 1), t = clamp((unfold - u * 0.35) / 0.65);
            const e = t * t * (3 - 2 * t);
            pts.push([lerp(x, lerp(X0, X1, u), e), lerp(y, Y - f(u, NEW.v) * Hh, e)]);
          }
          // a quiet frame: pitch up, time across, as on every whistle picture
          const fa = A * sm(0.55, 1, unfold) * 0.5;
          if (fa > 0.01) {
            cx.strokeStyle = `rgba(${PALE},${fa * 0.5})`; cx.lineWidth = 1; cx.setLineDash([2, 6]);
            cx.beginPath(); cx.moveTo(X0, Y + Hh * 1.15); cx.lineTo(X1, Y + Hh * 1.15); cx.moveTo(X0 - 14, Y - Hh * 1.1); cx.lineTo(X0 - 14, Y + Hh * 1.15); cx.stroke();
            cx.setLineDash([]);
            this.label(cx, 'time →', X1, Y + Hh * 1.15 + 22, fa * 1.6, 'right');
            this.label(cx, 'pitch ↑', X0 - 14, Y - Hh * 1.1 - 12, fa * 1.6);
          }
          this.line(cx, pts, A, 2.1);
          this.label(cx, '…a new whistle', pts[this.N - 1][0], pts[this.N - 1][1] - 26, A * sm(0.75, 1, unfold), 'right');
          if (P.playT != null && unfold > 0.5) {               // playhead while "Hear it" plays
            const i = Math.round(P.playT * (this.N - 1)), px = pts[i][0];
            cx.strokeStyle = `rgba(${INK},${0.35 * A})`; cx.lineWidth = 1;
            cx.beginPath(); cx.moveTo(px, Y - Hh * 1.1); cx.lineTo(px, Y + Hh * 1.15); cx.stroke();
            this.dot(cx, px, pts[i][1], 4.5, A);
          }
        }
      }
    }

    /** draw everything for scroll position ``pos``; A = overall visibility; playT = 0..1 while "Hear it" plays */
    draw(cx, pos, A, playT) {
      const V = window.VIZ;
      cx.lineJoin = 'round'; cx.lineCap = 'round';
      cx.font = `400 ${this.fs}px "IBM Plex Mono", monospace`;
      if ('letterSpacing' in cx) cx.letterSpacing = '0.06em';
      const rise = ph(V.rise, pos), tip = ph(V.tip, pos);
      if (pos < V.crowd[0]) this.drawLandscape(cx, A, rise, tip);
      else {
        const flat = this.ridge(1, 1).map(p => this.proj(p[0], p[1], p[2], this.camA(1)));
        this.drawMap(cx, A, { pos, playT, crowd: V.crowd, gather: V.gather, dive: V.dive, unfold: V.unfold }, flat);
      }
      cx.font = `400 ${this.fs - 2}px "IBM Plex Mono", monospace`;
      this.label(cx, 'illustration · not model output', this.box.x0, this.box.y1 + 34, A * 0.6);
    }
  }
  window.ResearchScene = ResearchScene;
})();
