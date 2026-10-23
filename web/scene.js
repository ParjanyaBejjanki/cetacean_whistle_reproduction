/**
 * Listen → Learn → Imagine as one 3D scene on the #sig canvas: canvas 2D with its own perspective projection, no
 * library. Like the rest of the film, every frame is a pure function of the scroll position (timings: shots.js VIZ),
 * and one camera moves continuously through all of it: nothing cuts. Hand-made to show the idea, not the trained
 * model's output; the corner says so.
 *
 *   rise, tip      A sound landscape (a wireframe mesh: time across, pitch into the distance, loudness up) with the
 *                  whistle as a glowing ridge. The camera tips over until it looks straight down: the ridge is now a
 *                  line, the whistle's shape (a spectrogram seen from above is exactly that picture).
 *   crowd          The sheet tilts away while the line rises off it, always facing the viewer, and shrinks into one
 *                  glowing point at the centre of the model's "mind": a faint sphere of drifting particles. Other
 *                  whistles stream in from outside it.
 *   gather         Each whistle folds into a point; look-alikes drift together into families, wired to each other
 *                  like neurons, with small pulses travelling the wires as you scroll. The camera orbits.
 *   dive, unfold   The camera glides into the first family; a new point is born between three whistles it knows,
 *                  then unrolls into a new whistle of that family: the one "Hear it" plays (newWhistleHz).
 */
(() => {
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const sm = (a, b, v) => { const t = clamp((v - a) / (b - a)); return t * t * (3 - 2 * t); };
  const smoother = t => t * t * t * (t * (6 * t - 15) + 10);
  const lerp = (a, b, t) => a + (b - a) * t;
  const lerp3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
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
  // each family keeps a hue of its own inside the mind, all within the water's palette
  const HUES = ['234,246,251', '143,212,232', '172,190,255', '124,222,200', '204,184,244'];
  const ISLANDS = [[-0.62, 0.3, -0.1], [0.6, 0.42, 0.3], [-0.14, -0.46, 0.4], [0.66, -0.32, -0.36], [0.02, 0.1, -0.7]];
  const MAIN = { f: 0, v: { a: 1, s: 0 } };              // the whistle we follow
  const NEW = { f: 0, v: { a: 0.82, s: 0.95 } };          // the new whistle of the same family
  const RM = 1.15;                                       // radius of the mind
  const INK = '234,246,251', PALE = '169,207,224';

  /* Round things (points, pulses, glows) are pre-rendered once and copied as images: on an integrated GPU, dozens
     of small vector circles and per-frame gradients cost several times more than the same pixels as image blits. */
  function sprite(rgb, kind) {
    const n = kind === 'dot' ? 64 : 128, c = document.createElement('canvas'), g = c.getContext('2d'), r = n / 2;
    c.width = c.height = n;
    const gr = g.createRadialGradient(r, r, 0, r, r, r);
    if (kind === 'dot') {                                 // a bright core (radius 1/4) inside a soft halo
      gr.addColorStop(0, `rgba(${rgb},1)`); gr.addColorStop(0.2, `rgba(${rgb},0.95)`); gr.addColorStop(0.28, `rgba(${rgb},0.32)`);
      gr.addColorStop(0.6, `rgba(${rgb},0.08)`); gr.addColorStop(1, `rgba(${rgb},0)`);
    } else { gr.addColorStop(0, `rgba(${rgb},1)`); gr.addColorStop(1, `rgba(${rgb},0)`); }
    g.fillStyle = gr; g.fillRect(0, 0, n, n);
    return c;
  }
  const SPRITES = {};
  const spr = (rgb, kind) => SPRITES[kind + rgb] || (SPRITES[kind + rgb] = sprite(rgb, kind));
  function blit(cx, img, x, y, R, a) {
    if (a < 0.01 || R < 0.3) return;
    cx.globalAlpha = Math.min(1, a); cx.drawImage(img, x - R, y - R, 2 * R, 2 * R); cx.globalAlpha = 1;
  }

  class ResearchScene {
    constructor() {
      const r = rng(20261024);
      this.R = 30; this.C = 100;                          // mesh rows (pitch) x columns (time)
      this.noise = [];
      for (let i = 0; i < this.R; i++) {                  // smoothed random floor, one row per pitch band
        const raw = Array.from({ length: this.C + 6 }, () => r());
        this.noise.push(Array.from({ length: this.C }, (_, c) => (raw[c] + raw[c + 1] * 2 + raw[c + 2] * 3 + raw[c + 3] * 2 + raw[c + 4]) / 9));
      }
      this.clicks = [0.18, 0.47, 0.83].map(u => -1.3 + 2.6 * u);   // a few broadband clicks, as in real recordings
      this.H = this.heights();
      // every whistle: from outside the mind (out) to a first place inside (p0), then to its family (p1)
      const gauss = () => (r() + r() + r() - 1.5) * 0.9, dir = () => { const a = r() * 6.2832, e = Math.acos(2 * r() - 1); return [Math.sin(e) * Math.cos(a), Math.cos(e), Math.sin(e) * Math.sin(a)]; };
      this.members = [];
      ISLANDS.forEach((q, f) => {
        for (let j = 0; j < 9; j++) {
          const d0 = dir(), d1 = dir(), rad = 0.35 + 0.55 * r();
          this.members.push({ f, v: { a: 0.75 + 0.45 * r(), s: (r() - 0.5) * 1.4 }, order: r(),
                              out: d0.map(c => c * (2.3 + 0.6 * r())), p0: d1.map(c => c * rad),
                              p1: [q[0] + gauss() * 0.12, q[1] + gauss() * 0.1, q[2] + gauss() * 0.12] });
        }
      });
      this.main = { main: true, f: 0, v: MAIN.v, p0: [0, 0.04, 0], p1: [ISLANDS[0][0] + 0.04, ISLANDS[0][1] + 0.03, ISLANDS[0][2]] };
      // wires: each whistle to its two nearest look-alikes, as they will sit once gathered
      this.edges = [];
      const all = [...this.members, this.main];
      all.forEach((m, i) => {
        const near = all.map((o, j) => [j, o.f === m.f && j !== i ? Math.hypot(o.p1[0] - m.p1[0], o.p1[1] - m.p1[1], o.p1[2] - m.p1[2]) : 9])
          .sort((a, b) => a[1] - b[1]).slice(0, 2);
        for (const [j] of near) if (!this.edges.some(e => (e.a === j && e.b === i))) this.edges.push({ a: i, b: j, f: m.f, seed: r() });
      });
      this.all = all;
      // the new whistle is born between the three whistles of family 0 closest to its middle
      const fam0 = all.map((m, i) => [i, m]).filter(([, m]) => m.f === 0)
        .sort((a, b) => Math.hypot(...a[1].p1.map((c, k) => c - ISLANDS[0][k])) - Math.hypot(...b[1].p1.map((c, k) => c - ISLANDS[0][k]))).slice(0, 3);
      this.parents = fam0.map(([i]) => i);
      this.Q = [0, 1, 2].map(k => fam0.reduce((s, [, m]) => s + m.p1[k], 0) / 3 + [0.05, -0.04, 0.03][k]);
      this.dust = Array.from({ length: 170 }, () => { const d = dir(), rr = Math.cbrt(r()) * RM * 1.02; return [d[0] * rr, d[1] * rr, d[2] * rr, r()]; });
      this.N = 64;                                        // points per drawn contour
    }

    heights() {                                           // the mesh's loudness at every node (time x pitch)
      const out = [];
      for (let i = 0; i < this.R; i++) {
        const z = -1 + 2 * i / (this.R - 1), row = [];
        for (let j = 0; j < this.C; j++) {
          const x = -1.3 + 2.6 * j / (this.C - 1), u = (x + 0.95) / 1.9;
          let h = 0.045 * this.noise[i][j];
          for (const xc of this.clicks) h += 0.07 * Math.exp(-(((x - xc) / 0.012) ** 2));
          if (u > 0 && u < 1) h += 0.55 * Math.pow(Math.sin(Math.PI * u), 0.35) * Math.exp(-(((z - 0.72 * FAMILIES[MAIN.f](u, MAIN.v)) / 0.075) ** 2));
          row.push(h);
        }
        out.push(row);
      }
      return out;
    }

    resize(W, H, box) {
      this.W = W; this.Hpx = H; this.box = box;
      this.ox = (box.x0 + box.x1) / 2; this.oy = (box.y0 + box.y1) / 2;
      this.S = Math.min((box.x1 - box.x0) / 2.75, (box.y1 - box.y0) / 1.45);
      this.fs = Math.round(clamp(W * 0.0068, 12, 16));
    }

    /** the new whistle as frequencies (Hz) for the audio engine: n points over its duration */
    newWhistleHz(n) { return Array.from({ length: n }, (_, i) => 9200 + 2600 * FAMILIES[NEW.f](i / (n - 1), NEW.v)); }

    /* ------------------------------------------------------------------ one camera for the whole scene */
    camA(tip) { return { yaw: lerp(-0.34, 0, tip), pitch: lerp(0.44, Math.PI / 2, tip), dist: lerp(3.05, 2.7, tip), t: [0, 0, 0] }; }
    camMind(g, d) {
      return { yaw: lerp(-0.42, 0.36, g) + 0.34 * d, pitch: lerp(0.3, 0.2, g) - 0.05 * d,
               dist: lerp(lerp(3.55, 3.3, g), 1.55, d), t: lerp3([0, 0, 0], this.Q, d) };
    }
    camera(P) {
      if (P.crowd <= 0) return this.camA(P.tip);
      const a = this.camA(1), b = this.camMind(P.gather, P.dive), e = smoother(P.crowd);
      return { yaw: lerp(a.yaw, b.yaw, e), pitch: lerp(a.pitch, b.pitch, e), dist: lerp(a.dist, b.dist, e), t: lerp3(a.t, b.t, e) };
    }

    /* yaw about the vertical axis, then pitch (tilt down), around the point the camera looks at */
    proj(x, y, z, c) {
      x -= c.t[0]; y -= c.t[1]; z -= c.t[2];
      const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw), cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const y2 = y * cp + z1 * sp, z2 = -y * sp + z1 * cp, d = Math.max(0.12, c.dist + z2);
      const k = 2.55 / d;
      return [this.ox + x1 * k * this.S, this.oy - y2 * k * this.S, k, z2];
    }

    line(cx, pts, alpha, width = 1.7, glow = true, rgb = INK) {
      if (alpha < 0.01 || pts.length < 2) return;
      const path = () => { cx.beginPath(); cx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) cx.lineTo(pts[i][0], pts[i][1]); };
      if (glow) { path(); cx.strokeStyle = `rgba(${PALE},${0.16 * alpha})`; cx.lineWidth = width * 4.5; cx.stroke(); }
      path(); cx.strokeStyle = `rgba(${rgb},${0.96 * alpha})`; cx.lineWidth = width; cx.stroke();
    }

    label(cx, text, x, y, alpha, align = 'left') {
      if (alpha < 0.02) return;
      cx.textAlign = align; cx.fillStyle = `rgba(214,236,244,${0.9 * alpha})`; cx.fillText(text, x, y);
    }

    glowDot(cx, x, y, r, alpha, rgb = INK) { blit(cx, spr(rgb, 'dot'), x, y, r * 4, alpha); }

    /* the followed whistle as a line facing the viewer: centre c (world), scale s (1 = as on the landscape) */
    billboard(c, shape, s, cam) {
      const [sx, sy, k] = this.proj(c[0], c[1], c[2], cam), out = [];
      for (let i = 0; i < this.N; i++) {
        const u = i / (this.N - 1);
        out.push([sx + (-0.95 + 1.9 * u) * k * this.S * s, sy - 0.72 * FAMILIES[shape.f](u, shape.v) * k * this.S * s]);
      }
      return out;
    }

    /* ------------------------------------------------------------------ Listen: the mesh */
    drawMesh(cx, cam, A, rise, tip, away) {
      const hs = rise * (1 - tip), { R, C, H } = this, keep = lerp(1, 0.34, tip) * (1 - away);
      if (A * keep < 0.004) return;
      let prevTop = null;
      for (let i = R - 1; i >= 0; i--) {                  // far rows first: each row hides what is behind it
        const z = -1 + 2 * i / (R - 1), top = [], base = [];
        for (let j = 0; j < C; j++) {
          const x = -1.3 + 2.6 * j / (C - 1);
          top.push(this.proj(x, H[i][j] * hs, z, cam));
          if (hs > 0.001) base.push(this.proj(x, 0, z, cam));
        }
        const near = 1 - i / (R - 1);
        if (hs > 0.001) {                                 // haze: far rows lighter, near rows the deep water
          cx.beginPath(); cx.moveTo(top[0][0], top[0][1]);
          for (let j = 1; j < C; j++) cx.lineTo(top[j][0], top[j][1]);
          for (let j = C - 1; j >= 0; j--) cx.lineTo(base[j][0], base[j][1]);
          cx.closePath();
          cx.fillStyle = `rgba(${Math.round(lerp(12, 3, near))},${Math.round(lerp(50, 20, near))},${Math.round(lerp(84, 38, near))},${0.9 * A * rise * keep})`;
          cx.fill();
        }
        const a = A * rise * keep * (0.16 + 0.46 * near);
        cx.beginPath(); cx.moveTo(top[0][0], top[0][1]);
        for (let j = 1; j < C; j++) cx.lineTo(top[j][0], top[j][1]);
        // the mesh's other direction: lines along pitch, from this row to the one behind it
        if (prevTop) for (let j = 4; j < C; j += 8) { cx.moveTo(prevTop[j][0], prevTop[j][1]); cx.lineTo(top[j][0], top[j][1]); }
        cx.strokeStyle = `rgba(${PALE},${a})`; cx.lineWidth = 1; cx.stroke();
        prevTop = top;
      }
    }

    ridge(cam, rise, tip) {                               // the whistle on the mesh, projected
      const out = [];
      for (let i = 0; i < this.N; i++) {
        const u = i / (this.N - 1), x = -0.95 + 1.9 * u, env = Math.pow(Math.sin(Math.PI * clamp(u * 0.98 + 0.01)), 0.35);
        out.push(this.proj(x, 0.55 * env * rise * (1 - tip), 0.72 * FAMILIES[MAIN.f](u, MAIN.v), cam));
      }
      return out;
    }

    /* ------------------------------------------------------------------ Learn: the mind */
    drawMind(cx, cam, A, P) {
      const { crowd, gather: g, dive: d, unfold: u } = P, mind = A * sm(0.15, 0.75, crowd) * (1 - u);
      if (mind < 0.004) return;
      // diving in, the camera passes through the sphere: its light and its cage dissolve around the viewer (and
      // stop costing full-screen fills once it is gone)
      const cage = mind * (1 - sm(0.1, 0.9, d));
      // the sphere: a faint wireframe, its far side fainter than its near side
      const center = this.proj(0, 0, 0, cam)[3], circles = [];
      if (cage > 0.004) {
        for (const yy of [-0.62, 0, 0.62]) circles.push(t => { const rr = Math.sqrt(RM * RM - (yy * RM) ** 2); return [rr * Math.cos(t), yy * RM, rr * Math.sin(t)]; });
        for (let m = 0; m < 4; m++) { const a = m * Math.PI / 4; circles.push(t => [RM * Math.sin(t) * Math.cos(a), RM * Math.cos(t), RM * Math.sin(t) * Math.sin(a)]); }
        const front = new Path2D(), back = new Path2D();
        for (const f of circles) {
          // one continuous run per stretch on the same side (near / far): few long lines, not many short ones
          let prev = null, side = null;
          for (let s = 0; s <= 44; s++) {
            const q = f(s / 44 * 6.2832), p = this.proj(q[0], q[1], q[2], cam);
            if (prev) {
              const path = (p[3] + prev[3]) / 2 < center ? front : back;
              if (path !== side) { path.moveTo(prev[0], prev[1]); side = path; }
              path.lineTo(p[0], p[1]);
            }
            prev = p;
          }
        }
        // a faint inner light and a rim, so the sphere reads as a living volume rather than a cage
        const [mx, my, mk] = this.proj(0, 0, 0, cam), mr = RM * mk * this.S;
        blit(cx, spr(PALE, 'glow'), mx, my, mr * 0.75, 0.11 * cage);
        cx.lineWidth = 1;
        cx.strokeStyle = `rgba(${PALE},${0.05 * cage})`; cx.stroke(back);
        cx.strokeStyle = `rgba(${PALE},${0.13 * cage})`; cx.stroke(front);
        cx.strokeStyle = `rgba(${PALE},${0.16 * cage})`; cx.beginPath(); cx.arc(mx, my, mr, 0, 6.2832); cx.stroke();
      }
      // drifting particles: depth in the dark
      const dustA = mind * (1 - 0.5 * d) * sm(0.5, 1, crowd);   // drifts in once the sheet has gone
      if (dustA > 0.01) for (const [b, lo] of [[0.12, 0], [0.24, 0.34], [0.4, 0.67]]) {
        cx.fillStyle = `rgba(${PALE},${b * dustA})`;
        for (const q of this.dust) {                       // plain rectangles: the GPU's cheapest fill
          if (q[3] < lo || q[3] >= lo + 0.34) continue;
          const p = this.proj(q[0], q[1], q[2], cam), rr = clamp(0.5 + 0.6 * p[2], 0.5, 2.2);
          cx.fillRect(p[0] - rr / 2, p[1] - rr / 2, rr, rr);
        }
      }
      // families: a soft glow where look-alikes gather
      ISLANDS.forEach((q, f) => {
        const [x, y, k] = this.proj(q[0], q[1], q[2], cam), r = 0.28 * this.S * Math.min(k, 1.4);
        const a = A * 0.2 * sm(0.4, 1, g) * (f === 0 ? 1 - 0.85 * u : 1 - 0.85 * d);
        blit(cx, spr(HUES[f], 'glow'), x, y, r, a);
      });
      // every whistle: streams in (crowd), folds into a point and drifts to its family (gather)
      const eg = smoother(g), fold = 1 - sm(0.05, 0.55, g), pointA = sm(0.2, 0.65, g);
      const pos = this.all.map(m => {
        if (m.main) return null;
        const t = sm(m.order * 0.55, m.order * 0.55 + 0.45, crowd);
        return lerp3(lerp3(m.out, m.p0, smoother(t)), m.p1, eg);
      });
      pos[this.all.length - 1] = this.mainPos(P);
      const scr = pos.map(p => this.proj(p[0], p[1], p[2], cam));
      const famA = f => (f === 0 ? 1 : 1 - 0.9 * d) * (1 - 0.9 * u);
      // wires between look-alikes, with pulses moving along them as you scroll
      const wireA = A * sm(0.3, 0.9, g);
      if (wireA > 0.01) {
        for (let f = 0; f < 5; f++) {
          const w = new Path2D(), fa = wireA * famA(f), pa = 0.85 * fa * sm(0.5, 1, g), img = spr(HUES[f], 'dot');
          if (fa < 0.01) continue;
          for (const e of this.edges) {
            if (e.f !== f) continue;
            const a = scr[e.a], b = scr[e.b];
            w.moveTo(a[0], a[1]); w.lineTo(b[0], b[1]);
          }
          cx.lineWidth = 1; cx.strokeStyle = `rgba(${HUES[f]},${0.3 * fa * (f === 0 ? 1 + d : 1)})`; cx.stroke(w);
          if (pa > 0.01) for (const e of this.edges) {
            if (e.f !== f) continue;
            const a = scr[e.a], b = scr[e.b], t = (P.pos * 24 + e.seed) % 1;
            blit(cx, img, lerp(a[0], b[0], t), lerp(a[1], b[1], t), 6.5, pa);
          }
        }
      }
      // the whistles themselves, far ones first
      const order = this.all.map((m, i) => i).filter(i => !this.all[i].main).sort((a, b) => scr[b][3] - scr[a][3]);
      for (const i of order) {
        const m = this.all[i], s = scr[i], r = Math.hypot(...pos[i]);
        if (s[2] > 2.4) continue;                          // too close to the camera to read
        // whistles still outside the mind are only hinted at; they light up as they enter it
        const appear = sm(m.order * 0.55, m.order * 0.55 + 0.25, crowd) * (1 - 0.85 * sm(RM * 1.02, RM * 1.8, r));
        const depth = clamp(1.2 - s[3] * 0.38, 0.4, 1.15), a = A * appear * depth * famA(m.f);
        if (a < 0.01) continue;
        const k = Math.min(s[2], 1.7);
        if (fold > 0.02) {
          const L = 0.34 * this.S * k * fold, Hh = 0.11 * this.S * k * fold, pts = [];
          for (let j = 0; j < this.N; j += 3) { const uu = j / (this.N - 1); pts.push([s[0] + (uu - 0.5) * L, s[1] - FAMILIES[m.f](uu, m.v) * Hh]); }
          this.line(cx, pts, a * 0.8, 1.1, false, HUES[m.f]);
        }
        if (pointA > 0.01) blit(cx, spr(HUES[m.f], 'dot'), s[0], s[1], 4 * (1.3 + 1.5 * k), a * pointA);
      }
      // plain-language labels
      const top = this.proj(0, RM * 1.12, 0, cam), isl = this.proj(ISLANDS[0][0], ISLANDS[0][1], ISLANDS[0][2], cam);
      this.label(cx, 'inside the model', top[0], top[1], A * sm(0.45, 0.9, crowd) * (1 - sm(0.1, 0.35, g)), 'center');
      this.label(cx, 'every whistle becomes a point', top[0], top[1], A * sm(0.3, 0.45, g) * (1 - sm(0.6, 0.8, g)), 'center');
      this.label(cx, 'look-alikes wire together into families', isl[0], isl[1] - 0.4 * this.S * isl[2], A * sm(0.75, 1, g) * (1 - sm(0, 0.35, d)), 'center');
      return scr;
    }

    mainPos(P) { return lerp3(lerp3([0, 0, 0], this.main.p0, sm(0.3, 0.95, P.crowd)), this.main.p1, smoother(P.gather)); }

    /* the followed whistle across Listen and Learn: ridge, then a line facing the viewer, then one bright point */
    drawMain(cx, cam, A, P) {
      if (P.crowd <= 0) {
        const pts = this.ridge(cam, P.rise, P.tip);
        this.line(cx, pts, A * P.rise, 1.9);
        return pts;
      }
      const c = this.mainPos(P), s = 1 - sm(0.35, 0.95, P.crowd), [x, y, k] = this.proj(c[0], c[1], c[2], cam);
      if (s > 0.01) this.line(cx, this.billboard(c, MAIN, s, cam), A, lerp(1.1, 1.9, s));
      const pa = A * sm(0.65, 1, P.crowd) * (1 - 0.6 * P.unfold);
      this.glowDot(cx, x, y, 2.4 + 1.6 * Math.min(k, 1.6), pa);
      return null;
    }

    /* ------------------------------------------------------------------ Imagine: a new point, a new whistle */
    drawNew(cx, cam, A, P, scr) {
      const { dive: d, unfold: u } = P;
      if (d <= 0.25) return;
      const q = this.Q, [x, y, k] = this.proj(q[0], q[1], q[2], cam), born = sm(0.3, 0.7, d);
      // wired to the three whistles it lies between
      if (scr) {
        const w = new Path2D();
        for (const i of this.parents) { w.moveTo(x, y); w.lineTo(scr[i][0], scr[i][1]); }
        cx.lineWidth = 1.2; cx.strokeStyle = `rgba(${INK},${0.55 * A * born * (1 - u)})`; cx.stroke(w);
      }
      for (let i = 0; i < 3; i++) {                         // rings spreading out from the new point
        const rr = sm(0.4, 1, d) * (18 + 26 * i) * k * (1 - u * 0.5);
        if (rr > 1) { cx.strokeStyle = `rgba(${PALE},${A * 0.45 * (1 - i / 3) * (1 - u)})`; cx.lineWidth = 1;
                      cx.beginPath(); cx.arc(x, y, rr, 0, 6.2832); cx.stroke(); }
      }
      this.glowDot(cx, x, y, 3.2 * born + 1, A * born * (1 - 0.9 * u));
      this.label(cx, 'a new spot, between whistles it knows…', x + 24, y - 20, A * sm(0.55, 0.9, d) * (1 - sm(0.1, 0.4, u)));
      if (u <= 0.001) return;
      const b = this.box, X0 = b.x0 + 0.05 * (b.x1 - b.x0), X1 = b.x1 - 0.02 * (b.x1 - b.x0);
      const Y = (b.y0 + b.y1) / 2 + 0.04 * (b.y1 - b.y0), Hh = 0.27 * (b.y1 - b.y0), f = FAMILIES[NEW.f], pts = [];
      for (let i = 0; i < this.N; i++) {                    // unrolls from the point, left to right
        const uu = i / (this.N - 1), e = smoother(clamp((u - uu * 0.35) / 0.65));
        pts.push([lerp(x, lerp(X0, X1, uu), e), lerp(y, Y - f(uu, NEW.v) * Hh, e)]);
      }
      const fa = A * sm(0.55, 1, u) * 0.5;                   // a quiet frame: pitch up, time across
      if (fa > 0.01) {
        cx.strokeStyle = `rgba(${PALE},${fa * 0.5})`; cx.lineWidth = 1; cx.setLineDash([2, 6]);
        cx.beginPath(); cx.moveTo(X0, Y + Hh * 1.15); cx.lineTo(X1, Y + Hh * 1.15); cx.moveTo(X0 - 14, Y - Hh * 1.1); cx.lineTo(X0 - 14, Y + Hh * 1.15); cx.stroke();
        cx.setLineDash([]);
        this.label(cx, 'time →', X1, Y + Hh * 1.15 + 22, fa * 1.6, 'right');
        this.label(cx, 'pitch ↑', X0 - 14, Y - Hh * 1.1 - 12, fa * 1.6);
      }
      this.line(cx, pts, A, 2.1);
      this.label(cx, '…unfolds into a new whistle', X1, Y - Hh * 1.1 - 12, A * sm(0.75, 1, u), 'right');
      if (P.playT != null && u > 0.5) {                      // playhead while "Hear it" plays
        const i = Math.round(P.playT * (this.N - 1)), px = pts[i][0];
        cx.strokeStyle = `rgba(${INK},${0.35 * A})`; cx.lineWidth = 1;
        cx.beginPath(); cx.moveTo(px, Y - Hh * 1.1); cx.lineTo(px, Y + Hh * 1.15); cx.stroke();
        this.glowDot(cx, px, pts[i][1], 4, A);
      }
    }

    /** draw everything for scroll position ``pos``; A = overall visibility; playT = 0..1 while "Hear it" plays */
    draw(cx, pos, A, playT) {
      const V = window.VIZ;
      const P = { pos, playT, rise: ph(V.rise, pos), tip: ph(V.tip, pos), crowd: ph(V.crowd, pos), gather: ph(V.gather, pos),
                  dive: ph(V.dive, pos), unfold: ph(V.unfold, pos) };
      const cam = this.camera(P);
      cx.lineJoin = 'round'; cx.lineCap = 'round';
      cx.font = `400 ${this.fs}px "IBM Plex Mono", monospace`;
      if ('letterSpacing' in cx) cx.letterSpacing = '0.06em';
      this.drawMesh(cx, cam, A, P.rise, P.tip, sm(0, 0.4, P.crowd));
      const scr = P.crowd > 0 ? this.drawMind(cx, cam, A, P) : null;
      const ridge = this.drawMain(cx, cam, A, P);
      this.drawNew(cx, cam, A, P, scr);
      if (ridge) {                                           // Listen's labels: axes on the landscape, "its shape" from above
        const o = A * P.rise * (1 - sm(0.3, 0.75, P.tip)), q = A * sm(0.65, 1, P.tip) * (1 - sm(0, 0.2, P.crowd));
        const t1 = this.proj(0.95, 0, -1.16, cam), p1 = this.proj(1.42, 0, 1.0, cam), peak = ridge[Math.round(this.N * 0.13)];
        this.label(cx, 'time →', t1[0], t1[1] + 18, o, 'right');
        this.label(cx, 'pitch', p1[0] + 10, p1[1], o);
        this.label(cx, 'loudness ↑', peak[0] - 14, peak[1] - 26, o * sm(0.5, 1, P.rise), 'right');
        this.label(cx, 'a whistle', ridge[Math.round(this.N * 0.62)][0] + 18, ridge[Math.round(this.N * 0.62)][1] - 30, o * sm(0.6, 1, P.rise));
        const tl = this.proj(1.02, 0, -1.08, cam), pl = this.proj(-1.38, 0, 0.92, cam);
        this.label(cx, 'time →', tl[0], tl[1] + 22, q, 'right');
        this.label(cx, 'pitch ↑', pl[0], pl[1], q);
        this.label(cx, 'its shape', ridge[this.N - 1][0] + 16, ridge[this.N - 1][1] + 5, q);
      }
      cx.font = `400 ${this.fs - 2}px "IBM Plex Mono", monospace`;
      this.label(cx, 'illustration · not model output', this.box.x0, this.box.y1 + 34, A * 0.6);
    }
  }
  window.ResearchScene = ResearchScene;
})();
