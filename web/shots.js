/**
 * THE FILM — two clips, four movements, one body of water.
 *
 * Positions are `pos`: section index + progress through it
 * (0 Ocean · 1 Listen · 2 Research · 3 Return).
 *
 * Edit (re-cut, Oct 2026). The earlier cut used five 1080p/phone clips that
 * looked soft and mismatched on a desktop display. This cut keeps only two
 * 4K takes that share the same deep, sunlit blue:
 *
 *   whale     Pixabay 244188 (uploader LisaOlivia; Pixabay Content License).
 *             A humpback hangs head-down beneath the surface and drifts toward
 *             the lens. 4K/60 source, played at half speed (every frame is
 *             real, no interpolation) and rendered as a palindrome so it loops
 *             without a seam.
 *   dolphins  Pexels 5743314 (Daniel Torobekov; Pexels License). A pair rises
 *             from the reef, one passes close, both leave into open blue.
 *             Plays once and rests on its final frame.
 *
 * Transitions — each one designed for its pair, never the same twice:
 *   whale → dolphins     through the water: the whale recedes into the blue
 *                        (darkens toward the water's own navy, not black); the
 *                        dolphins arrive out of that same blue as two specks.
 *                        The animals never overlap.
 *   dolphins → research  an ellipsis within one shot: the moving take
 *                        dissolves to its own final frame (empty water, two
 *                        dolphins far away), which then deepens beneath the
 *                        research. No blur, no stand-in imagery.
 *   research → return    the water darkens fully, and the whale from the
 *                        opening resolves out of it: the same water, now heard
 *                        differently.
 */
window.SHOTS = [
  {
    id: 'whale', file: 'media4k/whale.mp4', poster: 'media4k/whale.jpg',
    windows: [[-1, -1, 0.6, 0.85]], loop: true,
    // the whale sits right of centre; the title owns the open water left
    pan: [50, 40], shift: [6.5, 0], push: [1.15, 1.2]
  },
  {
    id: 'dolphins', file: 'media4k/dolphins.mp4', poster: 'media4k/dolphins.jpg',
    windows: [[0.75, 0.98, 1.5, 1.62]], once: true, restart: true,
    pan: [50, 46], push: [1.0, 1.03]
  },
  {
    // The dolphins' own final frame, held beneath the research.
    id: 'still', image: 'media4k/dolphins-end.jpg',
    windows: [[1.5, 1.62, 2.74, 2.92]],
    pan: [50, 46], push: [1.03, 1.08]
  },
  {
    // The opening whale again: same file, entered closer to the lens.
    id: 'return', file: 'media4k/whale.mp4', poster: 'media4k/whale.jpg',
    windows: [[2.8, 3.1, 99, 99]], loop: true, start: 12,
    pan: [50, 40], shift: [6.5, 0], push: [1.2, 1.15]
  }
];

/** Depth: the water's own deep blue laid over the frame (0 = none). */
window.SHADE = [
  [0, 0.04], [0.5, 0.06], [0.7, 0.5], [0.8, 0.74], [0.9, 0.55], [1.04, 0.06],
  [1.5, 0.08], [1.75, 0.22], [2.0, 0.5], [2.4, 0.56], [2.72, 0.62], [2.84, 0.8],
  [2.95, 0.6], [3.15, 0.08], [9, 0.08]
];

/** Text blocks: [in0, in1, out0, out1] in pos. "dive" fills the crossing from the whale to the dolphins, where the
    water is at its darkest. */
window.TEXT = {
  hero: [-1, -1, 0.4, 0.54],
  dive: [0.62, 0.72, 1.02, 1.12],
  listen: [1.6, 1.7, 1.9, 1.98],
  signal: [1.98, 2.04, 2.24, 2.29],
  model: [2.29, 2.34, 2.52, 2.57],
  generate: [2.57, 2.62, 2.76, 2.81],
  coda: [3.2, 3.5, 99, 99]
};

/** The 3D scene (scene.js), keyed to the same beats. Each phase runs 0 -> 1 over its range. */
window.VIZ = {
  on: [1.92, 2.0, 2.79, 2.87],  // scene visible
  rise: [1.98, 2.07],           // Listen: the sound landscape rises, the whistle a ridge across it
  tip: [2.11, 2.21],            //   the camera tips over until it looks straight down: the ridge becomes a line
  crowd: [2.23, 2.37],          // Learn: the sheet tilts away, the line rises and shrinks into the mind; others stream in
  gather: [2.37, 2.5],          //   each folds into a point; look-alikes wire together into families
  dive: [2.54, 2.64],           // Imagine: the camera glides into one family; a new point is born between three whistles
  unfold: [2.64, 2.73]          //   it unfolds into a new whistle (the one "Hear it" plays)
};

/** Where the index links land (pos). */
window.STOPS = { ocean: 0, signal: 2.1, model: 2.47, generate: 2.75 };
