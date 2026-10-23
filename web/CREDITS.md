# Credits & licensing

Nothing here is AI-generated, composited or synthetic footage. Neither licence
below requires visible attribution, so the site carries none on screen; the
sources are recorded here and in an HTML comment in `index.html`.

## Footage (`media4k/`)

| File | Source | Creator | Licence | Notes |
|---|---|---|---|---|
| `whale.mp4` | [Pixabay 244188](https://pixabay.com/videos/whale-animal-ocean-wildlife-sea-244188/) | uploader *LisaOlivia* | [Pixabay Content License](https://pixabay.com/service/license-summary/) | Humpback beneath the surface. 4K/60 source → half speed (every frame real, no interpolation), rendered forward + reverse so it loops without a seam. Pixabay labels AI content; this clip carries no such label, and frame-by-frame inspection shows a rigid, consistent animal and natural sensor noise. Uploader identity and original authorship can't be independently verified. |
| `dolphins.mp4` | [Pexels 5743314](https://www.pexels.com/video/5743314/) | Daniel Torobekov | [Pexels License](https://www.pexels.com/license/) | Dolphins over a reef. 4K, full length (9 s). |
| `dolphins-end.jpg` | the final frame of `dolphins.mp4` | — | as above | Held beneath the research section. |

Both licences permit free use and modification. Neither permits selling
unaltered copies, redistributing the clips as stock, or implying endorsement.

Encoding: H.264 High, 8-bit 4:2:0, 3840×2160, 30 fps, CRF 18, 1-second GOP,
`+faststart`, no audio. Masters are in `_source/`. The previous cut's 1080p
clips are kept, unused, in `_source/previous-cut/`.

## Type (`fonts/`)

Cormorant Garamond (Christian Thalmann), IBM Plex Sans and IBM Plex Mono (IBM),
all under the SIL Open Font License 1.1. Latin subsets, variable weights.

## Sound

One synthetic whistle, generated in the browser by `audio-engine.js` when the
visitor asks for it. It is never presented as a field recording or attributed
to any animal shown.
