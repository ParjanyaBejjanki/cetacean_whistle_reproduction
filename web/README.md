# Cetacean Acoustic Lab — local demo

A scroll-driven film about how dolphin whistles become data. It moves from
real underwater footage to the whistle contour, the latent space and a
generated call.

## Run it

Requires Python 3 (no other dependencies):

```
python serve.py
```

Then open http://localhost:5178 in Chrome or Edge on a desktop.

Use `serve.py` rather than `python -m http.server`. The built-in server
ignores HTTP Range requests, so the browser can't seek inside the video
files, and the return shot would start from the wrong frame.

## What's where

- `index.html`: the four movements (Ocean, Listen, Research, Return) and their copy
- `shots.js`: the edit: shot windows, depth curve, text and visualisation timings (all in scroll position)
- `app.js`: the controller: playback, cross-fades, the research plot, the "Hear it" tone
- `audio-engine.js`: whistle contour functions and the synthetic tone
- `media4k/`, `fonts/`: local assets (see `CREDITS.md`)

The site targets desktop displays and needs about 100 MB of video. 4K H.264
decodes in hardware on current desktop GPUs.
