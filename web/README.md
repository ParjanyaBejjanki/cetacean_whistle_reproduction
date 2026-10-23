# Webpage — Learning the shape of a whistle

A scroll-driven film about how false killer whale whistles become data. It
moves from underwater footage to the whistle contour, the latent space and a
generated call. Desktop only, by design.

Status: the research section (Signal → Model → Generate) is a hand-made
illustration and is labelled as such on the page. It is replaced with the
trained model's real outputs once training and evaluation are done.

## Run it

Requires Python 3 (no other dependencies):

```
python web/serve.py
```

Then open http://localhost:5178 in Chrome or Edge on a desktop.

Use `serve.py` rather than `python -m http.server`. The built-in server
ignores HTTP Range requests, so the browser can't seek inside the video
files, and the return shot would start from the wrong frame.

## Video files

The two 4K clips (`media4k/whale.mp4`, `media4k/dolphins.mp4`, about 100 MB)
are not in git. Copy them into `web/media4k/` from the original
`whistle-ui/media4k/` folder; sources and encode settings are in
`CREDITS.md`. Without them the page still runs and shows the posters.

## What's where

- `index.html`: the four movements (Ocean, Listen, Research, Return) and their copy
- `shots.js`: the edit: shot windows, depth curve, text and visualisation timings (all in scroll position)
- `app.js`: the controller: playback, cross-fades, the research plot, the "Hear it" tone
- `audio-engine.js`: the illustrative whistle contours and tone (not the model)
- `media4k/`, `fonts/`: local assets (see `CREDITS.md`)

## Playback notes

On Windows the browser decodes video on the same GPU thread that draws the
page, so decoder work for a clip that is off screen costs frames on screen.
`app.js` therefore prerolls every later clip while the opening frame holds
still (at most 1.5 s, typically about 0.35 s), folds the depth shade into the
planes' opacity instead of drawing a full-screen layer, and hides the plot
canvas and the text veil whenever they are empty. On a 2560×1600, 240 Hz
display driven by a Radeon 610M this removed every frame stall measured
during a fast scroll from the top (95th-percentile frame time 16.6–20.9 ms →
4.3 ms) and through the whole film.
