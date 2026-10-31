# Inspecting Luna runs in ChatGPT

The server's `watch.html` is a live HTTP page. It loads `watch.js`,
`watch-status.json`, each trial's `watch-data.json`, and media using relative
URLs. Opening only that HTML in an isolated preview does not supply those
companion files. A page shell can appear while analysis and videos fail to load.

`tools/export_watch_preview.py` creates `watch-chat.html`, a self-contained
snapshot with inline code/data, embedded MP4 previews, camera images, task
outcomes, per-tool timings, contact-call history and recorded explanations.
There are no network requests or server dependencies after export. A run selector
provides access to every published trial; videos retain their original duration.
This is a saved snapshot, not an automatically synchronized live dashboard.

```bash
python tools/export_watch_preview.py \
  results/luna-explicit-motion-rerun-20261031
```

For an in-conversation view, pass `--fragment /absolute/writable/thread/path/luna-watch.html`
and return that fragment using the visualization skill. The default export limit
is 950,000 bytes, below the inline renderer's 1 MB limit. Standalone exports can
use `--max-bytes 8000000` for higher preview quality. Embedded videos reduce
resolution/frame rate as needed, never playback duration. If a recording cannot
fit, the page explicitly reports that limitation rather than silently dropping it.

In the conversation host, “Ask for an updated snapshot” requests a fresh export.
“Show full-resolution recording” requests the original recording in chat. These
actions are hidden when the host API is unavailable. Outside ChatGPT, regenerate
and reopen the exported page to see newly published data. VS Code's original
live page continues to work as before.

The exporter reads only the published index/details and explicitly referenced
media. FFmpeg uses one CPU thread and no GPU. Compressed assets are cached in
`/tmp/luna-watch-preview-cache` by source timestamp, size and encoding profile;
unchanged exports are not rewritten. It starts no watcher, HTTP server or polling
job, and does not modify the active experiment's frozen runtime sources.

Validation: Chrome with networking disabled loads the exported trial selector,
timings, diagnoses and camera images, and plays embedded H.264 recordings. The
page's own content policy disallows network connections. This checks the isolated
preview behavior locally; it is not an end-to-end test of the user's ChatGPT client.
