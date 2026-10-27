# Video Brightness/Contrast Editor

A small desktop app built with OpenCV and Tkinter for CAP 5410
(Computer Vision), Assignment 1. It loads a video, shows it side by
side with a live-adjusted preview, and draws a histogram under each
one. Brightness and contrast are controlled with two scroll bars, the
adjusted video can be exported to a new file, and the last settings
are restored the next time the program is launched.

## Features

- Loads and plays `barriers.avi` continuously, restarting from frame 0
  as soon as it reaches the end.
- Shows a live histogram for both the original and the adjusted video.
- Brightness and contrast scroll bars update the preview and its
  histogram on the next frame, with no extra wiring.
- Saves the adjusted video to a new file (button or `Ctrl+S`), with a
  progress bar and status message.
- Remembers brightness, contrast, and the last exported file in
  `settings.json`, so both windows reopen the same way next time.
- Extras: pause/resume (spacebar), reset sliders, and a checkbox to
  toggle the left panel between the original source and the last
  exported file.

## Requirements

- Python 3.13
- Dependencies listed in `requirements.txt`:

  ```
  pip install -r requirements.txt
  ```

## Running it

```
python main.py
```

## Controls

| Control | Action |
|---|---|
| Brightness slider | Shifts pixel intensity up/down (`-100` to `100`) |
| Contrast slider | Scales pixel intensity around mid-gray (`0.1` to `3.0`) |
| Pause / spacebar | Pause or resume playback |
| Reset | Sets both sliders back to their defaults |
| Save Video... / `Ctrl+S` | Exports the adjusted video to a new file |
| Left panel checkbox | Switches the left panel between the original source and the last exported file |

## Files

- `main.py` -- the application.
- `requirements.txt` -- pinned dependency versions.
- `barriers.avi` -- the source video used by the app.
