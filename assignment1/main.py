"""
Video Editor - side-by-side comparison with live histograms.

Layout
------
    +-------------------------+-------------------------+----------------+
    |      ORIGINAL video     |       EDITED video      |                |
    +-------------------------+-------------------------+   CONTROLS     |
    |    original histogram   |    edited histogram     |                |
    +-------------------------+-------------------------+----------------+

The sliders affect ONLY the right-hand video. The left stays untouched
so you always have a reference.

Settings (brightness, contrast, last exported file) are written to
settings.json, so relaunching the program restores the previous session:
the sliders come back where they were and the left panel plays the last
exported video, i.e. both windows open with the new settings applied.

Requires: opencv-python, pillow, numpy
    pip install opencv-python pillow numpy
"""

import json
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

SOURCE = "barriers.avi"
SETTINGS = "settings.json"
DISPLAY_W = 400        # medium-size preview, per panel
HIST_H = 140           # histogram strip height
FRAME_DELAY = 30       # ms between frames  (~33 fps)


class VideoEditor:
    def __init__(self, root, path=SOURCE):
        self.root = root
        self.path = path

        self.cap = cv.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open '{path}'")
            root.destroy()
            return

        src_w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        src_h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.disp_w = DISPLAY_W
        self.disp_h = int(src_h * DISPLAY_W / src_w)

        self.current = None          # most recent raw frame (full resolution)
        self.saved_frame = None      # most recent frame of the exported file
        self.paused = False
        self.saving = False
        self.save_progress = 0.0
        self.save_error = None
        self._pending_out = None

        # ---- restore settings from the last session ----
        cfg = self._load_settings()

        self.brightness = tk.DoubleVar(value=cfg.get("brightness", 0.0))
        self.contrast = tk.DoubleVar(value=cfg.get("contrast", 1.0))

        self.saved_path = cfg.get("saved_path") or None
        self.saved_cap = None                     # capture over the exported file
        self.show_saved = tk.BooleanVar(value=bool(cfg.get("show_saved", True)))

        # keeps PhotoImage objects alive; Tkinter does not hold its own ref
        self._refs = {}

        self._build_ui()
        self._open_saved()           # left panel picks up the exported video
        self._sync_labels()
        self.root.bind("<space>", lambda e: self._toggle_pause())
        self.root.bind("<Control-s>", lambda e: self._save())
        self.root.bind("<Control-S>", lambda e: self._save())
        self._tick()

    # ---- persistence ----

    def _load_settings(self):
        """Read settings.json. A missing or broken file is not an error."""
        try:
            with open(SETTINGS, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _store_settings(self):
        """Write the current settings so the next launch reproduces them."""
        data = {
            "brightness": round(self.brightness.get(), 2),
            "contrast": round(self.contrast.get(), 3),
            "saved_path": self.saved_path,
            "show_saved": bool(self.show_saved.get()),
        }
        try:
            with open(SETTINGS, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError:
            pass                      # never let a settings write break the app

    def _open_saved(self):
        """(Re)attach the capture that feeds the left-hand panel."""
        if self.saved_cap is not None:
            self.saved_cap.release()
            self.saved_cap = None
        self.saved_frame = None

        if self.saved_path and os.path.exists(self.saved_path):
            cap = cv.VideoCapture(self.saved_path)
            if cap.isOpened():
                self.saved_cap = cap
            else:
                cap.release()
                self.saved_path = None
        else:
            self.saved_path = None

        self._refresh_left_caption()

    def _using_saved(self):
        return self.saved_cap is not None and self.show_saved.get()

    def _refresh_left_caption(self):
        """The left column always states which file it is actually playing."""
        name = self.saved_path if self._using_saved() else self.path
        prefix = "Saved" if self._using_saved() else "Original"
        self.orig_caption.configure(text=f"{prefix} — {os.path.basename(name)}")
        self.saved_check.state(
            ["!disabled" if self.saved_cap is not None else "disabled"])

    def _toggle_saved(self):
        """Checkbox handler: rewind so the two panels stay frame-aligned."""
        if self.saved_cap is not None:
            self.saved_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        self.saved_frame = None
        self._refresh_left_caption()

    # ---- UI ----

    def _build_ui(self):
        self.root.title("Video Editor")

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # ---- LEFT: two video columns, histogram under each ----
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nw")

        (self.orig_caption, self.orig_video,
         self.orig_hist) = self._video_column(left, 0, "Original")
        (_, self.edit_video,
         self.edit_hist) = self._video_column(left, 1, "Edited")

        # ---- RIGHT: control panel ----
        right = ttk.LabelFrame(outer, text="Controls", padding=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        ttk.Label(right, text="Brightness").pack(anchor="w")
        ttk.Scale(right, from_=-100, to=100, orient="horizontal",
                  variable=self.brightness, length=180,
                  command=lambda v: self._sync_labels()).pack(fill="x")
        self.b_label = ttk.Label(right, text="0")
        self.b_label.pack(anchor="e")

        ttk.Label(right, text="Contrast").pack(anchor="w", pady=(12, 0))
        ttk.Scale(right, from_=0.1, to=3.0, orient="horizontal",
                  variable=self.contrast, length=180,
                  command=lambda v: self._sync_labels()).pack(fill="x")
        self.c_label = ttk.Label(right, text="1.00")
        self.c_label.pack(anchor="e")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=14)

        self.saved_check = ttk.Checkbutton(
            right, text="Left panel: exported video",
            variable=self.show_saved, command=self._toggle_saved)
        self.saved_check.pack(anchor="w", pady=(0, 8))

        self.play_btn = ttk.Button(right, text="Pause", command=self._toggle_pause)
        self.play_btn.pack(fill="x", pady=3)

        ttk.Button(right, text="Reset", command=self._reset).pack(fill="x", pady=3)

        self.save_btn = ttk.Button(right, text="Save Video…  (Ctrl+S)",
                                   command=self._save)
        self.save_btn.pack(fill="x", pady=3)

        self.progress = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(10, 2))
        self.status = ttk.Label(right, text="Ready", foreground="#555")
        self.status.pack(anchor="w")

    def _video_column(self, parent, col, caption):
        """One column: caption, video panel, histogram panel underneath."""
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, padx=6)

        # fixed width so swapping the caption cannot resize the window
        title = ttk.Label(frame, text=caption, width=30, anchor="center",
                          font=("TkDefaultFont", 10, "bold"))
        title.pack()

        video = ttk.Label(frame, borderwidth=1, relief="solid")
        video.pack()

        ttk.Label(frame, text="Histogram", foreground="#555").pack(pady=(6, 0))

        hist = ttk.Label(frame, borderwidth=1, relief="solid")
        hist.pack()

        return title, video, hist

    def _sync_labels(self):
        self.b_label.configure(text=f"{self.brightness.get():.0f}")
        self.c_label.configure(text=f"{self.contrast.get():.2f}")

    # ---- image pipeline ----

    def _process(self, frame):
        """The one place editing happens. Pure function of (frame, params).

        g(x) = alpha*(f(x) - 127.5) + 127.5 + beta

        Contrast (alpha) scales around mid-gray (127.5) instead of zero, so
        it stretches/compresses the tonal range without also shifting the
        overall brightness. Brightness (beta) is the only additive term, so
        it's the only control that uniformly lightens/darkens the frame.
        Rewritten as alpha*f(x) + beta' so cv.convertScaleAbs can still do
        the scaling and clipping in one call.
        """
        alpha = self.contrast.get()
        beta = self.brightness.get()
        beta_centered = beta + 127.5 * (1.0 - alpha)
        return cv.convertScaleAbs(frame, alpha=alpha, beta=beta_centered)

    def _hist_image(self, frame, w, h):
        """Draw a 3-channel histogram into a BGR image."""
        canvas = np.full((h, w, 3), 28, np.uint8)
        colors = ((255, 110, 110), (110, 255, 110), (110, 110, 255))  # B, G, R
        xs = np.linspace(0, w - 1, 256).astype(np.int32)

        for ch, color in enumerate(colors):
            hist = cv.calcHist([frame], [ch], None, [256], [0, 256]).flatten()
            cv.normalize(hist, hist, 0, h - 6, cv.NORM_MINMAX)
            ys = (h - 3 - hist).astype(np.int32)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv.polylines(canvas, [pts], False, color, 1, cv.LINE_AA)

        return canvas

    def _show(self, label, bgr, key):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=img)
        self._refs[key] = img          # without this the panel goes blank

    # ---- main loop ----

    @staticmethod
    def _read_looping(cap):
        """Next frame, rewinding to frame 1 at end of file."""
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        return frame if ret else None

    def _tick(self):
        if not self.paused:
            frame = self._read_looping(self.cap)
            if frame is not None:
                self.current = frame

            if self.saved_cap is not None:
                frame = self._read_looping(self.saved_cap)
                if frame is not None:
                    self.saved_frame = frame

        # left panel: the exported file when we have one, else the source
        left_src = self.saved_frame if self._using_saved() else self.current

        if self.current is not None:
            if left_src is None:
                left_src = self.current
            left = cv.resize(left_src, (self.disp_w, self.disp_h))
            small = cv.resize(self.current, (self.disp_w, self.disp_h))
            edited = self._process(small)

            self._show(self.orig_video, left, "ov")
            self._show(self.edit_video, edited, "ev")
            self._show(self.orig_hist,
                       self._hist_image(left, self.disp_w, HIST_H), "oh")
            self._show(self.edit_hist,
                       self._hist_image(edited, self.disp_w, HIST_H), "eh")

        self.root.after(FRAME_DELAY, self._tick)

    # ---- actions ----

    def _toggle_pause(self):
        self.paused = not self.paused
        self.play_btn.configure(text="Play" if self.paused else "Pause")

    def _reset(self):
        self.brightness.set(0.0)
        self.contrast.set(1.0)
        self._sync_labels()

    def _save(self):
        if self.saving:
            return
        out_path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            filetypes=[("AVI", "*.avi"), ("MP4", "*.mp4")],
            initialfile="edited.avi")
        if not out_path:
            return

        self.saving = True
        self.save_error = None
        self.save_progress = 0.0
        self._pending_out = out_path
        self.save_btn.state(["disabled"])
        self.status.configure(text="Saving…")

        # snapshot the settings so later slider moves don't affect this export
        args = (out_path, self.contrast.get(), self.brightness.get())
        threading.Thread(target=self._save_worker, args=args, daemon=True).start()
        self._poll_save()

    def _save_worker(self, out_path, alpha, beta):
        """Runs off the UI thread. Re-reads the source at full resolution."""
        cap = writer = None
        try:
            cap = cv.VideoCapture(self.path)
            fps = cap.get(cv.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 1

            ext = os.path.splitext(out_path)[1].lower()
            fourcc = cv.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))
            writer = cv.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise IOError("VideoWriter could not be opened")

            n = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(cv.convertScaleAbs(frame, alpha=alpha, beta=beta))
                n += 1
                self.save_progress = min(100.0, 100.0 * n / total)
        except Exception as exc:
            self.save_error = str(exc)
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            self.saving = False

    def _poll_save(self):
        """UI-thread poller: never touch widgets from the worker thread."""
        self.progress["value"] = self.save_progress
        if self.saving:
            self.root.after(100, self._poll_save)
        else:
            self.progress["value"] = 0 if self.save_error else 100
            self.save_btn.state(["!disabled"])
            if self.save_error:
                self.status.configure(text="Save failed", foreground="#b00")
                messagebox.showerror("Save failed", self.save_error)
            else:
                # the export becomes the left panel's source from now on,
                # and survives a restart through settings.json
                self.saved_path = self._pending_out
                self._open_saved()
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                self._store_settings()
                self.status.configure(text="Saved", foreground="#070")

    def close(self):
        self._store_settings()
        self.cap.release()
        if self.saved_cap is not None:
            self.saved_cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = VideoEditor(root, SOURCE)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    root.mainloop()