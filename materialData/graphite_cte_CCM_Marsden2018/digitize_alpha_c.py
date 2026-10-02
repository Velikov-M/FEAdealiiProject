"""Digitize the red dashed curve (CCM acoustic + optical modes) from source_figure_alpha_c.png.

Pixel-based, not by eye: every image column is scanned for pure-red pixels and the sub-pixel
centroid of the thin run (the dashed line) is taken as alpha_c at that column's temperature.

Axis calibration, from the tick marks themselves (not the labels):
  x: minor ticks every 18.5 px = 100 K, T = 0 at x = 95.5 px (500 K major at 188, 2500 K at 558)
  y: majors every 38.5 px = 5e-6/K, alpha = 35e-6 at y = 38 px, 5e-6 at y = 269 px
Sanity check: the extracted curve gives alpha_c(100 K) ~ 17e-6, matching Bailey & Yates' data.

Only T >= 300 K is kept. Below that the curve runs through the (identically red) Bailey & Yates
markers and cannot be separated from them -- and the model never goes below room temperature.
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
im = np.array(Image.open(HERE / "source_figure_alpha_c.png").convert("RGB")).astype(float)
r, g, b = im[..., 0], im[..., 1], im[..., 2]

# Pure red and its anti-aliased blends with white are (255, v, v). Orange markers (255, 128, 0)
# and orange-red overlaps have g >> b, so |g - b| excludes them.
redness = np.where((r > 200) & (np.abs(g - b) < 20), (r - g) / 255.0, 0.0)

FRAME_TOP, FRAME_BOTTOM = 17, 312          # inside the plot frame
T_of_x = lambda x: (x - 95.5) * 100.0 / 18.5
alpha_of_y = lambda y: 35.0 - (y - 38.0) * 30.0 / 231.0   # 1e-6/K

rows = []
for x in range(int(95.5 + 3.0 * 18.5), 559):              # 300 K .. 2500 K
    col = redness[FRAME_TOP:FRAME_BOTTOM, x]
    ys = np.where(col > 0.3)[0]
    if len(ys) == 0:
        continue                                           # gap between dashes
    runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
    thin = [run for run in runs if len(run) <= 4]          # the line is 2-3 px thick
    if len(thin) != 1:
        continue                                           # ambiguous column -- skip, don't guess
    run = thin[0]
    w = col[run]
    y = FRAME_TOP + float(np.sum(run * w) / np.sum(w))
    rows.append((T_of_x(x), alpha_of_y(y)))

with open(HERE / "digitized_alpha_c_CCM_acoustic_optical.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["T_K", "alpha_c_1e-6_per_K"])
    for T, a in rows:
        wr.writerow([f"{T:.1f}", f"{a:.4f}"])
print(f"{len(rows)} points, T {rows[0][0]:.0f}-{rows[-1][0]:.0f} K, "
      f"alpha_c {rows[0][1]:.2f}-{rows[-1][1]:.2f} e-6/K")
