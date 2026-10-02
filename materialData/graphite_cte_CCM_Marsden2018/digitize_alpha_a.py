"""Digitize the dashed grey-blue curve (CCM acoustic and optical modes) from source_figure_alpha_a.png.

Same method as digitize_alpha_c.py: per image column, the sub-pixel centroid of the single thin run
of the curve's colour.

Axis calibration, from the tick marks (not the labels):
  x: major ticks at x = 146, 274, 401, 529, 656, 784 px for T = 0, 500, ..., 2500 K
  y: alpha = 0 on the drawn zero line (row 238.75); majors every 63.15 px = 0.5e-6/K
     (1.5e-6 at y = 49, -1.5e-6 at y = 428)

The curve colour is RGB (94, 129, 181) -- Mathematica's default first plot colour. Its anti-aliased
blends with white keep g - r > 0 and b > g. That separates it from the pure-blue Kellett markers
(0, 0, 255), whose blends have r = g, and from every other series.

Excluded: the whole legend box (x 340-640 px, y 295-380 px) -- its dashed sample line, and anti-aliased
glyph edges of the label text, which pass the colour test (8 stray points at alpha ~ -0.95 before).
The curve itself is at y < 160 px over those columns, so nothing real is masked.
Kept: T >= 300 K, the same lower bound as alpha_c -- the model never goes below room temperature.
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
im = np.array(Image.open(HERE / "source_figure_alpha_a.png").convert("RGB")).astype(float)
r, g, b = im[..., 0], im[..., 1], im[..., 2]

# Weight = how far the pixel is from white towards (94,129,181), measured on the red channel.
is_curve = (g - r > 8) & (b - g > 15) & (b > 150) & (r < 235)
weight = np.where(is_curve, (255.0 - r) / (255.0 - 94.0), 0.0)
weight[295:381, 340:641] = 0.0                              # legend box

FRAME_TOP, FRAME_BOTTOM = 29, 438
T_of_x = lambda x: (x - 146.0) * 2500.0 / (784.0 - 146.0)
alpha_of_y = lambda y: (238.75 - y) / 126.3                 # 1e-6/K

rows = []
x_start = int(np.ceil(146.0 + 300.0 * (784.0 - 146.0) / 2500.0))
for x in range(x_start, 785):
    col = weight[FRAME_TOP:FRAME_BOTTOM, x]
    ys = np.where(col > 0.3)[0]
    if len(ys) == 0:
        continue                                           # gap between dashes
    runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
    thin = [run for run in runs if len(run) <= 4]
    if len(thin) != 1 or len(thin) != len(runs):
        continue                                           # ambiguous column -- skip, don't guess
    run = thin[0]
    w = col[run]
    y = FRAME_TOP + float(np.sum(run * w) / np.sum(w))
    rows.append((T_of_x(x), alpha_of_y(y)))

with open(HERE / "digitized_alpha_a_CCM_acoustic_optical.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["T_K", "alpha_a_1e-6_per_K"])
    for T, a in rows:
        wr.writerow([f"{T:.1f}", f"{a:.4f}"])
print(f"{len(rows)} points, T {rows[0][0]:.0f}-{rows[-1][0]:.0f} K, "
      f"alpha_a {rows[0][1]:.3f} .. {rows[-1][1]:.3f} e-6/K")
