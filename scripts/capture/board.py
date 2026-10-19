"""Generate a printable ChArUco board and its spec (board.json).

Print at 100% scale (no "fit to page"), then measure one square with calipers
and pass the measured size with --square_mm if it differs from the request.

Usage: python scripts/capture/board.py --out capture/board [--cols 9 --rows 6 --square_mm 30]
"""
import argparse
import cv2
import numpy as np
from common import make_board, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="prefix: writes <out>.png and <out>.json")
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square_mm", type=float, default=30.0)
ap.add_argument("--marker_mm", type=float, default=22.0)
ap.add_argument("--dict", default="5x5_100")
ap.add_argument("--dpi", type=int, default=300)
a = ap.parse_args()

spec = {"cols": a.cols, "rows": a.rows, "square_m": a.square_mm / 1000, "marker_m": a.marker_mm / 1000,
        "dict": a.dict}
px = a.square_mm / 25.4 * a.dpi
margin = int(round(px / 2))
img = make_board(spec).generateImage((int(round(a.cols * px)) + 2 * margin, int(round(a.rows * px)) + 2 * margin),
                                     marginSize=margin)
cv2.imwrite(a.out + ".png", img)
save_json(a.out + ".json", spec)
w_mm, h_mm = img.shape[1] / a.dpi * 25.4, img.shape[0] / a.dpi * 25.4
print(f"{a.out}.png: {img.shape[1]}x{img.shape[0]} px = {w_mm:.0f} x {h_mm:.0f} mm at {a.dpi} dpi; "
      f"print at 100% and check that one square measures {a.square_mm} mm")
