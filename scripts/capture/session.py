"""Per-step masks of the removed object, from image differences, plus a disturbance check.

Between step k and step k+1 the only intended change is the removal of one object,
so |I_k - I_{k+1}| in every camera is that object's visible silhouette. Changes
away from it mean a neighbour was nudged (or the light changed) and are reported.

Scene layout (one photo per camera per step, file name = camera name):
  <scene>/steps/step_00/{main,phoneA,...}.png|jpg     all objects in place
  <scene>/steps/step_01/...                           first object removed
  ...
  <scene>/steps/step_NN/...                           empty table (all removed)
  <scene>/removed.txt                                 object names in removal order,
                                                      one per line (library keys)
Writes <scene>/undist/step_XX/<cam>.png, <scene>/masks/step_XX/<cam>.png (the
object removed between step XX and XX+1) and <scene>/session_report.json.

Lock exposure, white balance and focus on every camera; auto-exposure changes
between shots show up as global differences.

Usage: python scripts/capture/session.py --rig rig.json --scene captures/scene_001
"""
import argparse, glob, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_rig, save_json, undistort


def read_steps(scene, cams):
    steps = sorted(glob.glob(os.path.join(scene, "steps", "step_*")))
    imgs = {}
    for c in cams:
        seq = []
        for s in steps:
            f = [p for p in glob.glob(os.path.join(s, c + ".*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not f:
                raise SystemExit(f"missing {c} image in {s}")
            seq.append(cv2.imread(f[0]))
        imgs[c] = seq
    return steps, imgs


def removal_mask(A, B, a):
    """Mask of what disappeared between A and B, and the stray changes elsewhere."""
    Af, Bf = A.astype(np.float32), B.astype(np.float32)
    g = (np.median(Af.reshape(-1, 3), 0) + 1) / (np.median(Bf.reshape(-1, 3), 0) + 1)
    Bf = Bf * g                                              # absorb a small global exposure drift
    d = cv2.GaussianBlur(np.abs(Af - Bf).max(2), (5, 5), 0)
    mad = np.median(np.abs(d - np.median(d)))
    raw = d > max(a.min_thr, 6 * 1.4826 * mad)
    if a.suppress_shadows:
        # a vanished cast shadow: after-removal is brighter by a similar factor in all
        # channels (same chromaticity); a vanished object usually changes chromaticity
        r = (Af + 1) / (Bf + 1)
        ca, cb = Af / (Af.sum(2, keepdims=True) + 1), Bf / (Bf.sum(2, keepdims=True) + 1)
        shadow = np.all((r > 0.3) & (r < 0.85), 2) & (np.abs(ca - cb).max(2) < 0.03)
        raw &= ~shadow
    m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, bool), []
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    near = cv2.dilate((lab == main).astype(np.uint8), np.ones((a.join_px, a.join_px), np.uint8)) > 0
    obj = np.zeros(m.shape, bool)
    stray = []
    for i in range(1, n):
        comp = lab == i
        if i == main or (comp & near).any():
            obj |= comp
        elif stats[i, cv2.CC_STAT_AREA] >= a.min_area:
            x, y, w, h = stats[i, :4]
            stray.append({"area_px": int(stats[i, cv2.CC_STAT_AREA]), "bbox": [int(x), int(y), int(w), int(h)]})
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    return obj, stray


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--min_thr", type=float, default=18.0, help="min per-pixel change (0-255) counted as removal")
    ap.add_argument("--min_area", type=int, default=150, help="stray change (px) reported as disturbance")
    ap.add_argument("--join_px", type=int, default=25, help="pieces this close to the main blob belong to it")
    ap.add_argument("--suppress_shadows", action="store_true")
    a = ap.parse_args()

    rig = load_rig(a.rig)
    cams = [c for c, v in rig["cams"].items() if "c2w" in v]
    steps, imgs = read_steps(a.scene, cams)
    removed = [l.strip() for l in open(os.path.join(a.scene, "removed.txt")) if l.strip()]
    if len(removed) != len(steps) - 1:
        raise SystemExit(f"{len(steps)} steps need {len(steps) - 1} names in removed.txt, found {len(removed)}")

    report = {"cams": cams, "removed": removed, "steps": []}
    for c in cams:
        imgs[c] = [undistort(im, rig["cams"][c]) for im in imgs[c]]
        for k, im in enumerate(imgs[c]):
            p = os.path.join(a.scene, "undist", f"step_{k:02d}", c + ".png")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, im)
    for k in range(len(steps) - 1):
        row = {"step": k, "object": removed[k], "cams": {}}
        for c in cams:
            m, stray = removal_mask(imgs[c][k], imgs[c][k + 1], a)
            p = os.path.join(a.scene, "masks", f"step_{k:02d}", c + ".png")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, m.astype(np.uint8) * 255)
            row["cams"][c] = {"mask_px": int(m.sum()), "disturbance": stray}
            if stray:
                print(f"  step {k} ({removed[k]}), {c}: {len(stray)} change(s) away from the removed object, "
                      f"largest {max(s['area_px'] for s in stray)} px; check that nothing else moved")
        report["steps"].append(row)
        print(f"step {k}: {removed[k]}  mask px " + "  ".join(f"{c}:{row['cams'][c]['mask_px']}" for c in cams))
    save_json(os.path.join(a.scene, "session_report.json"), report)


if __name__ == "__main__":
    main()
