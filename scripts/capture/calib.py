"""Camera calibration for a capture session, all into one rig.json.

  intrinsics  calibrate one camera from 15-30 ChArUco photos taken at varied angles
              and distances (same focus/zoom you will capture with)
  import      take the main camera's intrinsics from an existing OpenCV stereo
              calibration XML (cameraMatrixL/distL ...) instead
  extrinsics  one photo per camera of the board lying flat on the table, cameras in
              their final positions; gives every camera's pose in the board/world
              frame. Remove the board afterwards and do not move the cameras.

Usage:
  python scripts/capture/calib.py intrinsics --rig rig.json --board board.json --cam phoneA --images calib/phoneA
  python scripts/capture/calib.py import --rig rig.json --cam main --xml stereo_calibration_params.xml --side L --size 746 668
  python scripts/capture/calib.py extrinsics --rig rig.json --board board.json --shot session/board_shot
      (board_shot/<cam>.png|jpg, one image per camera, file name = camera name)
"""
import argparse, glob, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W_FROM_B, load_json, make_board, save_json


def detect(img, board):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ch_c, ch_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None, None
    obj, imgp = board.matchImagePoints(ch_c, ch_ids)
    return obj.reshape(-1, 3), imgp.reshape(-1, 2)


def rig_load(path):
    return load_json(path) if os.path.exists(path) else {"cams": {}}


def cmd_intrinsics(a):
    board = make_board(load_json(a.board))
    objs, imgs, size = [], [], None
    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    for f in files:
        im = cv2.imread(f)
        size = (im.shape[1], im.shape[0])
        o, p = detect(im, board)
        if o is not None:
            objs.append(o.astype(np.float32)); imgs.append(p.astype(np.float32))
    if len(objs) < 8:
        raise SystemExit(f"board found in only {len(objs)}/{len(files)} images; need >= 8")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    rig = rig_load(a.rig)
    rig["cams"].setdefault(a.cam, {}).update({"K": K.tolist(), "dist": dist.ravel().tolist(), "size": list(size),
                                              "intrinsics_rms_px": float(rms), "intrinsics_from": a.images})
    save_json(a.rig, rig)
    print(f"{a.cam}: {len(objs)}/{len(files)} images, reprojection RMS {rms:.3f} px, f = {K[0,0]:.1f}")
    if rms > 1.0:
        print("  RMS > 1 px: retake with sharper, more varied board views (tilted, near the corners)")


def cmd_import(a):
    fs = cv2.FileStorage(a.xml, cv2.FILE_STORAGE_READ)
    K, dist = fs.getNode("cameraMatrix" + a.side).mat(), fs.getNode("dist" + a.side).mat()
    rig = rig_load(a.rig)
    rig["cams"].setdefault(a.cam, {}).update({"K": K.tolist(), "dist": dist.ravel().tolist(), "size": a.size,
                                              "intrinsics_from": f"{a.xml}:{a.side}"})
    save_json(a.rig, rig)
    print(f"{a.cam}: imported f = {K[0,0]:.1f}, c = ({K[0,2]:.1f}, {K[1,2]:.1f}), dist = {dist.ravel().round(3).tolist()}")


def cmd_extrinsics(a):
    board = make_board(load_json(a.board))
    rig = rig_load(a.rig)
    for name, cam in rig["cams"].items():
        hits = [f for f in glob.glob(os.path.join(a.shot, name + ".*"))]
        if not hits:
            print(f"{name}: no image in {a.shot}, skipped")
            continue
        im = cv2.imread(hits[0])
        if [im.shape[1], im.shape[0]] != list(cam["size"]):
            raise SystemExit(f"{name}: board shot is {im.shape[1]}x{im.shape[0]}, intrinsics are for {cam['size']}")
        K, dist = np.array(cam["K"]), np.array(cam["dist"])
        o, p = detect(im, board)
        if o is None:
            raise SystemExit(f"{name}: board not detected in {hits[0]} (markers need roughly 25+ px "
                             "across: print a bigger board or bring the camera closer for this shot)")
        ok, rvec, tvec = cv2.solvePnP(o, p, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = cv2.solvePnPRefineLM(o, p, K, dist, rvec, tvec)
        proj, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
        err = float(np.sqrt(((proj.reshape(-1, 2) - p) ** 2).sum(1).mean()))
        T_cb = np.eye(4); T_cb[:3, :3] = cv2.Rodrigues(rvec)[0]; T_cb[:3, 3] = tvec.ravel()   # board -> camera
        B2W = np.eye(4); B2W[:3, :3] = W_FROM_B
        c2w = B2W @ np.linalg.inv(T_cb)
        cam.update({"c2w": c2w.tolist(), "extrinsics_reproj_px": err, "extrinsics_corners": int(len(o))})
        h = c2w[2, 3]
        print(f"{name}: {len(o)} corners, reprojection {err:.2f} px, camera {h*100:.1f} cm above the table")
        if h <= 0:
            print("  camera is below the table plane: the board was probably upside down in the shot")
    save_json(a.rig, rig)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("intrinsics"); p.add_argument("--rig", required=True); p.add_argument("--board", required=True)
p.add_argument("--cam", required=True); p.add_argument("--images", required=True)
p = sub.add_parser("import"); p.add_argument("--rig", required=True); p.add_argument("--cam", required=True)
p.add_argument("--xml", required=True); p.add_argument("--side", choices=["L", "R"], default="L")
p.add_argument("--size", type=int, nargs=2, required=True, metavar=("W", "H"))
p = sub.add_parser("extrinsics"); p.add_argument("--rig", required=True); p.add_argument("--board", required=True)
p.add_argument("--shot", required=True)

if __name__ == "__main__":
    a = ap.parse_args()
    {"intrinsics": cmd_intrinsics, "import": cmd_import, "extrinsics": cmd_extrinsics}[a.cmd](a)
