"""Shared pieces of the physical-capture toolkit (see docs/capture.md).

World frame: the ChArUco board lying on the table, x along its columns, z UP
(out of the table), so the table is the plane z = 0. Cameras use the OpenCV
convention (x right, y down, z forward); c2w maps camera to world.
"""
import json, os
import cv2
import numpy as np
import open3d as o3d

DICTS = {"4x4_100": cv2.aruco.DICT_4X4_100, "5x5_100": cv2.aruco.DICT_5X5_100,
         "6x6_250": cv2.aruco.DICT_6X6_250}
# board frame (y down the printed page, z into the table) -> world (z up)
W_FROM_B = np.diag([1.0, -1.0, -1.0])


def make_board(spec):
    d = cv2.aruco.getPredefinedDictionary(DICTS[spec["dict"]])
    return cv2.aruco.CharucoBoard((spec["cols"], spec["rows"]), spec["square_m"], spec["marker_m"], d)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(p, obj):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def load_rig(path):
    rig = load_json(path)
    for c in rig["cams"].values():
        c["K"], c["dist"] = np.array(c["K"]), np.array(c["dist"])
        if "c2w" in c:
            c["c2w"] = np.array(c["c2w"])
    return rig


def undistort(img, cam):
    """Undistort keeping K unchanged, so every later step uses the pinhole K."""
    return cv2.undistort(img, cam["K"], cam["dist"])


def pixel_rays_world(cam, ds=1, box=None):
    """World-space rays for a pixel grid (every ds-th pixel), optionally inside a box
    (x0, y0, x1, y1). Returns origins, unnormalised directions (z_cam = 1, so the
    ray parameter equals camera depth), and the pixel coordinates."""
    W, H = cam["size"]
    x0, y0, x1, y1 = box if box is not None else (0, 0, W, H)
    us, vs = np.meshgrid(np.arange(x0, x1, ds) + 0.5, np.arange(y0, y1, ds) + 0.5)
    K = cam["K"]
    d = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1).reshape(-1, 3)
    R, t = cam["c2w"][:3, :3], cam["c2w"][:3, 3]
    return np.broadcast_to(t, d.shape).copy(), d @ R.T, us.reshape(-1), vs.reshape(-1)


def raycast_depth(scene, origins, dirs):
    r = scene.cast_rays(o3d.core.Tensor(np.hstack([origins, dirs]).astype(np.float32)))
    return r["t_hit"].numpy(), r["geometry_ids"].numpy()


def load_mesh_m(path, units):
    """CAD mesh as (V, F) in metres."""
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles) == 0:
        raise SystemExit(f"{path}: no triangles")
    s = {"m": 1.0, "cm": 0.01, "mm": 0.001}[units]
    return np.asarray(m.vertices) * s, np.asarray(m.triangles).astype(np.int64)


def transform(V, T):
    return V @ T[:3, :3].T + T[:3, 3]
