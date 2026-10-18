import blenderproc as bproc   # must be the first line: `blenderproc run` checks for it
# Generate one synthetic layered tabletop scene with full occlusion ground truth.
#
# Run with BlenderProc (one scene per call; see make_dataset.sh):
#   blenderproc run scripts/data/gen_layered_scene.py --models_dir data/gso \
#       --out_dir data/layered --scene_id s0000 --seed 0
#
# A table in a walled room, 4-9 Google Scanned Objects dropped in a cluster with
# physics, and a low camera looking across the cluster so objects occlude each
# other. Camera poses are rejected until the GT occlusion graph (found by
# multi-hit ray casting) has at least --min_pairs occluding object pairs.
#
# Writes <out_dir>/<scene_id>/:
#   rgb.png, depth.npy (z-depth, m), inst.npy (int16, -1 = background)
#   gt_mesh.ply       every surface in the scene, world frame, triangles
#   face_inst.npy     instance id per gt_mesh face (-1 = table / room)
#   cam.npz           K, width, height, c2w (OpenCV convention, world frame)
#   peel/step_XX_{rgb.png,depth.npy,inst.npy}
#                     GT clean plates: the scene re-rendered after physically
#                     removing the first XX objects of the GT peel order
#   meta.json         objects, GT occlusion edges (+ ray support), GT peel order,
#                     edges dropped to break cycles, camera, seed
# and appends "<scene_id> rgb.png gt_mesh.ply cam.npz" to <out_dir>/scenes.txt,
# the format scripts/run_pilot.sh reads.
import argparse, glob, json, os, random
import numpy as np
import bpy
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument("--models_dir", required=True, help="GSO root: <name>/meshes/model.obj")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--scene_id", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_min", type=int, default=4)
ap.add_argument("--n_max", type=int, default=9)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--samples", type=int, default=64, help="Cycles samples per render")
ap.add_argument("--min_pairs", type=int, default=2, help="min GT occluding object pairs")
ap.add_argument("--min_support", type=int, default=3, help="rays (at --ray_stride) needed for an edge")
ap.add_argument("--ray_stride", type=int, default=8)
ap.add_argument("--max_cam_tries", type=int, default=60)
ap.add_argument("--up_axis", default="Z", help="up axis of the model .obj files (GSO: Z)")
ap.add_argument("--cc_textures", default=None, help="optional ambientCG/CC0 texture dir for table/room")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
np.random.seed(a.seed)                 # BlenderProc samplers draw from these
random.seed(a.seed)
SDIR = os.path.join(a.out_dir, a.scene_id)
os.makedirs(os.path.join(SDIR, "peel"), exist_ok=True)
bproc.init()

TABLE_TOP = 0.75


# ── room + table ─────────────────────────────────────────────────────────────
def material(name):
    m = bproc.material.create(name)
    m.set_principled_shader_value("Base Color", [*rng.uniform(0.15, 0.9, 3), 1.0])
    m.set_principled_shader_value("Roughness", float(rng.uniform(0.3, 0.9)))
    return m


cc = None
if a.cc_textures:
    cc = bproc.loader.load_ccmaterials(a.cc_textures, ["Wood", "Fabric", "Tiles", "Concrete", "Plaster", "Marble"])


def dress(obj, name):
    obj.clear_materials()
    obj.add_material(random.choice(cc) if cc else material(name))


room = []
floor = bproc.object.create_primitive("PLANE", scale=[3, 3, 1])
room.append(floor)
for i, (loc, rot) in enumerate([([0, 3, 1.5], [np.pi / 2, 0, 0]), ([0, -3, 1.5], [-np.pi / 2, 0, 0]),
                                ([3, 0, 1.5], [0, -np.pi / 2, 0]), ([-3, 0, 1.5], [0, np.pi / 2, 0])]):
    w = bproc.object.create_primitive("PLANE", scale=[3, 1.5, 1], location=loc, rotation=rot)
    room.append(w)
tw, td = rng.uniform(0.9, 1.4), rng.uniform(0.6, 0.9)
table = bproc.object.create_primitive("CUBE", scale=[tw / 2, td / 2, 0.02], location=[0, 0, TABLE_TOP - 0.02])
body = bproc.object.create_primitive("CUBE", scale=[tw / 2 - 0.05, td / 2 - 0.05, (TABLE_TOP - 0.04) / 2],
                                     location=[0, 0, (TABLE_TOP - 0.04) / 2])
dress(floor, "floor")
for i, w in enumerate(room[1:]):
    dress(w, f"wall{i}")
dress(table, "table")
body.clear_materials(); body.add_material(table.get_materials()[0])
for o in room + [table, body]:
    o.set_cp("inst_id", -1)
    o.enable_rigidbody(False, collision_shape="BOX" if o in (table, body) else "MESH")

# ── objects ──────────────────────────────────────────────────────────────────
paths = sorted(glob.glob(os.path.join(a.models_dir, "*", "meshes", "model.obj")))
if len(paths) < a.n_max:
    raise SystemExit(f"only {len(paths)} models in {a.models_dir}")
n = int(rng.integers(a.n_min, a.n_max + 1))
chosen = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
objs = []
for i, p in enumerate(chosen):
    parts = bproc.loader.load_obj(p, forward_axis="Y" if a.up_axis == "Z" else "NEGATIVE_Z", up_axis=a.up_axis)
    o = parts[0]
    if len(parts) > 1:
        o.join_with_other_objects(parts[1:])
    o.set_name(f"obj_{i:02d}")
    o.set_cp("inst_id", i)
    o.set_cp("source", os.path.basename(os.path.dirname(os.path.dirname(p))))
    o.enable_rigidbody(True, collision_shape="CONVEX_HULL", mass=1.0, friction=0.8)
    objs.append(o)

center = np.array([rng.uniform(-0.15, 0.15) * tw, rng.uniform(-0.1, 0.1) * td])
spread = rng.uniform(0.08, 0.16)                 # tight cluster -> objects overlap in view


def place(obj):
    xy = center + rng.normal(0, spread, 2)
    xy = np.clip(xy, [-tw / 2 + 0.08, -td / 2 + 0.08], [tw / 2 - 0.08, td / 2 - 0.08])
    obj.set_location([xy[0], xy[1], TABLE_TOP + 0.2])
    obj.set_rotation_euler([0, 0, rng.uniform(0, 2 * np.pi)])


placed = bproc.object.sample_poses_on_surface(objs, table, place, max_tries=200, min_distance=0.0,
                                              max_distance=2.0, check_all_bb_corners_over_surface=False)
for o in objs:                                  # the sampler only hides what it could not place
    if o not in placed:
        o.delete()
objs = placed
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=10,
                                                  check_object_interval=1)
kept = []
for o in objs:                                  # drop anything that fell off the table
    if o.get_bound_box().min(0)[2] > TABLE_TOP - 0.05:
        kept.append(o)
    else:
        o.delete()
objs = kept
if len(objs) < 3:
    raise SystemExit("fewer than 3 objects stayed on the table; try another seed")
by_id = {o.get_cp("inst_id"): o for o in objs}

# ── lighting ─────────────────────────────────────────────────────────────────
bproc.renderer.set_world_background([1, 1, 1], strength=float(rng.uniform(0.2, 0.8)))
for _ in range(int(rng.integers(1, 3))):
    L = bproc.types.Light()
    L.set_type("POINT")
    L.set_location([*rng.uniform(-1.5, 1.5, 2), rng.uniform(1.8, 2.8)])
    L.set_energy(float(rng.uniform(150, 600)))

# ── camera: sample until the GT occlusion graph is rich enough ──────────────
W, H = a.width, a.height
hfov = np.deg2rad(rng.uniform(50, 65))
f = W / 2 / np.tan(hfov / 2)
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
bproc.camera.set_intrinsics_from_K_matrix(K, W, H)


def occlusion_graph(c2w):
    """Multi-hit ray casting on a pixel grid: along each ray, the order in which
    objects are first hit. support[(a, b)] counts rays where a is in front of b."""
    dg = bpy.context.evaluated_depsgraph_get()
    R, o = c2w[:3, :3], c2w[:3, 3]
    support, first_px, near = {}, {}, {}
    for v in range(a.ray_stride // 2, H, a.ray_stride):
        for u in range(a.ray_stride // 2, W, a.ray_stride):
            d = R @ np.array([(u + 0.5 - K[0, 2]) / f, -(v + 0.5 - K[1, 2]) / f, -1.0])
            d /= np.linalg.norm(d)
            org, seq, travelled = o.copy(), [], 0.0
            for hit in range(10):
                ok, loc, _, _, ob, _ = bpy.context.scene.ray_cast(dg, Vector(org), Vector(d), distance=20.0)
                if not ok:
                    break
                loc = np.array(loc)
                travelled += float(np.linalg.norm(loc - org))
                iid = int(ob.get("inst_id", -1))
                if iid >= 0 and iid not in seq:
                    if not seq and hit == 0:
                        first_px[iid] = first_px.get(iid, 0) + 1
                    near[iid] = min(near.get(iid, 1e9), travelled)
                    seq.append(iid)
                org = loc + d * 1e-4
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    support[(seq[i], seq[j])] = support.get((seq[i], seq[j]), 0) + 1
    edges = {e: c for e, c in support.items() if c >= a.min_support}
    return edges, first_px, near


def gt_order(nodes, edges, near):
    """Kahn sort of the GT graph; nearest object first among free choices; a cycle
    is broken by dropping its least-supported edge (recorded)."""
    E = dict(edges)
    order, dropped = [], []
    while len(order) < len(nodes):
        rest = [x for x in nodes if x not in order]
        free = [x for x in rest if not any((p, x) in E for p in rest if p != x)]
        if not free:
            e = min((e for e in E if e[0] in rest and e[1] in rest), key=E.get)
            dropped.append([int(e[0]), int(e[1]), int(E.pop(e))])
            continue
        order.append(min(free, key=lambda x: near[x]))
    return order, dropped


poi = np.mean([o.get_bound_box().mean(0) for o in objs], 0)
best = None
for t in range(a.max_cam_tries):
    loc = bproc.sampler.shell(center=poi, radius_min=0.45, radius_max=0.95, elevation_min=8,
                              elevation_max=40, azimuth_min=-180, azimuth_max=180)
    if loc[2] < TABLE_TOP + 0.05:
        continue
    rot = bproc.camera.rotation_from_forward_vec(poi + rng.normal(0, 0.03, 3) - loc,
                                                 inplane_rot=float(rng.uniform(-0.1, 0.1)))
    c2w = bproc.math.build_transformation_mat(loc, rot)
    edges, first_px, near = occlusion_graph(c2w)
    score = (len(edges), len(first_px))
    if best is None or score > best[0]:
        best = (score, c2w, edges, first_px, near)
    if len(edges) >= a.min_pairs and len(first_px) >= max(3, len(objs) - 2):
        break
(n_edges, n_vis), c2w, edges, first_px, near = best
if n_edges < a.min_pairs:
    raise SystemExit(f"no camera with >= {a.min_pairs} occluding pairs after {a.max_cam_tries} tries")
bproc.camera.add_camera_pose(c2w)
nodes = sorted(near)                           # every object some ray reaches
order, dropped = gt_order(nodes, edges, near)

# ── GT mesh export (before anything is hidden) ───────────────────────────────
def export_mesh(path_ply, path_inst):
    dg = bpy.context.evaluated_depsgraph_get()
    V, F, I, off = [], [], [], 0
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH" or ob.hide_render:
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        Mw = np.array(ob.matrix_world)
        co = co @ Mw[:3, :3].T + Mw[:3, 3]
        tri = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        ev.to_mesh_clear()
        V.append(co); F.append(tri.reshape(-1, 3) + off); off += len(co)
        I.append(np.full(len(tri) // 3, int(ob.get("inst_id", -1)), np.int32))
    V, F, I = np.vstack(V).astype(np.float32), np.vstack(F), np.concatenate(I)
    face = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face["n"], face["i"] = 3, F
    with open(path_ply, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        fh.write(V.astype("<f4").tobytes())
        fh.write(face.tobytes())
    np.save(path_inst, I)
    return len(V), len(F)


nv, nf = export_mesh(os.path.join(SDIR, "gt_mesh.ply"), os.path.join(SDIR, "face_inst.npy"))
c2w_cv = np.array(c2w) @ np.diag([1.0, -1.0, -1.0, 1.0])       # Blender/OpenGL -> OpenCV camera axes
np.savez(os.path.join(SDIR, "cam.npz"), K=K, width=W, height=H, c2w=c2w_cv)

# ── renders: input view, then one GT clean plate per removal ─────────────────
import imageio
bproc.renderer.set_max_amount_of_samples(a.samples)
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["inst_id"], default_values={"inst_id": -1})


def render_to(prefix):
    data = bproc.renderer.render()
    imageio.imwrite(prefix + "rgb.png", np.asarray(data["colors"][0])[..., :3].astype(np.uint8))
    np.save(prefix + "depth.npy", np.asarray(data["depth"][0], np.float32))
    np.save(prefix + "inst.npy", np.asarray(data["inst_id_segmaps"][0]).astype(np.int16))


render_to(os.path.join(SDIR, ""))
for step, iid in enumerate(order, 1):
    by_id[iid].hide(True)
    render_to(os.path.join(SDIR, "peel", f"step_{step:02d}_"))

meta = {"scene_id": a.scene_id, "seed": a.seed,
        "objects": [{"inst_id": int(o.get_cp("inst_id")), "source": o.get_cp("source"),
                     "visible_rays": int(first_px.get(o.get_cp("inst_id"), 0)),
                     "peel_step": (order.index(o.get_cp("inst_id")) + 1) if o.get_cp("inst_id") in order else None}
                    for o in objs],
        "gt_edges": [[int(x), int(y), int(c)] for (x, y), c in sorted(edges.items())],
        "gt_peel_order": [int(x) for x in order], "dropped_for_cycles": dropped,
        "ray_stride": a.ray_stride, "camera": {"hfov_deg": float(np.rad2deg(hfov)), "width": W, "height": H},
        "mesh": {"vertices": nv, "faces": nf}}
json.dump(meta, open(os.path.join(SDIR, "meta.json"), "w"), indent=2)
with open(os.path.join(a.out_dir, "scenes.txt"), "a") as fh:
    ab = lambda x: os.path.abspath(os.path.join(SDIR, x))
    fh.write(f"{a.scene_id} {ab('rgb.png')} {ab('gt_mesh.ply')} {ab('cam.npz')}\n")
print(f"[{a.scene_id}] {len(objs)} objects, {len(edges)} GT occlusion edges, "
      f"peel order {order}, {len(dropped)} cycle edge(s) dropped")
