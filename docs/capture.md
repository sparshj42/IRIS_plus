# Physically peeled capture protocol (3D-printed objects)

Real scenes with exact hidden geometry: every object is 3D-printed from a CAD
mesh, the scene is photographed, then objects are removed one at a time with the
cameras fixed. Each removal is photographed, so every peel step has a real
clean plate. Object poses are recovered from the photos themselves; no markers
are ever on the objects.

## Why no markers on objects

Any object face can end up in the main camera's view: whatever is frontmost is
photographed fully exposed, and a visible tag is an artificial cue that methods
would inpaint or reconstruct as texture. A tag on the back of a front object
faces into the cluster, where the other objects block the auxiliary cameras.
ArUco is used only on a ChArUco board, for calibration, and the board is removed
before the scene is built.

## How poses are recovered

1. Between step k and step k+1 only one object disappears, so the image
   difference in each camera is that object's visible silhouette
   (`session.py`). Changes elsewhere are reported as possible disturbance.
2. The object removed at step k was frontmost when it was removed. Its CAD is
   fitted to those masks in every camera (`posefit.py`): each stable resting pose
   of the CAD at 24 yaws, each shifted under the mask, the best refined over
   (x, y, yaw), then a 6-DoF refinement kept only if it improves the fit.
3. Objects are fitted in reverse removal order, so the objects behind (removed
   later) already have poses and occlude the rendered silhouette correctly.
4. The GT occlusion graph and peel order come from the fitted geometry
   (`finalize.py`). If the operator removed something that was not truly
   frontmost, the photos are still valid clean plates; `meta.json` records the
   pairs that were taken out of order.

On a simulated capture (`selftest_capture.py`), camera poses come out within
0.4° and 0.7% of range, and object poses within 0.6–1.6 mm ADD-S.

## Hardware

- **Main camera:** the stereo rig's left camera, fixed for the whole session.
  `calib.py import` can read its intrinsics from the rig's
  `stereo_calibration_params.xml`. The current file has large distortion terms
  (k3 ≈ −1.4), so recalibrating with `calib.py intrinsics` is recommended. Its
  ~24° field of view needs ~1.5–2 m of distance to frame a tabletop cluster.
- **Auxiliary cameras:** 2–3 phones on tripods, behind-left, behind-right and high
  front-side, each tilted down 30–60°. Avoid looking straight down: a head-on
  planar board gives the least accurate camera position (0.7% vs ≤0.2% tilted).
- **Phone settings:** lock focus, exposure and white balance (e.g. Open Camera),
  1× lens, no HDR or night mode, remote shutter.
- **Lighting:** large diffuse sources (softboxes, or a bright overcast room). Hard
  cast shadows change with each removal and enlarge the masks.
  `session.py --suppress_shadows` can remove some of them.
- **Board:** `board.py` makes the print. It must look big in every camera:
  markers at least ~25 px across. With the narrow main lens, print A2/A1, or use
  `--square_mm 60` and a 7×5 board.

## Objects

- Print in the orientation you like. Resting poses are computed from the CAD.
- Make them textured: multi-colour prints, paint or decals. Single-colour prints
  are textureless, a known failure case and a domain gap.
- `library.json`: `{"mug_01": {"mesh": "cad/mug_01.stl", "units": "mm"}, ...}`
  The mesh must be the printed geometry (same scale, no supports).
- Suggested set of ~30–40: printed Google Scanned Objects (the same objects as the
  synthetic set, for digital-twin pairs), simple primitives, three-piece
  interlocking shapes that occlude cyclically, containers (bowl and contents),
  and stackable pieces.

## Session

```
calib/                 intrinsics photos per camera (15-30 each, varied angles)
board_shot/<cam>.jpg   board flat on the table, one photo per camera, final positions
scene_001/
  steps/step_00/<cam>.jpg      all objects placed
  steps/step_01/<cam>.jpg      after removing the first object
  ...
  steps/step_NN/<cam>.jpg      empty table
  removed.txt                  library names in removal order, one per line
```

1. Mount every camera and lock its settings. Shoot `calib/<cam>/` (or import the
   stereo XML for the main camera).
2. Lay the board flat and shoot `board_shot/`. Remove the board. **Do not move any
   camera from here on**; re-shoot the board if one is bumped.
3. Build a scene with 4–9 objects that occlude each other from the main view.
4. Shoot `step_00` on every camera. Remove a fully visible object (lift it straight
   up without touching neighbours), write its name in `removed.txt`, shoot the next
   step. Repeat until the table is empty.
5. Process and check:

```bash
python scripts/capture/calib.py intrinsics --rig rig.json --board board.json --cam phoneA --images calib/phoneA
python scripts/capture/calib.py import     --rig rig.json --cam main --xml stereo_calibration_params.xml --side L --size W H
python scripts/capture/calib.py extrinsics --rig rig.json --board board.json --shot board_shot
python scripts/capture/session.py  --rig rig.json --scene scene_001
python scripts/capture/posefit.py  --rig rig.json --scene scene_001 --library library.json
python scripts/capture/finalize.py --rig rig.json --scene scene_001 --library library.json --out data/real --scene_id r001
```

6. Look at `data/real/r001/qa/step_XX.png`. Fitted CAD outlines are drawn on the
   real photos; outlines off the object mean a bad mask or fit. Also check the
   `disturbance` flags in `meta.json`.

## Benchmark slices to plan scenes for

- **Standard peels:** 4–9 objects, 2+ occlusion layers.
- **Designed hard cases:** genuine three-way cyclic occlusion, containment, stacks,
  thin parts.
- **Twin scenes:** an identical visible view with different hidden content. Shoot
  scene A at step 0, swap only the hidden object from behind (the disturbance
  check confirms nothing visible moved), shoot scene B, then peel B. Additions are
  not yet handled by `session.py`; this is the next extension.
- **Digital twins:** re-render any finalized scene from its meshes and poses with
  `scripts/data` for paired real/synthetic evaluation.
