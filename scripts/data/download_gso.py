"""Download Google Scanned Objects (GSO) from Gazebo Fuel, CC-BY 4.0.

GSO is ~1k real scanned household objects with metric scale and textures, the
same domain as tabletop captures. Each model is unpacked to <out>/<name>/ with
meshes/model.obj and materials/textures/texture.png.

Usage: python scripts/data/download_gso.py --out data/gso [--n 300] [--seed 0]
"""
import argparse, io, json, os, random, shutil, urllib.error, urllib.request, zipfile

API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"


def list_models():
    names, page = [], 1
    while True:
        try:
            with urllib.request.urlopen(f"{API}?per_page=100&page={page}") as r:
                batch = json.load(r)
        except urllib.error.HTTPError as e:          # Fuel answers 404 past the last page
            if e.code == 404:
                return names
            raise
        if not batch:
            return names
        names += [m["name"] for m in batch]
        page += 1


def fix_texture(dst):
    """GSO's model.mtl says `map_Kd texture.png` relative to meshes/, but the image
    ships in materials/textures/; without a copy next to the .obj, Blender renders
    the missing-texture magenta."""
    src = os.path.join(dst, "materials", "textures", "texture.png")
    tgt = os.path.join(dst, "meshes", "texture.png")
    if os.path.exists(src) and not os.path.exists(tgt):
        shutil.copyfile(src, tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="random subset size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = sorted(list_models())
    print(f"{len(names)} GSO models listed")
    if a.n:
        names = random.Random(a.seed).sample(names, min(a.n, len(names)))
    for i, name in enumerate(names, 1):
        dst = os.path.join(a.out, name)
        if os.path.exists(os.path.join(dst, "meshes", "model.obj")):
            fix_texture(dst)
            continue
        try:
            with urllib.request.urlopen(f"{API}/{name}/tip/{name}.zip") as r:
                zipfile.ZipFile(io.BytesIO(r.read())).extractall(dst)
            fix_texture(dst)
            print(f"[{i}/{len(names)}] {name}")
        except Exception as e:                       # one bad model must not stop the batch
            print(f"[{i}/{len(names)}] {name} FAILED: {e}")


if __name__ == "__main__":
    main()
