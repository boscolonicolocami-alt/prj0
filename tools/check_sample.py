"""Check that a rendered sample is geometrically correct.

The idea: take the object's 3D mesh, move it into the camera frame with the saved
pose T_cam_obj, project it onto the image with the saved intrinsics K, and see if
it lands where the renderer drew the object. If any convention is wrong (Blender
vs OpenCV axes, a transposed matrix, metres vs millimetres...) the points miss.

Two numbers are checked:
  * on_mask:  fraction of projected vertices that fall inside the object mask.
  * depth_ok: fraction whose distance z is >= the rendered depth at that pixel
    (a vertex can be hidden behind the front surface, so deeper is fine, but it
    can never be in front of what the renderer saw).
Vertices on the object's outline can round to the pixel just next to it, so
both checks accept a 1-pixel neighbourhood. Both should then be ~100%. It also writes overlay.png: the projected vertices drawn
on rgb.png, for your eyes.

Usage:
    uv run python tools/check_sample.py ~/data/renders/test/sample_000000
"""

import argparse
import json
from pathlib import Path

import bpy
import numpy as np
from PIL import Image, ImageDraw

MAX_POINTS = 3000      # vertices drawn/checked (a random subset, for speed)
DEPTH_TOL_M = 0.002    # 2 mm tolerance on depth (rounding to mm + mesh details)


def neighbourhood(image, reduce):
    """For each pixel, combine (max or min) the values of its 3x3 neighbourhood.

    np.pad adds a 1-pixel border so the shifted copies keep the same size.
    """
    padded = np.pad(image, 1, mode="edge")
    H, W = image.shape
    shifted = [padded[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)]
    return reduce(np.stack(shifted), axis=0)


def mesh_vertices_in_object_frame(obj_path):
    """All vertices of the .usda, as an (N, 3) array in object coordinates (metres)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(obj_path), import_cameras=False, import_lights=False)
    # "Evaluated" = after modifiers (e.g. subdivision), i.e. the geometry that was rendered.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    all_points = []
    for part in bpy.context.scene.objects:
        if part.type != "MESH":
            continue
        mesh = part.evaluated_get(depsgraph).to_mesh()
        coords = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", coords)  # fast bulk copy of all coordinates
        points = coords.reshape(-1, 3)
        # part frame -> world (= object) frame, with homogeneous coordinates [x y z 1]
        M = np.array(part.matrix_world)
        points = (M[:3, :3] @ points.T).T + M[:3, 3]
        all_points.append(points)
    return np.concatenate(all_points)


def main():
    parser = argparse.ArgumentParser(description="Verify pose, K, mask and depth of a sample.")
    parser.add_argument("sample_dir", type=Path)
    args = parser.parse_args()
    sample_dir = args.sample_dir.expanduser()

    meta = json.loads((sample_dir / "meta.json").read_text())
    K = np.array(meta["K"])
    T_cam_obj = np.array(meta["T_cam_obj"])
    rgb = Image.open(sample_dir / "rgb.png").convert("RGB")
    mask = np.array(Image.open(sample_dir / "mask.png")) > 0
    depth_m = np.array(Image.open(sample_dir / "depth.png")).astype(np.float64) / 1000

    points_obj = mesh_vertices_in_object_frame(meta["obj_path"])
    rng = np.random.default_rng(0)
    if len(points_obj) > MAX_POINTS:
        points_obj = points_obj[rng.choice(len(points_obj), MAX_POINTS, replace=False)]

    # object frame -> camera frame (OpenCV): p_cam = R @ p_obj + t
    points_cam = (T_cam_obj[:3, :3] @ points_obj.T).T + T_cam_obj[:3, 3]
    z = points_cam[:, 2]
    # camera frame -> pixels: [u v 1] = K @ (p_cam / z)
    uv = (K @ (points_cam / z[:, None]).T).T[:, :2]
    cols, rows = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)

    H, W = mask.shape
    inside = (z > 0) & (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    if not inside.any():
        print("FAIL: no vertex lands in the image (object behind the camera? wrong axes?)")
        return
    # "Is there object in this pixel or a neighbour?"  (max of booleans = any)
    mask_near = neighbourhood(mask, np.max)
    on_mask = mask_near[rows[inside], cols[inside]]
    # "Closest surface the renderer saw around this pixel" (0 = background -> ignore)
    depth_near = neighbourhood(np.where(depth_m > 0, depth_m, np.inf), np.min)
    rendered = depth_near[rows[inside], cols[inside]]
    depth_ok = z[inside] >= rendered - DEPTH_TOL_M

    print(f"vertices checked: {len(points_obj)}, inside the image: {inside.mean():.1%}")
    print(f"on_mask:  {on_mask.mean():.1%}")
    print(f"depth_ok: {depth_ok.mean():.1%}")
    good = inside.mean() > 0.95 and on_mask.mean() > 0.95 and depth_ok.mean() > 0.95
    print("PASS" if good else "FAIL: pose, K or a convention is wrong somewhere")

    overlay = rgb.copy()
    draw = ImageDraw.Draw(overlay)
    for u, v, ok in zip(uv[inside, 0], uv[inside, 1], on_mask):
        draw.point((u, v), fill=(0, 255, 0) if ok else (255, 0, 0))  # green = on mask
    overlay.save(sample_dir / "overlay.png")
    print(f"wrote {sample_dir / 'overlay.png'}")


if __name__ == "__main__":
    main()
