# prj0 — notes for Claude

## Who you are working with
Nicolò is a 2nd-year CS bachelor student, **new to Python** (knows other languages) and
new to git. This project is for learning and for the CV: the goal is that Nicolò
understands every line, not just that the code works. Nicolò's brother (PhD in computer
vision) supervises the research direction.

- **Priority: Nicolò understands everything, more than writing it by hand.** Claude may
  write the code, but every change comes with an explanation of what it does and why,
  and of the choices made (and the alternatives rejected). Relate Python concepts to
  C/Java when useful. Check understanding; answer "why" questions in depth.
- Code is written to be read: clear names, small functions, comments that explain the
  reasoning (not the syntax). Prefer simple over clever.
- Keep changes small and reviewable. Propose a plan before anything non-trivial.
- Every feature comes with a way to verify it (a check script, a picture, a number).
- Nicolò commits and pushes. Do not run `git commit` / `git push` unless asked.
- Comment `bpy` code generously: Blender's API is not obvious.

## Goal
Extend [FoundationPose](https://github.com/NVlabs/FoundationPose) (6D pose of an object
from RGB-D + its 3D model) with the **pose of a robot hand as an extra input**, to
improve precision. Step 1, in progress: a synthetic dataset rendered with Blender.

## Environment
- Runs on the workstation `chioggia` (Ubuntu 24.04, RTX 3090 24GB, shared with others:
  check `nvidia-smi` before long GPU jobs). Disk is ~94% full: never download whole datasets.
- `uv` project, Python 3.13 (required by `bpy` 5.2). Run things with `uv run python ...`;
  add dependencies with `uv add <pkg>`, never `pip install`.
- `bpy` is Blender as a Python module: headless, Cycles + OPTIX on the GPU works.
- Visualization with `viser` (web UI on port 8080; from the laptop:
  `ssh -L 8080:localhost:8080 chioggia`, then open http://localhost:8080).

## Data (outside git, in ~/data)
- `~/data/objs/` objects from HF `Sandflow/sand3d-objaverse`: `.usda`, **metres, Z up**.
  Shapes were rebuilt by an agent from Objaverse (approximate, simple materials).
  182 GB in total: download subsets only, with `tools/download_data.py`.
  `~/data/objs/index.json` describes what was downloaded.
- `~/data/hdris/` from HF `leodriesch/open-hdri-1k` (CC0, 1k `.exr`).
- `~/data/renders/<run>/sample_XXXXXX/` outputs.

## render_object.py design (agreed)
```
render_object.py --obj-path --hdri-dir --out-dir --num-images 10 --seed 42 --width 480 --height 480

rng = np.random.default_rng(seed)        # every random choice comes from this rng
setup_scene(width, height)               # once: Cycles/GPU, camera, intrinsics K
obj = load_object(obj_path)              # once: import .usda
for i in range(num_images):
    params = sample(rng, hdri_paths)     # hdri path/rotation/strength, T_cam_obj
    apply(params, obj)
    rgb, depth, mask = render()          # numpy arrays
    save_sample(out_dir / f"sample_{i:06d}", rgb, depth, mask, meta)
```
Per sample: `rgb.png`, `depth.png` (uint16, **millimetres**, BOP convention), `mask.png`
(255 = object), `meta.json` with obj_path, hdri_path, hdri_rotation, hdri_strength,
scale (keep 1.0), `T_cam_obj` (4x4), `K` (3x3), width, height, depth_unit, seed.
Later: `T_cam_hand`.

**Conventions (most likely source of bugs):** poses are stored in the **OpenCV camera
convention** (x right, y down, z forward), which FoundationPose uses. Blender cameras look
down -Z with Y up: convert explicitly, in one well-tested function. Verify every change
by projecting the object into the image or showing it in viser.

## Implementation notes
- The object never moves (object frame = world frame); the **camera** is moved around it.
  Same result as moving the object, simpler maths.
- One render writes a multilayer EXR (color + depth + object-index passes), read with
  the `OpenEXR` package. RGB uses the "Standard" view transform so linear→sRGB is a plain
  formula done in numpy.
  Blender 5.x quirks: set `image_settings.media_type = "MULTI_LAYER_IMAGE"` before
  `file_format = "OPEN_EXR_MULTILAYER"`; the EXR is multi-part (one part per pass), the
  index channel is `ViewLayer.Object Index.X`; `World.use_nodes` is deprecated (not needed).
- `tools/check_sample.py` verifies a sample: projects the mesh with K and T_cam_obj and
  checks it lands on the mask and agrees with the depth (1-px tolerance); writes
  `overlay.png`. Run it after any change to the renderer. Verified 2026-11-14: 100%/100%
  on 4 samples; principal point (W-1)/2 fits better than W/2; un-flipping the
  Blender→OpenCV conversion makes it FAIL (so it does catch that bug).
- The GPU is shared and is sometimes completely full: `--device cpu` works (~2 s/image
  at 480x480, 64 samples, 24 cores).

## Known facts about the data (checked 2026-11-14)
- The `.usda` files import fine with `bpy.ops.wm.usd_import`.
- Objects are at **real-world size**, from a 5 mm screw to a 54 m aircraft and 10 m
  buildings. For a robot hand we need graspable objects: filtering by `family` and by
  size (e.g. 3-30 cm) is an open decision for Nicolò and the brother.
- An object can be made of many mesh parts (one building had 2177): join them, or treat
  them as one object, for the mask and the pose.
- Materials are flat colours, no image textures: a domain gap with real photos.
