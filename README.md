# prj0

Synthetic data generation for 6D object pose estimation, as the first step of
extending [FoundationPose](https://github.com/NVlabs/FoundationPose) with the pose
of a robot hand as an extra input.

Objects come from [sand3d-objaverse](https://huggingface.co/datasets/Sandflow/sand3d-objaverse),
lighting from [open-hdri-1k](https://huggingface.co/datasets/leodriesch/open-hdri-1k).
Rendering uses Blender as a Python module (`bpy`), visualization uses [viser](https://viser.studio).

## Setup

```sh
uv sync          # creates .venv with Python 3.13 and all dependencies
```

## Data layout (outside the repo)

```
~/data/
  objs/       downloaded .usda objects
  hdris/      downloaded .exr environment maps
  renders/<run>/sample_000000/
    rgb.png     color image
    depth.png   uint16, millimetres
    mask.png    255 = object
    meta.json   object/hdri paths, K, T_cam_obj (OpenCV convention), seed, ...
```
