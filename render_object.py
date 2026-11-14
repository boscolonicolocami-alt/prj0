"""Render synthetic RGB-D images of one object, with its ground-truth pose.

For each sample we pick a random camera viewpoint and a random HDRI (the 360°
photo that lights the scene and shows up in the background), render with Cycles
on the GPU, and save:

    sample_000000/rgb.png     color image, 8-bit
    sample_000000/depth.png   distance along the camera axis, uint16 millimetres, 0 = nothing
    sample_000000/mask.png    255 where the object is, 0 elsewhere
    sample_000000/meta.json   everything needed to use (and reproduce) the sample

Usage (from the repo root):
    uv run python render_object.py --obj-path ~/data/objs/usda/0000/objaverse-xxx.usda \
        --hdri-dir ~/data/hdris --out-dir ~/data/renders/test --num-images 10

Coordinate frames -- this is where pose bugs come from, so read it:
  * Object frame: the frame of the .usda file. We never move the object, so the
    object frame is also Blender's world frame (T_world_obj = identity), and we
    move the camera around it instead.
  * Camera frame, OpenCV convention: x right, y down, z forward (into the scene).
    FoundationPose and most computer-vision code use this one.
  * Blender's camera instead looks down its own -z axis, with y up.
    blender_to_opencv() converts between the two.
  The saved pose T_cam_obj takes a point from object coordinates to OpenCV camera
  coordinates:   p_cam = T_cam_obj @ [x, y, z, 1]
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import bpy  # Blender as a Python module
import numpy as np
import OpenEXR  # reads .exr images (float pixels, many layers in one file)
from mathutils import Matrix, Vector  # Blender's own small vector/matrix types
from PIL import Image

FX = 500.0  # focal length in pixels: ~51° field of view at 480 px, like a typical RGB-D camera
SAMPLES = 64  # Cycles samples per pixel: more = less noise, slower (the denoiser cleans up the rest)
OBJECT_PASS_INDEX = 1  # id written into the object-index pass for every part of our object
MAX_DEPTH_MM = np.iinfo(np.uint16).max  # 65535 mm = 65 m, the most a uint16 depth PNG can store


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Render RGB-D images of one object with its pose.")
    parser.add_argument("--obj-path", type=Path, required=True, help=".usda file of the object")
    parser.add_argument("--hdri-dir", type=Path, required=True, help="folder searched for .exr HDRIs")
    parser.add_argument("--out-dir", type=Path, required=True, help="where the sample_XXXXXX folders go")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="same seed -> same images")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="cpu is slower, but works when the shared GPU is full")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Scene setup (done once)
# ----------------------------------------------------------------------------

def enable_gpu():
    """Tell Cycles to render on the NVIDIA GPU (OptiX if available, else CUDA)."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("OPTIX", "CUDA"):
        prefs.compute_device_type = device_type
        prefs.get_devices()  # refreshes the device list for this type
        gpus = [d for d in prefs.devices if d.type == device_type]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == device_type  # use the GPU(s), not the CPU
            return device_type
    raise RuntimeError("No NVIDIA GPU found by Cycles")


def setup_scene(width, height, device):
    """Empty scene + renderer settings + camera + world. Returns the camera, world nodes, K."""
    bpy.ops.wm.read_factory_settings(use_empty=True)  # start from a completely empty scene
    scene = bpy.context.scene

    # Renderer: Cycles is Blender's path tracer (physically based light simulation).
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    if device == "gpu":
        scene.cycles.device = "GPU"
        print(f"rendering on GPU ({enable_gpu()})")
    else:
        scene.cycles.device = "CPU"
        print("rendering on CPU")
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    # "Standard" = plain sRGB conversion, no artistic tone mapping. That way the
    # conversion from the renderer's linear light values to an 8-bit image is a
    # known formula that we apply ourselves (linear_to_srgb below).
    scene.view_settings.view_transform = "Standard"

    # Extra outputs ("passes") besides the color image:
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True             # depth
    view_layer.use_pass_object_index = True  # which object is in each pixel -> mask

    # Write all passes into one multilayer EXR, with full float precision.
    scene.render.image_settings.media_type = "MULTI_LAYER_IMAGE"  # (Blender >= 5.0)
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_depth = "32"

    # Camera. Blender describes it like a physical camera (focal length in mm and
    # sensor width in mm); we choose them so the focal length in pixels is FX:
    #     f_px = f_mm * width_px / sensor_mm   ->   f_mm = f_px * sensor_mm / width_px
    cam_data = bpy.data.cameras.new("camera")
    cam_data.sensor_fit = "HORIZONTAL"  # sensor_width refers to the image width
    cam_data.sensor_width = 36.0
    cam_data.lens = FX * cam_data.sensor_width / width
    cam_data.clip_start = 0.01  # nothing closer than 1 cm is rendered
    cam_data.clip_end = 1000.0
    camera = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    # Intrinsics matrix K (OpenCV convention): pixel = K @ (p_cam / z).
    # Square pixels -> fy = fx. Principal point at the image center; OpenCV puts
    # pixel centers at integer coordinates, so the center is at (W-1)/2, (H-1)/2.
    K = np.array([[FX, 0.0, (width - 1) / 2],
                  [0.0, FX, (height - 1) / 2],
                  [0.0, 0.0, 1.0]])

    # World = the background and lighting, built as a small node graph:
    #   Texture Coordinate -> Mapping (rotation) -> Environment Texture (the HDRI)
    #   -> Background (strength) -> World Output
    world = bpy.data.worlds.new("world")
    scene.world = world
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    env = nodes.new("ShaderNodeTexEnvironment")
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(env.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    world_nodes = {"mapping": mapping, "env": env, "background": background}

    return camera, world_nodes, K


def load_object(obj_path):
    """Import the .usda and return its center and radius (in metres, object frame)."""
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(obj_path), import_cameras=False, import_lights=False)
    meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no meshes found in {obj_path}")

    # An object can be made of many parts: give them all the same index, so the
    # mask covers the whole object.
    for part in meshes:
        part.pass_index = OBJECT_PASS_INDEX

    # Bounding box of all parts: each part has 8 box corners in its own frame,
    # matrix_world moves them into world (= object) coordinates.
    corners = np.array([part.matrix_world @ Vector(c) for part in meshes for c in part.bound_box])
    low, high = corners.min(axis=0), corners.max(axis=0)
    center = (low + high) / 2
    radius = float(np.linalg.norm(high - low) / 2)  # radius of a sphere containing the box
    print(f"object: {len(meshes)} parts, size {np.round(high - low, 3)} m")
    return center, radius


# ----------------------------------------------------------------------------
# Random sampling (once per image)
# ----------------------------------------------------------------------------

def sample(rng, hdri_paths, center, radius, width):
    """Pick everything random about one image. Returns a dict of parameters."""
    # Viewing direction: uniform on the sphere. A 3D Gaussian sample points in a
    # uniformly random direction; normalizing it gives a unit vector.
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    # Distance: the object's bounding sphere should cover a random 40-90% of the
    # image width. tan(half_fov) = (W/2) / FX; a sphere of radius r at distance d
    # looks (r/d) / tan(half_fov) wide relative to the image.
    tan_half_fov = (width / 2) / FX
    fill = rng.uniform(0.4, 0.9)
    distance = radius / (fill * tan_half_fov)

    return {
        "camera_position": center + distance * direction,
        "look_at": center,
        "roll": rng.uniform(0, 2 * np.pi),  # rotation of the camera around its viewing axis
        "hdri_path": hdri_paths[rng.integers(len(hdri_paths))],
        "hdri_rotation": rng.uniform(0, 2 * np.pi),  # spin the environment around the vertical axis
        "hdri_strength": rng.uniform(0.6, 1.5),  # brightness of the lighting
    }


def look_at(camera_position, target, roll):
    """4x4 pose of a Blender camera at camera_position, looking at target.

    The columns of the rotation part are the camera's x, y, z axes expressed in
    world coordinates. A Blender camera looks along its -z, so z points backwards.
    """
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    z = -forward
    # Pick "up" as world Z, unless we are looking almost straight up/down (then
    # the cross product below would be ~zero), in which case use world Y.
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.column_stack([x, y, z])

    # Roll: rotate around the camera's own z axis (the viewing direction).
    c, s = np.cos(roll), np.sin(roll)
    rotation = rotation @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = camera_position
    return pose


def blender_to_opencv(T_world_cam_blender):
    """Same camera, OpenCV axes: flip the camera's y (up -> down) and z (back -> forward)."""
    return T_world_cam_blender @ np.diag([1.0, -1.0, -1.0, 1.0])


def apply(params, camera, world_nodes):
    """Put the sampled parameters into the Blender scene. Returns T_cam_obj (OpenCV)."""
    T_world_cam_blender = look_at(params["camera_position"], params["look_at"], params["roll"])
    camera.matrix_world = Matrix(T_world_cam_blender.tolist())

    env_image = bpy.data.images.load(str(params["hdri_path"]), check_existing=True)
    world_nodes["env"].image = env_image
    world_nodes["mapping"].inputs["Rotation"].default_value[2] = params["hdri_rotation"]
    world_nodes["background"].inputs["Strength"].default_value = params["hdri_strength"]

    # The object sits at the world origin with no rotation (T_world_obj = identity), so
    #   T_cam_obj = inverse(T_world_cam) @ T_world_obj = inverse(T_world_cam)
    T_world_cam = blender_to_opencv(T_world_cam_blender)
    return np.linalg.inv(T_world_cam)


# ----------------------------------------------------------------------------
# Rendering and saving
# ----------------------------------------------------------------------------

def linear_to_srgb(linear):
    """Renderer light values (linear) -> sRGB in [0, 1], the standard curve screens expect."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def render(tmp_dir):
    """Render the scene. Returns rgb (H,W,3 uint8), depth (H,W float metres), mask (H,W bool)."""
    exr_path = Path(tmp_dir) / "render.exr"
    bpy.context.scene.render.filepath = str(exr_path)
    bpy.ops.render.render(write_still=True)

    # Blender writes each pass as a separate "part" of the EXR file, and each part
    # has named channels, e.g. "ViewLayer.Combined.R". Collect them all in one dict.
    with OpenEXR.File(str(exr_path), separate_channels=True) as exr:
        channels = {name: ch.pixels for part in exr.parts for name, ch in part.channels.items()}
    rgb_linear = np.stack([channels[f"ViewLayer.Combined.{c}"] for c in "RGB"], axis=-1)
    depth = channels["ViewLayer.Depth.Z"].astype(np.float64)
    object_index = channels["ViewLayer.Object Index.X"]

    rgb = np.round(linear_to_srgb(rgb_linear) * 255).astype(np.uint8)
    mask = np.round(object_index) == OBJECT_PASS_INDEX
    # Pixels that see only the background get a huge depth (1e10): mark them as 0 = "no depth".
    depth[depth >= bpy.context.scene.camera.data.clip_end] = 0.0
    return rgb, depth, mask


def save_sample(sample_dir, rgb, depth, mask, meta):
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(sample_dir / "rgb.png")

    depth_mm = np.round(depth * 1000)
    if depth_mm.max() > MAX_DEPTH_MM:
        raise ValueError(f"depth up to {depth_mm.max() / 1000:.0f} m does not fit in uint16 mm: "
                         "object too large, pick a smaller one")
    Image.fromarray(depth_mm.astype(np.uint16)).save(sample_dir / "depth.png")  # 16-bit PNG

    Image.fromarray(mask.astype(np.uint8) * 255).save(sample_dir / "mask.png")
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    hdri_paths = sorted(args.hdri_dir.expanduser().rglob("*.exr"))  # sorted: same order every run
    if not hdri_paths:
        raise SystemExit(f"no .exr files found in {args.hdri_dir}")

    rng = np.random.default_rng(args.seed)  # every random choice comes from this generator
    camera, world_nodes, K = setup_scene(args.width, args.height, args.device)
    center, radius = load_object(args.obj_path.expanduser())

    with tempfile.TemporaryDirectory() as tmp_dir:  # deleted automatically at the end
        for i in range(args.num_images):
            start = time.time()
            params = sample(rng, hdri_paths, center, radius, args.width)
            T_cam_obj = apply(params, camera, world_nodes)
            rgb, depth, mask = render(tmp_dir)

            meta = {
                "obj_path": str(args.obj_path.expanduser().resolve()),
                "hdri_path": str(params["hdri_path"]),
                "hdri_rotation": params["hdri_rotation"],
                "hdri_strength": params["hdri_strength"],
                "scale": 1.0,
                "T_cam_obj": T_cam_obj.tolist(),  # numpy arrays -> nested lists for JSON
                "K": K.tolist(),
                "width": args.width,
                "height": args.height,
                "depth_unit": "mm",
                "seed": args.seed,
                "index": i,
            }
            save_sample(args.out_dir.expanduser() / f"sample_{i:06d}", rgb, depth, mask, meta)
            print(f"[{i + 1}/{args.num_images}] {time.time() - start:.1f}s  "
                  f"object covers {mask.mean():.0%} of the image")


if __name__ == "__main__":
    main()
