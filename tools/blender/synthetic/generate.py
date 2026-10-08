from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Quaternion, Vector
from bpy_extras.object_utils import world_to_camera_view
from PIL import Image, ImageChops

# Blender --python need not add the script directory to sys.path.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from layout_xy_optimizer import XYGuideOptimizer

COMPOSITE_SIZE = (320, 320)
PADDING_RGB = (0, 0, 0)
REGIONS = {
    "completed_hand": {"aspect": (17, 4), "dest": (7, 0, 306, 72)},
    "dora_indicators": {"aspect": (17, 4), "dest": (7, 74, 306, 72)},
    "melds": {"aspect": (1, 1), "dest": (74, 148, 172, 172)},
}
CATEGORY = {"id": 1, "name": "mahjong_tile", "supercategory": "mahjong_tile"}
SUITS = "mps"
HONORS = ("east", "south", "west", "north", "white", "green", "red")
ALL_TILES = tuple([f"{n}{s}" for s in SUITS for n in range(1, 10)] + list(HONORS) + ["red5m", "red5p", "red5s"])
ASSET_NAMES = {
    **{f"{n}m": f"Man{n}" for n in range(1, 10)},
    **{f"{n}p": f"Pin{n}" for n in range(1, 10)},
    **{f"{n}s": f"Sou{n}" for n in range(1, 10)},
    "east": "Ton", "south": "Nan", "west": "Shaa", "north": "Pei",
    "white": "Haku", "green": "Hatsu", "red": "Chun",
    "red5m": "Man5-Dora", "red5p": "Pin5-Dora", "red5s": "Sou5-Dora",
}
# Japanese automatic-table scale: ~26-27 x 36-38 x 19 mm tiles on an
# ~815 mm square felt field.  The camera intentionally sees only the player's
# near-side working area, never the whole table.
W, H, T = 0.027, 0.038, 0.019
BACK_T = 0.004
TABLE_X = TABLE_Y = 0.815
OUTER_TABLE_X = OUTER_TABLE_Y = 0.960
INNER_X = INNER_Y = 0.785


def script_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.production.json"))
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / ".outputs/default")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--skip-assemble", action="store_true")
    p.add_argument("--assemble-only", action="store_true")
    p.add_argument("--seed", type=int, default=760601)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-full-views", action="store_true")
    return p.parse_args(script_argv())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capacity(tile: str) -> int:
    if tile in {"5m", "5p", "5s"}:
        return 3
    if tile in {"red5m", "red5p", "red5s"}:
        return 1
    return 4


def can_allocate(used: Counter[str], tiles: list[str] | tuple[str, ...]) -> bool:
    req = Counter(tiles)
    return all(used[t] + n <= capacity(t) for t, n in req.items())


def allocate(used: Counter[str], tiles: list[str] | tuple[str, ...]) -> None:
    if not can_allocate(used, tiles):
        raise ValueError(f"illegal inventory allocation: {tiles} against {dict(used)}")
    used.update(tiles)


def sequence_candidates(rng: random.Random) -> list[list[str]]:
    out: list[list[str]] = []
    for suit in SUITS:
        for start in range(1, 8):
            base = [f"{n}{suit}" for n in range(start, start + 3)]
            out.append(base)
            if start <= 5 <= start + 2:
                red = [f"red5{suit}" if n == 5 else f"{n}{suit}" for n in range(start, start + 3)]
                out.append(red)
    rng.shuffle(out)
    return out


def repeated_candidates(count: int, rng: random.Random) -> list[list[str]]:
    out = [[t] * count for t in ALL_TILES if capacity(t) >= count]
    rng.shuffle(out)
    return out


def pick_component(kind: str, used: Counter[str], rng: random.Random, force_tile: str | None = None) -> list[str]:
    if force_tile is not None:
        count = 3 if kind == "pon" else 4
        tiles = [force_tile] * count
        if kind == "chi" or not can_allocate(used, tiles):
            raise ValueError("forced component is unavailable")
        return tiles
    candidates = sequence_candidates(rng) if kind == "chi" else repeated_candidates(3 if kind == "pon" else 4, rng)
    for tiles in candidates:
        if can_allocate(used, tiles):
            return tiles
    raise RuntimeError(f"no legal component for {kind}")


def pick_concealed_component(used: Counter[str], rng: random.Random) -> list[str]:
    candidates = sequence_candidates(rng) + repeated_candidates(3, rng)
    rng.shuffle(candidates)
    for tiles in candidates:
        if can_allocate(used, tiles):
            return tiles
    raise RuntimeError("no legal concealed component")


def pick_pair(used: Counter[str], rng: random.Random) -> list[str]:
    candidates = list(ALL_TILES)
    rng.shuffle(candidates)
    for tile in candidates:
        pair = [tile, tile]
        if can_allocate(used, pair):
            return pair
    raise RuntimeError("no legal pair")


def tile_sort_key(tile: str) -> tuple[int, int, int]:
    if tile.endswith("m"):
        return (0, 5 if tile == "red5m" else int(tile[0]), 0 if tile.startswith("red") else 1)
    if tile.endswith("p"):
        return (1, 5 if tile == "red5p" else int(tile[0]), 0 if tile.startswith("red") else 1)
    if tile.endswith("s"):
        return (2, 5 if tile == "red5s" else int(tile[0]), 0 if tile.startswith("red") else 1)
    return (3, HONORS.index(tile), 0)


def build_hand(index: int, rng: random.Random, meld_count: int, force_white_meld: bool,
               completed_hand_empty: bool, dora_indicators_empty: bool) -> dict[str, Any]:
    used: Counter[str] = Counter()
    kinds = rng.choices(["chi", "pon", "open-kan", "closed-kan"], weights=[0.34, 0.32, 0.19, 0.15], k=meld_count)
    if force_white_meld and meld_count:
        eligible = [i for i, k in enumerate(kinds) if k != "chi"]
        if not eligible:
            kinds[rng.randrange(meld_count)] = "pon"
            eligible = [i for i, k in enumerate(kinds) if k != "chi"]
        white_i = rng.choice(eligible)
    else:
        white_i = -1
    melds: list[dict[str, Any]] = []
    for mi, kind in enumerate(kinds):
        tiles = pick_component(kind, used, rng, "white" if mi == white_i else None)
        allocate(used, tiles)
        called = None if kind == "closed-kan" else rng.randrange(len(tiles))
        slots = []
        for ti, tile in enumerate(tiles):
            face = "back" if kind == "closed-kan" and ti in {0, len(tiles) - 1} else "front"
            rotation = 90 if called == ti else 0
            slots.append({"tile": tile, "face": face, "rotation": rotation})
        melds.append({"kind": kind, "tiles": slots})
    concealed: list[str] = []
    if not completed_hand_empty:
        # A nonempty completed hand is a standard winning hand: the exposed
        # groups plus (4 - meld_count) concealed groups and one pair.
        # Kans contain 4 physical tiles but count as one of the four groups.
        for _ in range(4 - meld_count):
            component = pick_concealed_component(used, rng)
            allocate(used, component)
            concealed.extend(component)
        pair = pick_pair(used, rng)
        allocate(used, pair)
        concealed.extend(pair)
        if len(concealed) != 14 - 3 * meld_count:
            raise AssertionError("completed hand must contain 4 groups and 1 pair")
        concealed.sort(key=tile_sort_key)

    indicator_count = 0 if dora_indicators_empty else rng.choices(
        [1, 2, 3, 4, 5], weights=[0.46, 0.26, 0.16, 0.08, 0.04], k=1
    )[0]
    indicators: list[str] = []
    if indicator_count:
        coverage_tile = ALL_TILES[index % len(ALL_TILES)]
        if can_allocate(used, [coverage_tile]):
            indicators.append(coverage_tile)
            allocate(used, [coverage_tile])
        candidates = list(ALL_TILES)
        rng.shuffle(candidates)
        for tile in candidates:
            if len(indicators) >= indicator_count:
                break
            if can_allocate(used, [tile]):
                indicators.append(tile)
                allocate(used, [tile])
        if len(indicators) != indicator_count:
            raise RuntimeError("could not allocate dora indicators")
    for tile, n in used.items():
        if n > capacity(tile):
            raise AssertionError((tile, n))
    return {
        "concealed": concealed,
        "dora": indicators,
        "melds": melds,
        "inventory": dict(used),
        "force_white_meld": force_white_meld,
        "completed_hand_empty": completed_hand_empty,
        "winning_shape_complete": not completed_hand_empty,
        "dora_indicators_empty": dora_indicators_empty,
    }


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)


def rounded_mesh(name: str, dims: tuple[float, float, float], bevel: float, segments: int = 4):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mod = obj.modifiers.new("edge_rounding", "BEVEL")
    mod.width = bevel
    mod.segments = segments
    mod.limit_method = "ANGLE"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    mesh = obj.data
    mesh.name = f"{name}_mesh"
    bpy.data.objects.remove(obj, do_unlink=True)
    return mesh


def plane_mesh(name: str, width: float, height: float):
    mesh = bpy.data.meshes.new(name)
    verts = [(-width/2, -height/2, 0), (width/2, -height/2, 0), (width/2, height/2, 0), (-width/2, height/2, 0)]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            uv.data[li].uv = coords[mesh.loops[li].vertex_index]
    return mesh


def principled_material(name: str, color: tuple[float, float, float, float], roughness: float):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    if "IOR Level" in bsdf.inputs:
        bsdf.inputs["IOR Level"].default_value = 0.34
    return mat


def felt_material() -> Any:
    base = (0.08, 0.34, 0.19, 1)
    mat = principled_material("felt", base, 0.72)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "FeltNoise"
    noise.inputs["Scale"].default_value = 260.0
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.75
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = "FeltGrainRamp"
    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = (0.78, 0.78, 0.78, 1)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (1.08, 1.08, 1.08, 1)
    mix = nodes.new("ShaderNodeMixRGB")
    mix.name = "FeltBaseMix"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    mix.inputs[1].default_value = base
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs[2])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.0007
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def transparent_decal_material(name: str, image_path: Path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path), check_existing=True)
    tex.interpolation = "Linear"
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.36
    try:
        mat.surface_render_method = "DITHERED"
    except Exception:
        pass
    return mat


def ensure_decals(asset_dir: Path, decal_dir: Path) -> dict[str, Path]:
    # Upstream Export/Regular PNGs are already transparent glyph-only artwork.
    # Front/Back are separate full-face layers; do not bake them onto the 3D tile face.
    result: dict[str, Path] = {}
    for tile, stem in ASSET_NAMES.items():
        source = asset_dir / f"{stem}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        result[tile] = source
    return result


def add_mesh_object(name: str, mesh, collection, material=None):
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if material is not None and material.name not in {m.name for m in obj.data.materials if m is not None}:
        obj.data.materials.append(material)
    return obj


def setup_world(config: dict[str, Any], decals: dict[str, Path]) -> dict[str, Any]:
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = int(config["render"]["width"])
    scene.render.resolution_y = int(config["render"]["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 25
    scene.render.film_transparent = False
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    static = bpy.data.collections.new("STATIC")
    dynamic = bpy.data.collections.new("DYNAMIC")
    scene.collection.children.link(static)
    scene.collection.children.link(dynamic)

    meshes = {
        "tile_body": rounded_mesh("tile_body_template", (W, H, T - BACK_T), 0.0026, 5),
        "tile_back": rounded_mesh("tile_back_template", (W * 0.985, H * 0.985, BACK_T), 0.0018, 4),
        "decals": {tile: plane_mesh(f"decal_template_{tile}", W * 0.84, H * 0.84) for tile in decals},
        "table": rounded_mesh("table_template", (TABLE_X, TABLE_Y, 0.018), 0.012, 5),
        "frame_h": rounded_mesh("frame_h_template", (TABLE_X + 0.055, 0.035, 0.033), 0.008, 4),
        "frame_v": rounded_mesh("frame_v_template", (0.035, TABLE_Y + 0.055, 0.033), 0.008, 4),
        "wood_h": rounded_mesh("wood_h_template", (OUTER_TABLE_X, 0.024, 0.018), 0.006, 3),
        "wood_v": rounded_mesh("wood_v_template", (0.024, OUTER_TABLE_Y, 0.018), 0.006, 3),
        "tray": rounded_mesh("tray_template", (0.118, 0.022, 0.008), 0.004, 3),
        "blocker": rounded_mesh("blocker_template", (0.30, 0.018, 0.018), 0.003, 3),
    }
    materials = {
        "felt": felt_material(),
        "body": principled_material("tile_ivory", (0.92, 0.89, 0.82, 1), 0.26),
        "back": principled_material("tile_back_amber", (0.94, 0.48, 0.055, 1), 0.30),
        "frame": principled_material("frame_black", (0.018, 0.021, 0.019, 1), 0.42),
        "wood": principled_material("wood_edge", (0.45, 0.20, 0.065, 1), 0.38),
        "tray": principled_material("tray_dark", (0.008, 0.010, 0.009, 1), 0.52),
        "blocker": principled_material("shadow_blocker", (0.015, 0.015, 0.015, 1), 0.50),
    }
    materials["decals"] = {tile: transparent_decal_material(f"decal_{tile}", path) for tile, path in decals.items()}

    table = add_mesh_object("felt_table", meshes["table"], static, materials["felt"])
    table.location.z = -0.012
    for y in (-TABLE_Y/2 - 0.016, TABLE_Y/2 + 0.016):
        bar = add_mesh_object(f"frame_h_{y:+.3f}", meshes["frame_h"], static, materials["frame"])
        bar.location = (0, y, 0.003)
    for x in (-TABLE_X/2 - 0.016, TABLE_X/2 + 0.016):
        bar = add_mesh_object(f"frame_v_{x:+.3f}", meshes["frame_v"], static, materials["frame"])
        bar.location = (x, 0, 0.003)
    outer_y = OUTER_TABLE_Y / 2 - 0.012
    outer_x = OUTER_TABLE_X / 2 - 0.012
    for y in (-outer_y, outer_y):
        edge = add_mesh_object(f"wood_h_{y:+.3f}", meshes["wood_h"], static, materials["wood"])
        edge.location = (0, y, -0.012)
    for x in (-outer_x, outer_x):
        edge = add_mesh_object(f"wood_v_{x:+.3f}", meshes["wood_v"], static, materials["wood"])
        edge.location = (x, 0, -0.012)
    # Only near-edge details are needed because the render is deliberately
    # zoomed into one player's sector rather than framing the whole table.
    for i, x in enumerate((-0.28, -0.14, 0.00, 0.14, 0.28)):
        tray = add_mesh_object(f"tray_{i}", meshes["tray"], static, materials["tray"])
        tray.location = (x, -TABLE_Y/2 - 0.016, 0.019)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    dynamic.objects.link(camera)
    scene.camera = camera
    return {"scene": scene, "static": static, "dynamic": dynamic, "meshes": meshes, "materials": materials, "camera": camera}


def clear_dynamic(ctx: dict[str, Any]) -> None:
    dynamic = ctx["dynamic"]
    camera = ctx["camera"]
    for obj in list(dynamic.objects):
        if obj == camera:
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if isinstance(data, bpy.types.Light) and data.users == 0:
            bpy.data.lights.remove(data)
        elif (
            isinstance(data, bpy.types.Mesh)
            and data.users == 0
            and data.name.startswith("blocker_mesh_")
        ):
            bpy.data.meshes.remove(data)
    for material in list(bpy.data.materials):
        if material.name.startswith("blocker_mat_") and material.users == 0:
            bpy.data.materials.remove(material)


def obb_corners(x: float, y: float, yaw: float, width: float = W, height: float = H) -> list[tuple[float, float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in ((-width/2, -height/2), (width/2, -height/2), (width/2, height/2), (-width/2, height/2)):
        out.append((x + dx*c - dy*s, y + dx*s + dy*c))
    return out


def project_interval(poly: list[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
    vals = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
    return min(vals), max(vals)


def polygons_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]], margin: float = 0.0004) -> bool:
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % 4]
            edge = (x2-x1, y2-y1)
            axis = (-edge[1], edge[0])
            n = math.hypot(*axis)
            axis = (axis[0]/n, axis[1]/n)
            amin, amax = project_interval(a, axis)
            bmin, bmax = project_interval(b, axis)
            if amax + margin <= bmin or bmax + margin <= amin:
                return False
    return True


def create_tile(ctx: dict[str, Any], tile_id: str, identity: str, face: str, region: str, group: str,
                meld_type: str | None, x: float, y: float, yaw: float, rng: random.Random) -> dict[str, Any]:
    dynamic, meshes, materials = ctx["dynamic"], ctx["meshes"], ctx["materials"]
    root = bpy.data.objects.new(tile_id, None)
    dynamic.objects.link(root)
    root.location = (x, y, T/2)
    root.rotation_euler = (math.pi if face == "back" else 0.0, 0.0, yaw)

    body = add_mesh_object(tile_id + "_body", meshes["tile_body"], dynamic, materials["body"])
    body.parent = root
    body.location = (0, 0, BACK_T / 2)
    back = add_mesh_object(tile_id + "_back", meshes["tile_back"], dynamic, materials["back"])
    back.parent = root
    back.location = (0, 0, -T / 2 + BACK_T / 2)

    children = [body, back]
    if face == "front":
        decal = add_mesh_object(tile_id + "_decal", meshes["decals"][identity], dynamic, materials["decals"][identity])
        decal.parent = root
        decal.location = (0, 0, T/2 + 0.00035)
        children.append(decal)
    return {
        "id": tile_id, "identity": identity, "face": face, "region": region, "group": group,
        "meld_type": meld_type, "root": root, "children": children, "x": x, "y": y, "yaw_deg": math.degrees(yaw),
        "visibility": 1.0,
    }


def linear_poses(slots: list[dict[str, Any]], center: tuple[float, float], base_angle: float,
                 rng: random.Random, regime: str, separated_last: bool = False,
                 align_right: float | None = None, align_bottom: float | None = None) -> list[tuple[float, float, float]]:
    p = {"neat": (0.0015, 0.0028, 0.7, 0.7), "ordinary": (0.001, 0.0055, 1.8, 1.8), "messy": (0.0007, 0.008, 3.6, 3.2)}[regime]
    gap_lo, gap_hi, yaw_sigma, y_sigma_mm = p
    extents = [H/2 if abs(int(s.get("rotation", 0))) == 90 else W/2 for s in slots]
    gaps = []
    state = rng.uniform(gap_lo, gap_hi)
    for i in range(max(0, len(slots)-1)):
        state = max(gap_lo, min(gap_hi, 0.64*state + 0.36*rng.uniform(gap_lo, gap_hi)))
        gaps.append(state)
    if separated_last and gaps:
        # A separated winning/drawn tile exists in real captures, but keep it
        # within the fixed mjtensu completed-hand guide instead of inventing an
        # exaggerated gap.
        gaps[-1] += rng.uniform(0.005, 0.013)
    xs = [0.0]
    for i in range(1, len(slots)):
        xs.append(xs[-1] + extents[i-1] + extents[i] + gaps[i-1])
    mid = (xs[0] + xs[-1]) / 2 if xs else 0
    xs = [v-mid for v in xs]
    poses = []
    y_state = rng.uniform(-0.001, 0.001)
    yaw_state = rng.uniform(-yaw_sigma, yaw_sigma)
    curvature = rng.uniform(-0.010, 0.010) if regime == "messy" else rng.uniform(-0.004, 0.004)
    outlier = rng.randrange(len(slots)) if slots and rng.random() < (0.04 if regime == "neat" else 0.11) else -1
    c, s = math.cos(base_angle), math.sin(base_angle)
    span = max(abs(xs[0]), abs(xs[-1]), 0.02) if xs else 0.02
    for i, (slot, lx) in enumerate(zip(slots, xs)):
        y_state = 0.62*y_state + rng.gauss(0, y_sigma_mm/1000)
        local_y = y_state + curvature*((lx/span)**2 - 0.33)
        yaw_state = 0.68*yaw_state + rng.gauss(0, yaw_sigma*0.55)
        local_yaw = math.radians(yaw_state)
        if i == outlier:
            local_y += rng.choice([-1, 1]) * rng.uniform(0.004, 0.009)
            local_yaw += math.radians(rng.uniform(-5.5, 5.5))
        wx = center[0] + lx*c - local_y*s
        wy = center[1] + lx*s + local_y*c
        yaw = base_angle + local_yaw + math.radians(float(slot.get("rotation", 0)))
        poses.append((wx, wy, yaw))
    if poses and (align_right is not None or align_bottom is not None):
        polys = [obb_corners(x, y, yaw) for x, y, yaw in poses]
        dx = 0.0 if align_right is None else align_right - max(px for poly in polys for px, _ in poly)
        dy = 0.0 if align_bottom is None else align_bottom - min(py for poly in polys for _, py in poly)
        poses = [(x + dx, y + dy, yaw) for x, y, yaw in poses]
    return poses


def world_xy_for_pixel(scene, camera, px: float, py: float, guess: tuple[float, float]) -> tuple[float, float]:
    x, y = guess
    eps = 0.001
    for _ in range(10):
        def uv(wx: float, wy: float) -> tuple[float, float]:
            ndc = world_to_camera_view(scene, camera, Vector((wx, wy, 0.0)))
            return ndc.x * scene.render.resolution_x, (1.0 - ndc.y) * scene.render.resolution_y
        u, v = uv(x, y)
        ex, ey = px - u, py - v
        if abs(ex) + abs(ey) < 0.05:
            break
        ux, vx = uv(x + eps, y)
        uy, vy = uv(x, y + eps)
        j00, j10 = (ux - u) / eps, (vx - v) / eps
        j01, j11 = (uy - u) / eps, (vy - v) / eps
        det = j00 * j11 - j01 * j10
        if abs(det) < 1e-9:
            raise ValueError("camera projection inversion is singular")
        x += (ex * j11 - j01 * ey) / det
        y += (j00 * ey - ex * j10) / det
    return x, y


def black_frame_inner_faces(ctx: dict[str, Any]) -> tuple[float, float]:
    """Get actual inner faces from instantiated bottom/right black-frame meshes."""
    right = next(o for o in ctx["static"].objects if o.name.startswith("frame_v_+"))
    bottom = next(o for o in ctx["static"].objects if o.name.startswith("frame_h_-"))
    return (min(right.location.x + v.co.x for v in right.data.vertices),
            max(bottom.location.y + v.co.y for v in bottom.data.vertices))


def tile_body_xy_outline(ctx: dict[str, Any], x: float, y: float, yaw: float) -> list[tuple[float, float]]:
    """Footprint of the real beveled tile body, rather than nominal OBB."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + v.co.x*c - v.co.y*s, y + v.co.x*s + v.co.y*c)
            for v in ctx["meshes"]["tile_body"].vertices]


def place_scene(ctx: dict[str, Any], hand: dict[str, Any], index: int, rng: random.Random, regime: str,
                crops: dict[str, tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    """Place a physically plausible one-player sector aligned to mjtensu capture guides.

    Player sits on world -Y.  Calls are anchored at the player's lower-right
    corner (+X, -Y) and accumulate upward (+Y).  Hand and dora are positioned
    relative to that same anchor so the real fixed PWA guide rectangles, not a
    post-hoc auto-crop, determine what enters the detector composite.
    """
    tiles: list[dict[str, Any]] = []
    footprints: list[tuple[str, list[tuple[float, float]]]] = []
    frame_right, frame_bottom = black_frame_inner_faces(ctx)

    def add_slots(slots: list[dict[str, Any]], region: str, group: str, meld_type: str | None,
                  center: tuple[float, float], angle: float, separated_last: bool = False,
                  align_right: float | None = None, align_bottom: float | None = None):
        poses = linear_poses(
            slots, center, angle, rng, regime, separated_last,
            align_right=align_right if region != "melds" else None,
            align_bottom=align_bottom if region != "melds" else None,
        )
        if poses and region == "melds":
            bodies = [tile_body_xy_outline(ctx, *pose) for pose in poses]
            dx = align_right - max(px for poly in bodies for px, _ in poly) if align_right is not None else 0.0
            dy = align_bottom - min(py for poly in bodies for _, py in poly) if align_bottom is not None else 0.0
            poses = [(x + dx, y + dy, yaw) for x, y, yaw in poses]
        for si, (slot, pose) in enumerate(zip(slots, poses)):
            x, y, yaw = pose
            poly = obb_corners(x, y, yaw)
            if any(polygons_overlap(poly, old) for _, old in footprints):
                raise ValueError(f"collision while placing {region}/{group}/{si}")
            if region == "melds":
                body = tile_body_xy_outline(ctx, x, y, yaw)
                if max(px for px, _ in body) > frame_right + 1e-7 or min(py for _, py in body) < frame_bottom - 1e-7:
                    raise ValueError(f"meld outside frame: {group}/{si}")
            elif max(abs(px) for px, _ in poly) > INNER_X/2 or max(abs(py) for _, py in poly) > INNER_Y/2:
                raise ValueError(f"tile outside inner table: {region}/{group}/{si}")
            tile = create_tile(ctx, f"tile_{index:06d}_{region}_{group}_{si}", slot["tile"], slot.get("face", "front"),
                               region, group, meld_type, x, y, yaw, rng)
            tiles.append(tile)
            footprints.append((tile["id"], poly))

    # These nominal centers correspond to the PWA guide centers for a camera
    # framing only the near-side half of an ~815 mm automatic table.  Small
    # correlated jitter prevents a synthetic pixel grid while the fixed-guide
    # acceptance test below remains the final authority.
    scene_dx = rng.uniform(-0.006, 0.006)
    scene_dy = rng.uniform(-0.004, 0.004)

    # The bottommost and rightmost meld tile bodies touch the actual black frame.
    # Align against frame meshes, not against the nominal inner-felt rectangle.
    m = len(hand["melds"])
    meld_left = None
    if m:
        stack_right = frame_right
        stack_bottom = frame_bottom
        step = rng.uniform(0.043, 0.047)
        for mi, meld in enumerate(hand["melds"]):
            add_slots(
                meld["tiles"], "melds", f"meld-{mi}", meld["kind"], (0.0, 0.0),
                0.0, False, align_right=stack_right, align_bottom=stack_bottom + mi * step,
            )
        meld_polys = [
            obb_corners(tile["x"], tile["y"], math.radians(tile["yaw_deg"]))
            for tile in tiles if tile["region"] == "melds"
        ]
        meld_left = min(px for poly in meld_polys for px, _ in poly)

    hand_slots = [{"tile": t, "face": "front", "rotation": 0} for t in hand["concealed"]]
    hand_center = (
        -0.032 + scene_dx + rng.uniform(-0.005, 0.005),
        -0.368 + scene_dy + rng.uniform(-0.003, 0.003),
    )
    add_slots(
        hand_slots, "completed_hand", "concealed", None, hand_center,
        math.radians(rng.uniform(-3.0, 3.0)), rng.random() < 0.22,
        align_right=(meld_left - 3.0 * W) if (hand_slots and meld_left is not None) else None,
    )

    dora_slots = [{"tile": t, "face": "front", "rotation": 0} for t in hand["dora"]]
    dora_center = (
        rng.uniform(-0.085, 0.018) + scene_dx,
        -0.255 + scene_dy + rng.uniform(-0.005, 0.005),
    )
    add_slots(
        dora_slots, "dora_indicators", "dora", None, dora_center,
        math.radians(rng.uniform(-5.0, 5.0)), False,
    )
    return tiles

def set_material_variation(ctx: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    felt_choices = [(0.018, 0.115, 0.045, 1), (0.024, 0.145, 0.055, 1), (0.030, 0.125, 0.065, 1), (0.014, 0.095, 0.038, 1)]
    felt = rng.choice(felt_choices)
    body_rough = rng.uniform(0.20, 0.38)
    back_color = (rng.uniform(0.32, 0.55), rng.uniform(0.065, 0.16), rng.uniform(0.002, 0.012), 1)
    ctx["materials"]["felt"].node_tree.nodes["FeltBaseMix"].inputs[1].default_value = felt
    ctx["materials"]["body"].node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = body_rough
    ctx["materials"]["back"].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = back_color
    return {"felt_rgba": list(felt), "tile_roughness": body_rough, "back_rgba": list(back_color)}


def rgb_for_temperature(kind: str) -> tuple[float, float, float]:
    return {"warm": (1.0, 0.63, 0.38), "neutral": (1.0, 0.90, 0.78), "cool": (0.66, 0.80, 1.0)}[kind]


def setup_lighting(ctx: dict[str, Any], index: int, rng: random.Random,
                   config: dict[str, Any]) -> dict[str, Any]:
    """Diffuse overhead room lighting with fill, preserving soft tile contours."""
    # Home ceiling illumination: large diffuse sources, never deliberately
    # hard shadows or a fake off-frame shadow-blocking bar.
    profiles = [
        ("warm", "soft", "normal"), ("neutral", "soft", "normal"),
        ("cool", "soft", "normal"), ("warm", "soft", "normal"),
        ("neutral", "soft", "normal"), ("cool", "soft", "dim"),
        ("neutral", "soft", "normal"), ("warm", "soft", "bright"),
    ]
    temperature, shadow_style, brightness = profiles[index % len(profiles)]
    lighting_cfg = config.get("lighting", {})
    scene = ctx["scene"]
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    base = rgb_for_temperature(temperature)
    bg.inputs["Color"].default_value = (*[min(1.0, v * 0.56) for v in base], 1)
    world_strength = rng.uniform(0.24, 0.38)
    if brightness == "dim":
        world_strength = rng.uniform(0.20, 0.28)
    world_strength *= float(lighting_cfg.get("world_multiplier", 1.0))
    bg.inputs["Strength"].default_value = world_strength

    exposure = rng.uniform(-0.55, -0.31)
    if brightness == "dim":
        exposure = rng.uniform(-0.72, -0.54)
    elif brightness == "bright":
        exposure = rng.uniform(-0.28, -0.10)
    exposure += float(lighting_cfg.get("exposure_shift_ev", 0.0))
    scene.view_settings.exposure = exposure

    lights = []
    blocker_meta = None
    count = 2 if shadow_style == "soft" else 1
    for li in range(count):
        data = bpy.data.lights.new(f"key_{li}", "AREA")
        obj = bpy.data.objects.new(f"key_{li}", data)
        ctx["dynamic"].objects.link(obj)

        if shadow_style == "partial":
            # Light -> bar -> table.  The bar itself is on the player's/camera
            # side, outside the capture frame; only its soft stripe enters the
            # photographed region.
            obj.location = (
                rng.uniform(-0.30, 0.18),
                rng.uniform(-1.10, -0.88),
                rng.uniform(1.05, 1.35),
            )
            target = Vector((rng.uniform(-0.03, 0.08), rng.uniform(-0.33, -0.24), 0.01))
            energy = rng.uniform(22, 36)
            data.shape = "DISK"
            data.size = rng.uniform(0.22, 0.34)
            blocker = add_mesh_object(f"shadow_blocker_{index}", ctx["meshes"]["blocker"], ctx["dynamic"], ctx["materials"]["blocker"])
            blocker.location = (
                rng.uniform(-0.10, 0.08),
                rng.uniform(-0.58, -0.53),
                rng.uniform(0.36, 0.46),
            )
            blocker.rotation_euler[2] = math.radians(rng.uniform(-16, 16))
            blocker_meta = {"location": list(blocker.location), "yaw_deg": math.degrees(blocker.rotation_euler[2])}
        else:
            az = rng.uniform(-math.pi, math.pi)
            radius = rng.uniform(0.70, 1.00)
            sampled_z = rng.uniform(0.90, 1.25)
            obj.location = (
                math.cos(az) * radius,
                math.sin(az) * radius - 0.22,
                float(lighting_cfg.get("area_height_m", sampled_z)),
            )
            target = Vector((rng.uniform(-0.04, 0.06), rng.uniform(-0.34, -0.20), 0.01))
            energy = rng.uniform(28, 40) if li == 0 else rng.uniform(19, 28)
            if brightness == "dim":
                energy *= rng.uniform(0.76, 0.88)
            elif brightness == "bright":
                energy *= rng.uniform(1.05, 1.14)
            if shadow_style == "soft":
                data.shape = "DISK"
                data.size = rng.uniform(1.05, 1.45)
            else:
                data.size = rng.uniform(0.14, 0.24)

        data.color = base if li == 0 else tuple(min(1.0, v * 0.94 + 0.06) for v in base)
        data.energy = energy * float(lighting_cfg.get("energy_multiplier", 1.0))
        direction = target - obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        lights.append({
            "location": list(obj.location), "height": obj.location.z,
            "energy": data.energy, "size": data.size,
        })

    return {
        "profile": "diffuse-room-v1", "temperature": temperature,
        "shadow_style": shadow_style, "brightness": brightness,
        "world_strength": world_strength, "exposure": exposure, "lights": lights,
        "off_frame_blocker": blocker_meta,
        "area_height_m": lighting_cfg.get("area_height_m"),
        "energy_multiplier": float(lighting_cfg.get("energy_multiplier", 1.0)),
        "world_multiplier": float(lighting_cfg.get("world_multiplier", 1.0)),
        "exposure_shift_ev": float(lighting_cfg.get("exposure_shift_ev", 0.0)),
    }

def setup_camera(ctx: dict[str, Any], rng: random.Random,
                 completed_center_xy: tuple[float, float] | None = None,
                 clockwise_deg_range: tuple[float, float] = (10.0, 10.0),
                 camera_height_m: float = 0.40) -> dict[str, Any]:
    """40 cm camera, clockwise roll sampled deterministically from config."""
    camera = ctx["camera"]
    z = camera_height_m
    lens = 17.7
    camera.data.lens = lens
    camera.data.sensor_width = 36.0

    table_ccw_deg = rng.uniform(0.0, 30.0)
    az = math.radians(table_ccw_deg)
    radius = 0.095
    target = Vector((*completed_center_xy, 0.0)) if completed_center_xy is not None else Vector((0.108, -0.296, 0.0))
    offset_x = -radius * math.sin(az)
    offset_y = -radius * math.cos(az)
    camera.location = (target.x + offset_x, target.y + offset_y, z)

    forward = (target - camera.location).normalized()
    desired_up = Vector((0.0, 1.0, 0.0))
    up = desired_up - forward * desired_up.dot(forward)
    if up.length < 1e-6:
        up = Vector((1.0, 0.0, 0.0))
    up.normalize()
    right = forward.cross(up).normalized()
    z_axis = -forward
    q = Matrix((right, up, z_axis)).transposed().to_quaternion()
    camera_clockwise_deg = rng.uniform(*clockwise_deg_range)
    q = q @ Quaternion((0, 0, 1), math.radians(-camera_clockwise_deg))
    camera.rotation_euler = q.to_euler()
    tilt_deg = math.degrees(math.atan2(radius, z))
    return {
        "location": list(camera.location), "target": list(target), "lens_mm": lens,
        "roll_deg": -camera_clockwise_deg,
        "camera_clockwise_from_table_bottom_deg": camera_clockwise_deg,
        "sensor_width_mm": 36.0,
        "target_visible_width_m": z * 36.0 / lens, "tilt_deg": tilt_deg,
        "view_offset_xy_m": [offset_x, offset_y], "table_ccw_deg": table_ccw_deg,
    }

def object_bbox_pixels(children: list[Any], scene, camera) -> list[float] | None:
    w, h = scene.render.resolution_x, scene.render.resolution_y
    pts = []
    for obj in children:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            ndc = world_to_camera_view(scene, camera, world)
            if ndc.z <= 0:
                continue
            pts.append((ndc.x*w, (1.0-ndc.y)*h))
    if not pts:
        return None
    left, top = min(p[0] for p in pts), min(p[1] for p in pts)
    right, bottom = max(p[0] for p in pts), max(p[1] for p in pts)
    return [left, top, right-left, bottom-top]


def project_world(scene, camera, point: tuple[float, float, float]) -> tuple[float, float]:
    ndc = world_to_camera_view(scene, camera, Vector(point))
    return ndc.x*scene.render.resolution_x, (1-ndc.y)*scene.render.resolution_y


def union_bbox(boxes: list[list[float]]) -> list[float]:
    l = min(b[0] for b in boxes); t = min(b[1] for b in boxes)
    r = max(b[0]+b[2] for b in boxes); btm = max(b[1]+b[3] for b in boxes)
    return [l, t, r-l, btm-t]


def capture_source_rects(
    scene, config: dict[str, Any], rng: random.Random
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, float]]:
    """Port PWA guide math into render pixels with bbox-independent viewport variation.

    The guide constants come from computeDisplayRegionRects and object-fit:cover
    follows calculateVideoCoverGeometry. Viewport dimensions are sampled once
    per accepted image from config and never derived from generated tile boxes.
    """
    ref = config["capture_reference"]
    reference_width, reference_height = map(float, ref["reference_viewport"])
    scale_lo, scale_hi = map(float, ref["viewport_scale_range"])
    aspect_lo, aspect_hi = map(float, ref["viewport_aspect_scale_range"])
    viewport_scale = rng.uniform(scale_lo, scale_hi)
    aspect_scale = rng.uniform(aspect_lo, aspect_hi)
    aspect_root = math.sqrt(aspect_scale)
    viewport_width = reference_width * viewport_scale * aspect_root
    viewport_height = reference_height * viewport_scale / aspect_root
    margin = 14.0
    header_reserve = 54.0
    footer_reserve = 78.0
    gap = 10.0
    available_width = max(1.0, viewport_width - margin * 2.0)
    available_height = max(1.0, viewport_height - header_reserve - footer_reserve)

    square = available_height
    row_height = (square - gap) / 2.0
    row_width = row_height * 17.0 / 4.0
    total_width = row_width + gap + square
    if total_width > available_width:
        factor = available_width / total_width
        square *= factor
        row_height *= factor
        row_width *= factor
        total_width = available_width

    left = (viewport_width - total_width) / 2.0
    top = header_reserve + (available_height - square) / 2.0
    display = {
        "dora_indicators": (left, top, row_width, row_height),
        "completed_hand": (left, top + row_height + gap, row_width, row_height),
        "melds": (left + row_width + gap, top, square, square),
    }

    source_width = float(scene.render.resolution_x)
    source_height = float(scene.render.resolution_y)
    scale = max(viewport_width / source_width, viewport_height / source_height)
    offset_x = (viewport_width - source_width * scale) / 2.0
    offset_y = (viewport_height - source_height * scale) / 2.0
    result = {}
    for key, (x, y, width, height) in display.items():
        sx = (x - offset_x) / scale
        sy = (y - offset_y) / scale
        sw = width / scale
        sh = height / scale
        left_px = min(source_width, max(0.0, sx))
        top_px = min(source_height, max(0.0, sy))
        right_px = min(source_width, max(left_px, sx + sw))
        bottom_px = min(source_height, max(top_px, sy + sh))
        result[key] = (left_px, top_px, right_px - left_px, bottom_px - top_px)
    meta = {
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
        "source_width": source_width,
        "source_height": source_height,
        "cover_scale": scale,
        "cover_offset_x": offset_x,
        "cover_offset_y": offset_y,
    }
    return result, meta

def clip_intersection(box: list[float], crop: tuple[int, int, int, int]) -> tuple[list[float] | None, float]:
    bx, by, bw, bh = box
    cx, cy, cw, ch = crop
    l=max(bx,cx); t=max(by,cy); r=min(bx+bw,cx+cw); b=min(by+bh,cy+ch)
    if r <= l or b <= t or bw <= 0 or bh <= 0:
        return None, 0.0
    return [l,t,r-l,b-t], ((r-l)*(b-t))/(bw*bh)


def to_composite_bbox(box: list[float], crop: tuple[int, int, int, int], dest: tuple[int,int,int,int]) -> list[float]:
    cx, cy, cw, ch = crop
    dx, dy, dw, dh = dest
    clipped, _ = clip_intersection(box, crop)
    if clipped is None:
        raise ValueError("box outside crop")
    l,t,w,h = clipped
    sx, sy = dw/cw, dh/ch
    return [dx+(l-cx)*sx, dy+(t-cy)*sy, w*sx, h*sy]


def compose_image(full_path: Path, crops: dict[str, tuple[float,float,float,float]], out_path: Path) -> None:
    # Parallel Blender workers can expose the freshly-written PNG a few ms
    # before it is readable from the shared filesystem. Retry that transient
    # window instead of losing an otherwise valid deterministic sample.
    src = None
    last_error: OSError | None = None
    for _ in range(30):
        try:
            with Image.open(full_path) as opened:
                src = opened.convert("RGB")
            break
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    if src is None:
        assert last_error is not None
        raise last_error
    out = Image.new("RGB", COMPOSITE_SIZE, PADDING_RGB)
    for region, crop in crops.items():
        dx, dy, dw, dh = REGIONS[region]["dest"]
        x, y, w, h = crop
        # PIL EXTENT accepts floating source coordinates, matching browser
        # Canvas drawImage more closely than integer pre-crops do.
        patch = src.transform(
            (dw, dh), Image.Transform.EXTENT, (x, y, x + w, y + h),
            resample=Image.Resampling.BICUBIC,
        )
        out.paste(patch, (dx, dy))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, optimize=True, compress_level=3)

def balanced_meld_count(index: int, base_seed: int) -> int:
    """Each five consecutive indices get all 0..4 counts in a seed-shuffled order."""
    choices = list(range(5))
    random.Random(base_seed + (index // 5) * 2654435761).shuffle(choices)
    return choices[index % 5]


def render_one(ctx: dict[str, Any], config: dict[str, Any], out: Path, index: int, base_seed: int,
               save_full_views: bool, retry_round: int = 0) -> dict[str, Any]:
    sample_seed = (base_seed + index*1000003) & 0x7fffffff
    # Coverage choices are index-deterministic and do not change on placement
    # rejection, so difficult 3/4-meld scenes are not silently under-sampled.
    coverage_rng = random.Random(sample_seed)
    if config["coverage"].get("meld_count_sampling") == "balanced_shuffled_blocks":
        meld_count = balanced_meld_count(index, base_seed)
    else:
        meld_count = coverage_rng.choices(
            [0, 1, 2, 3, 4], weights=config["coverage"]["meld_count_weights"], k=1
        )[0]
    force_white = (
        meld_count > 0
        and coverage_rng.random() < config["coverage"]["white_meld_probability_given_present"]
    )
    completed_hand_empty = (
        coverage_rng.random() < config["coverage"].get("completed_hand_empty_probability", 0.0)
    )
    dora_indicators_empty = (
        coverage_rng.random() < config["coverage"].get("dora_indicators_empty_probability", 0.0)
    )
    regime = coverage_rng.choices(
        ["neat", "ordinary", "messy"], weights=config["placement"]["regime_weights"], k=1
    )[0]
    capture_rng = random.Random(sample_seed ^ 0x5EEDC0DE)
    crops, capture_geometry = capture_source_rects(ctx["scene"], config, capture_rng)
    max_attempts = int(config["runtime"].get("max_attempts_per_image", 64))
    for attempt in range(max_attempts):
        rng = random.Random(sample_seed + attempt*7919 + 104729 + retry_round*15485863)
        clear_dynamic(ctx)
        try:
            hand = build_hand(
                index, rng, meld_count, force_white,
                completed_hand_empty, dora_indicators_empty,
            )
            materials = set_material_variation(ctx, rng)
            tiles = place_scene(ctx, hand, index, rng, regime, crops)
            hand_corners = [
                point for tile in tiles if tile["region"] == "completed_hand"
                for point in obb_corners(tile["x"], tile["y"], math.radians(tile["yaw_deg"]))
            ]
            completed_center = (
                (min(p[0] for p in hand_corners) + max(p[0] for p in hand_corners)) / 2,
                (min(p[1] for p in hand_corners) + max(p[1] for p in hand_corners)) / 2,
            ) if hand_corners else None
            camera_meta = setup_camera(
                ctx, rng, completed_center,
                tuple(config.get("camera", {}).get("clockwise_deg_range", (10.0, 10.0))),
                float(config.get("camera", {}).get("height_m", 0.40)),
            )
            bpy.context.view_layer.update()
            # Try wider fields of view before rejecting the scene. Rotating a
            # long 14-tile hand can otherwise exceed a fixed guide at 17.7mm.
            # Height, clockwise roll, tile positions and crops stay unchanged.
            xy_values = None
            for candidate_lens in config.get("camera", {}).get(
                "lens_retry_mm", (17.7, 16.0, 14.0, 12.0, 10.0)
            ):
                ctx["camera"].data.lens = float(candidate_lens)
                xy_solver = XYGuideOptimizer(
                    ctx, tiles, crops, sys.modules[__name__],
                    max_foreign_ratio=float(config["annotation"].get("max_foreign_area_ratio", 0.10)),
                )
                xy_score, xy_values, xy_result = xy_solver.solve()
                if xy_values is not None and not (
                    xy_result["bad_own"] or xy_result["bad_foreign"] or xy_result["physical_penalty"]
                ):
                    break
            if xy_values is None or (
                xy_result["bad_own"] or xy_result["bad_foreign"]
                or xy_result["physical_penalty"]
            ):
                raise ValueError(f"XY layout cannot fit fixed capture guides: {xy_result}")
            xy_solver.apply(xy_values)
            camera_meta["lens_mm"] = float(ctx["camera"].data.lens)
            camera_meta["target_visible_width_m"] = (
                float(config.get("camera", {}).get("height_m", 0.40)) * 36.0 / camera_meta["lens_mm"]
            )
            camera_meta["location"] = list(ctx["camera"].location)
            camera_meta["target"][0] += xy_values[0]
            camera_meta["target"][1] += xy_values[1]
            camera_meta["optimized_xy"] = {
                "camera_m": [xy_values[0], xy_values[1]],
                "dora_m": [xy_values[2], xy_values[3]],
                "objective": xy_score,
                "diagnostics": xy_result,
            }
            bpy.context.view_layer.update()
            lighting = setup_lighting(ctx, index, rng, config)
            bpy.context.view_layer.update()
            for tile in tiles:
                bbox = object_bbox_pixels(tile["children"], ctx["scene"], ctx["camera"])
                if bbox is None:
                    raise ValueError("unprojectable tile")
                tile["bbox_full"] = bbox

            # Every projected bbox must fit completely in its own fixed crop.
            # A tile may overlap another region by strictly less than 10% of
            # that tile's projected bbox area (not of the target crop area).
            for tile in tiles:
                b = tile["bbox_full"]
                x, y, w, h = crops[tile["region"]]
                if (b[0] < x - 1e-6 or b[1] < y - 1e-6
                        or b[0] + b[2] > x + w + 1e-6
                        or b[1] + b[3] > y + h + 1e-6):
                    raise ValueError(f"{tile['region']} tile bbox not completely inside its own crop")
                for other, other_crop in crops.items():
                    if other == tile["region"]:
                        continue
                    _, foreign_fraction = clip_intersection(b, other_crop)
                    if foreign_fraction >= float(config["annotation"].get("max_foreign_area_ratio", 0.10)):
                        raise ValueError(
                            f"{tile['region']} tile intrudes into {other} by "
                            f"{foreign_fraction:.3%} of its bbox area"
                        )

            frame_contact = None
            meld_tiles = [tile for tile in tiles if tile["region"] == "melds"]
            if meld_tiles:
                meld_polys = [obb_corners(tile["x"], tile["y"], math.radians(tile["yaw_deg"])) for tile in meld_tiles]
                body_polys = [tile_body_xy_outline(ctx, tile["x"], tile["y"], math.radians(tile["yaw_deg"]))
                              for tile in meld_tiles]
                frame_right, frame_bottom = black_frame_inner_faces(ctx)
                meld_right = max(px for poly in body_polys for px, _ in poly)
                meld_bottom = min(py for poly in body_polys for _, py in poly)
                right_clearance = frame_right - meld_right
                bottom_clearance = meld_bottom - frame_bottom
                if abs(right_clearance) > 1e-6 or abs(bottom_clearance) > 1e-6:
                    raise ValueError(
                        f"meld body not touching black frame meshes: "
                        f"right={right_clearance*1000:.3f}mm bottom={bottom_clearance*1000:.3f}mm"
                    )
                frame_contact = {
                    "right_clearance_mm": round(right_clearance * 1000, 6),
                    "bottom_clearance_mm": round(bottom_clearance * 1000, 6),
                    "right_tile_id": meld_tiles[max(range(len(body_polys)), key=lambda i: max(p[0] for p in body_polys[i]))]["id"],
                    "bottom_tile_id": meld_tiles[min(range(len(body_polys)), key=lambda i: min(p[1] for p in body_polys[i]))]["id"],
                }

                hand_tiles = [tile for tile in tiles if tile["region"] == "completed_hand"]
                if hand_tiles:
                    hand_polys = [obb_corners(tile["x"], tile["y"], math.radians(tile["yaw_deg"])) for tile in hand_tiles]
                    meld_left_world = min(px for poly in meld_polys for px, _ in poly)
                    hand_right_world = max(px for poly in hand_polys for px, _ in poly)
                    gap_tiles = (meld_left_world - hand_right_world) / W
                    if abs(gap_tiles - 3.0) > 0.02:
                        raise ValueError(f"completed-to-meld gap is not 3 tiles: {gap_tiles:.3f}")
            break
        except ValueError:
            if attempt == max_attempts - 1:
                raise
            continue

    tmp_dir = out / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    full_path = tmp_dir / f"full_{index:06d}.png"
    ctx["scene"].render.filepath = str(full_path)
    bpy.ops.render.render(write_still=True)
    image_rel = Path("images") / f"synthetic_{index:06d}.png"
    image_path = out / image_rel
    compose_image(full_path, crops, image_path)
    if save_full_views:
        full_dir = out / "full_views"
        full_dir.mkdir(parents=True, exist_ok=True)
        final_full = full_dir / full_path.name
        full_path.replace(final_full)
        full_rel = str(final_full.relative_to(out))
    else:
        full_path.unlink(missing_ok=True)
        full_rel = None

    annotations = []
    tile_meta = []
    ann_base = index * 100
    for ti, tile in enumerate(tiles):
        bbox = to_composite_bbox(tile["bbox_full"], crops[tile["region"]], REGIONS[tile["region"]]["dest"])
        x,y,w,h = bbox
        ann = {
            "id": ann_base + ti + 1, "image_id": index + 1, "category_id": 1,
            "bbox": [round(x,4),round(y,4),round(w,4),round(h,4)],
            "area": round(w*h,4), "iscrowd": 0,
            "segmentation": [[round(x,4),round(y,4),round(x+w,4),round(y,4),round(x+w,4),round(y+h,4),round(x,4),round(y+h,4)]],
            "region": tile["region"], "tile_identity": tile["identity"], "face": tile["face"],
            "meld_group": tile["group"] if tile["region"]=="melds" else None, "meld_type": tile["meld_type"],
        }
        annotations.append(ann)
        tile_meta.append({
            "tile_identity": tile["identity"], "region": tile["region"], "face": tile["face"],
            "meld_group": tile["group"], "meld_type": tile["meld_type"],
            "transform": {"x": tile["x"], "y": tile["y"], "z": T/2, "yaw_deg": tile["yaw_deg"]},
            "bbox_full": [round(v,4) for v in tile["bbox_full"]],
            "bbox_composite": ann["bbox"], "visibility": tile["visibility"],
        })
    image_record = {
        "id": index + 1, "file_name": image_rel.as_posix(), "width": 320, "height": 320,
        "dataset_origin": "blender_synthetic", "seed": sample_seed, "placement_regime": regime,
        "meld_group_count": meld_count, "white_dragon_meld": force_white,
    }
    record = {
        "schema": "mjtensu.blender-synthetic-nanodet-scene/v1",
        "index": index, "seed": sample_seed, "attempt": attempt,
        "image": image_record, "annotations": annotations,
        "scene": {
            "placement_regime": regime, "hand": hand, "crops": {k:[round(v,4) for v in rect] for k,rect in crops.items()},
            "capture_geometry": {k: round(v, 6) for k, v in capture_geometry.items()},
            "camera": camera_meta, "lighting": lighting, "materials": materials,
            "black_frame_contact": frame_contact,
            "full_view": full_rel, "tiles": tile_meta,
        },
    }
    rec_path = out / "records" / f"synthetic_{index:06d}.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(record, ensure_ascii=False, separators=(",",":"))+"\n", encoding="utf-8")
    return record

def assemble(out: Path, config: dict[str, Any], args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    records = []
    for path in sorted((out/"records").glob("synthetic_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    coco = {
        "info": {"description":"Blender-generated fixed-layout 320x320 Mahjong tile detector corpus", "version":"1.0"},
        "licenses": [{"id":1,"name":"CC0 / public-domain tile artwork; generated scene images"}],
        "images": [r["image"] for r in records],
        "annotations": [a for r in records for a in r["annotations"]],
        "categories": [CATEGORY],
    }
    ann_dir = out/"annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    all_path = ann_dir/"instances_all.json"
    all_path.write_text(json.dumps(coco,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
    manifest_path = out/"manifest.jsonl"
    with manifest_path.open("w",encoding="utf-8") as f:
        for r in records:
            ip = out/r["image"]["file_name"]
            f.write(json.dumps({"image":r["image"]["file_name"],"sha256":sha256(ip),"record":f"records/synthetic_{r['index']:06d}.json","seed":r["seed"]},separators=(",",":"))+"\n")
    provenance = {
        "schema":"mjtensu.blender-synthetic-nanodet-corpus/v1",
        "generator":str(Path(__file__).resolve()), "generator_sha256":sha256(Path(__file__).resolve()),
        "config":config, "config_path":str(args.config.resolve()), "config_sha256":sha256(args.config.resolve()),
        "tile_art_source":{"upstream":"https://github.com/FluffyStuff/riichi-mahjong-tiles","variant":"Export/Regular","license":"CC0/public domain"},
        "blender_version":bpy.app.version_string, "base_seed":args.seed,
        "image_count":len(coco["images"]), "annotation_count":len(coco["annotations"]),
        "composite_layout":{"width":320,"height":320,"padding_rgb":[0,0,0],"regions":REGIONS},
        "generated_unix":time.time(),
    }
    prov_path=out/"provenance.json"
    prov_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"images":len(coco["images"]),"annotations":len(coco["annotations"]),"coco":str(all_path),"manifest":str(manifest_path),"provenance":str(prov_path)}


def main() -> None:
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("start-index must be >=0")
    repo = Path(__file__).resolve().parents[3]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("lighting", {}).get("profile", "diffuse-room-v1") != "diffuse-room-v1":
        raise ValueError("unsupported lighting profile; must be diffuse-room-v1")
    out = args.output.resolve()
    for d in ("images","records","annotations","tmp"):
        (out/d).mkdir(parents=True, exist_ok=True)
    if args.assemble_only:
        summary = assemble(out, config, args, repo)
        print(json.dumps({"status":"assembled",**summary},indent=2),flush=True)
        return
    if args.count <= 0:
        raise ValueError("count must be >0 unless --assemble-only is used")
    tooling = repo/".local/recognition/blender_synthetic_tooling"
    asset_dir = tooling/"riichi-mahjong-tiles-master/Export/Regular"
    if not asset_dir.is_dir():
        raise FileNotFoundError(asset_dir)
    decals = ensure_decals(asset_dir, tooling/"glyph_decals")
    ctx = setup_world(config, decals)
    completed = 0
    reused = 0
    deferred: dict[int, str] = {}
    started = time.time()
    for index in range(args.start_index, args.start_index + args.count):
        rec = out/"records"/f"synthetic_{index:06d}.json"
        img = out/"images"/f"synthetic_{index:06d}.png"
        if args.resume and rec.is_file() and img.is_file():
            reused += 1
            continue
        try:
            render_one(ctx, config, out, index, args.seed, args.save_full_views)
        except ValueError as exc:
            deferred[index] = str(exc)
            print(json.dumps({"event":"deferred", "index":index, "error":str(exc)}, ensure_ascii=False), flush=True)
            continue
        completed += 1
        if completed % int(config["runtime"]["progress_every"]) == 0:
            elapsed = time.time()-started
            print(json.dumps({"progress":completed,"last_index":index,"elapsed_s":round(elapsed,1),"images_per_s":round(completed/max(elapsed,0.001),3)}),flush=True)
    # Do not abort the whole corpus on one difficult scene. Retry deferred
    # indices with independent deterministic placement streams after the pass.
    for retry_round in range(1, int(config["runtime"].get("deferred_retry_rounds", 3)) + 1):
        if not deferred:
            break
        for index in list(deferred):
            try:
                render_one(ctx, config, out, index, args.seed, args.save_full_views,
                           retry_round=retry_round)
            except ValueError as exc:
                deferred[index] = str(exc)
                print(json.dumps({"event":"retry_deferred", "index":index,
                                  "round":retry_round, "error":str(exc)}), flush=True)
            else:
                completed += 1
                del deferred[index]
    failed_path = out / f"failed_indices_{args.start_index:06d}_{args.start_index + args.count:06d}.json"
    if deferred:
        failed_path.write_text(json.dumps({"seed":args.seed,"failures":deferred},
                                          indent=2,ensure_ascii=False)+"\n")
    else:
        failed_path.unlink(missing_ok=True)
    status = "partial" if deferred else "completed"
    if args.skip_assemble:
        print(json.dumps({"status":status,"new_images":completed,"reused":reused,
                          "failed_indices":list(deferred),"output":str(out)}, indent=2),flush=True)
        return
    summary = assemble(out, config, args, repo)
    print(json.dumps({"status":status,"new_images":completed,"reused":reused,
                      "failed_indices":list(deferred),**summary},indent=2),flush=True)


if __name__ == "__main__":
    main()
