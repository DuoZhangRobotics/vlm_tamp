"""Local, read-only retrieval of public observation PNGs. Copied as visual.py.

Only derived images and an audit log are written. No robot RPC, simulator state,
planner, fitted geometry, learned detector or external model is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import uuid


HELP = """Saved visual observations (local; does not call the robot service)

python3 visual.py overview '{"observation_id":1}'
  One labeled contact sheet of all cameras from an existing capture. Use it to
  find a view where the relevant parts are visible, then open that camera's
  original image before selecting geometry pixels. Thumbnails are not original
  coordinates. This does not capture more images or move the robot.

python3 visual.py history '{"limit":8}'
  Lists captured observation IDs, control steps and available cameras in the
  latest attempt. Optional attempt selects another attempt in THIS workspace;
  before selects earlier observation IDs. limit is 1..32. This does not capture
  a new image, advance physics, or return another episode's private information.

python3 visual.py inspect '{"views":[{"observation_id":1,"camera":"frontview"},{"observation_id":4,"camera":"frontview"}]}'
  Select 1..4 already captured views, in the desired display order. Each view
  requires observation_id and camera. Optional region_xyxy is an integer
  [left,top,right,bottom] rectangle in ORIGINAL pixels, with right/bottom
  exclusive. Omit it for the full image. Optional label is a short caption;
  optional attempt defaults to the latest attempt. No automatic target choice,
  semantic interpretation, motion estimate or new observation is supplied.

The result gives comparison_image: open that file with your image viewer.
Each panel identifies attempt, observation, camera, control step and source
rectangle. Crops are enlarged with nearest-neighbor interpolation; enlargement
adds no observed detail. metadata_file records the exact original coordinates,
source hashes, image bounds and canvas-pixel-to-original-pixel transforms.
Use original pixel coordinates when calling robot geometry tools. Individual
crop files retain their original resolution. Depth and calibration stay in the
existing observation files; this tool does not alter them or the planner model.
"""


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return value


def schema_keys(value, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError("Arguments and views must be JSON objects")
    if set(value) - set(allowed):
        raise ValueError("Unknown fields: " + ", ".join(sorted(set(value) - set(allowed))))
    if set(required) - set(value):
        raise ValueError("Missing fields: " + ", ".join(sorted(set(required) - set(value))))


def confined(root, path, *, must_exist=True):
    root = root.resolve()
    path = Path(path)
    if not path.is_relative_to(root):
        raise ValueError("Path must stay inside this public workspace")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Observation and output paths must not be symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Resolved path left the public workspace")
    if must_exist and not path.exists():
        raise ValueError("Requested public observation is unavailable")
    return path


def latest_attempt(root):
    public = confined(root, root / "robot_data")
    values = [int(p.name[7:]) for p in public.iterdir()
              if re.fullmatch(r"attempt[0-9]+", p.name) and p.is_dir() and not p.is_symlink()]
    if not values:
        raise ValueError("No captured observations yet; use robot reset/observe first")
    return max(values)


def observation_dir(root, attempt, observation_id):
    integer(attempt, "attempt", 1, 999)
    integer(observation_id, "observation_id", 0, 999999)
    path = confined(root, root / "robot_data" / f"attempt{attempt:02d}" / f"obs{observation_id:03d}")
    for name in ["robot.json", "calibration.json"]:
        confined(root, path / name)
    return path


def history(root, arguments):
    schema_keys(arguments, ["attempt", "limit", "before"])
    attempt = arguments.get("attempt", latest_attempt(root))
    integer(attempt, "attempt", 1, 999)
    limit = integer(arguments.get("limit", 8), "limit", 1, 32)
    before = integer(arguments.get("before", 1000000), "before", 0, 1000000)
    parent = confined(root, root / "robot_data" / f"attempt{attempt:02d}")
    ids = sorted(int(p.name[3:]) for p in parent.iterdir()
                 if re.fullmatch(r"obs[0-9]+", p.name) and p.is_dir() and not p.is_symlink()
                 and int(p.name[3:]) < before)
    records, incomplete = [], 0
    for obs_id in reversed(ids):
        if len(records) >= limit:
            break
        try:
            folder = observation_dir(root, attempt, obs_id)
            state = json.loads((folder / "robot.json").read_text())
            calibration = json.loads((folder / "calibration.json").read_text())
            cameras = sorted(name for name in calibration if (folder / (name + ".png")).is_file())
            records.append({"attempt": attempt, "observation_id": obs_id,
                            "control_steps": state["control_steps"], "cameras": cameras})
        except (ValueError, KeyError, OSError):
            incomplete += 1
    records.reverse()
    return {"observations": records, "attempt": attempt, "incomplete_entries_skipped": incomplete,
            "earlier_than": records[0]["observation_id"] if records else None,
            "new_sensor_capture": False, "robot_motion_issued": False}


def inspect(root, arguments):
    from PIL import Image, ImageDraw

    schema_keys(arguments, ["views"], ["views"])
    views = arguments["views"]
    if not isinstance(views, list) or not 1 <= len(views) <= 4:
        raise ValueError("views must contain 1..4 selected observations")
    prepared = []
    for view in views:
        schema_keys(view, ["attempt", "observation_id", "camera", "region_xyxy", "label"],
                    ["observation_id", "camera"])
        camera = view["camera"]
        if not isinstance(camera, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", camera):
            raise ValueError("camera must be a recorded camera name")
        attempt = view.get("attempt", latest_attempt(root))
        folder = observation_dir(root, attempt, view["observation_id"])
        calibration = json.loads((folder / "calibration.json").read_text())
        if camera not in calibration:
            raise ValueError("Camera is absent from that observation's calibration")
        source = confined(root, folder / (camera + ".png"))
        with Image.open(source) as opened:
            w, h = opened.size
            if w * h > 16_000_000:
                raise ValueError("Saved image exceeds supported inspection size")
            frame = opened.convert("RGB")
        region = view.get("region_xyxy", [0, 0, w, h])
        if not isinstance(region, list) or len(region) != 4 or any(type(x) is not int for x in region):
            raise ValueError("region_xyxy must contain four integer original-pixel coordinates")
        x0, y0, x1, y1 = region
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("region_xyxy must be nonempty and inside the original image")
        label = view.get("label", "")
        if not isinstance(label, str) or len(label) > 60 or any(ord(c) < 32 for c in label):
            raise ValueError("label must be at most60 characters without control characters")
        state = json.loads((folder / "robot.json").read_text())
        prepared.append((frame.crop(region), {"attempt": attempt, "observation_id": view["observation_id"],
            "camera": camera, "label": label, "control_steps": state["control_steps"],
            "source_image": str(source.relative_to(root)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "original_image_size": [w, h], "region_xyxy": region,
            "calibration_file": str((folder / "calibration.json").relative_to(root))}))
    # Validate every view before creating any output image.
    parent = confined(root, root / "visual_inspections", must_exist=False)
    parent.mkdir(exist_ok=True)
    dest = parent / uuid.uuid4().hex
    dest.mkdir()
    panel_size, header, gap = 512, 64, 8
    columns = min(2, len(prepared))
    rows = (len(prepared) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * (panel_size + gap) + gap,
                               rows * (panel_size + header + gap) + gap), (245, 247, 250))
    draw, metadata = ImageDraw.Draw(canvas), []
    for i, (crop, entry) in enumerate(prepared):
        cw, ch = crop.size
        factor = min(panel_size / cw, panel_size / ch)
        rw, rh = max(1, round(cw * factor)), max(1, round(ch * factor))
        px = gap + (i % columns) * (panel_size + gap)
        py = gap + (i // columns) * (panel_size + header + gap)
        ix, iy = px + (panel_size - rw) // 2, py + header + (panel_size - rh) // 2
        caption = (f"{entry['label']}  attempt{entry['attempt']:02d}/obs{entry['observation_id']:03d}\n"
                   f"{entry['camera']} | control step {entry['control_steps']}\n"
                   f"Original crop {entry['region_xyxy']} | {entry['original_image_size']}")
        draw.multiline_text((px + 4, py + 3), caption, fill=(20, 35, 50), spacing=4)
        canvas.paste(crop.resize((rw, rh), Image.Resampling.NEAREST), (ix, iy))
        raw_path = dest / f"view{i:02d}.png"
        crop.save(raw_path)
        x0, y0, _, _ = entry["region_xyxy"]
        sx, sy = cw / rw, ch / rh
        entry.update(crop_image=str(raw_path.relative_to(root)),
            canvas_bounds_xyxy=[ix, iy, ix + rw, iy + rh],
            canvas_pixel_to_original_affine=[[sx, 0., x0 - ix * sx + .5 * (sx - 1.)],
                                             [0., sy, y0 - iy * sy + .5 * (sy - 1.)]])
        metadata.append(entry)
    output = dest / "comparison.png"
    canvas.save(output)
    data = {"schema": "public-visual-inspection-v1", "views": metadata,
            "comparison_image": str(output.relative_to(root)), "canvas_size": list(canvas.size),
            "pixel_coordinate_convention": "pixel centers; top-left origin; x right, y down",
            "interpolation": "nearest neighbor; no added observed detail",
            "new_sensor_capture": False, "robot_motion_issued": False,
            "planner_state_changed": False, "semantic_interpretation_supplied": False}
    meta = dest / "metadata.json"
    meta.write_text(json.dumps(data, indent=2) + "\n")
    return {"comparison_image": data["comparison_image"], "metadata_file": str(meta.relative_to(root)),
            "views": [{k: v[k] for k in ["attempt", "observation_id", "camera", "control_steps", "region_xyxy", "crop_image"]}
                      for v in metadata], "view_with": "Open comparison_image with your image viewer",
            "coordinates": "Use original pixels for robot geometry; transforms are in metadata_file",
            "new_sensor_capture": False, "robot_motion_issued": False, "planner_state_changed": False}


def overview(root,arguments):
    from PIL import Image,ImageDraw
    root=Path(root).resolve()
    schema_keys(arguments,['attempt','observation_id'],['observation_id'])
    attempt=arguments.get('attempt',latest_attempt(root))
    folder=observation_dir(root,attempt,arguments['observation_id'])
    calibration=json.loads((folder/'calibration.json').read_text())
    state=json.loads((folder/'robot.json').read_text())
    names=sorted(name for name in calibration if (folder/(name+'.png')).is_file())
    if not names or len(names)>32:raise ValueError('Expected 1..32 recorded camera images')
    parent=confined(root,root/'visual_inspections',must_exist=False);parent.mkdir(exist_ok=True)
    dest=parent/uuid.uuid4().hex;dest.mkdir()
    width=256;header=32;columns=min(4,len(names));rows=(len(names)+columns-1)//columns
    canvas=Image.new('RGB',(columns*width,rows*(width+header)),(245,247,250));draw=ImageDraw.Draw(canvas)
    views=[]
    for i,name in enumerate(names):
        source=confined(root,folder/(name+'.png'))
        with Image.open(source) as original:
            frame=original.convert('RGB');size=frame.size
        frame.thumbnail((width,width),Image.Resampling.LANCZOS)
        x=(i%columns)*width;y=(i//columns)*(width+header)
        draw.text((x+6,y+8),f"{name} | obs{arguments['observation_id']} / t{state['control_steps']}",fill=(20,35,50));canvas.paste(frame,(x,y+header))
        views.append({'camera':name,'original_image':str(source.relative_to(root)),
            'original_size':size,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'thumbnail_bounds_xyxy':[x,y+header,x+frame.width,y+header+frame.height]})
    canvas.save(dest/'overview.png')
    (dest/'metadata.json').write_text(json.dumps({'views':views,'new_sensor_capture':False,
        'semantic_interpretation_supplied':False,'observation_id':arguments['observation_id']},indent=2))
    return {'comparison_image':str((dest/'overview.png').relative_to(root)),
        'observation_id':arguments['observation_id'],'control_steps':state['control_steps'],
        'metadata_file':str((dest/'metadata.json').relative_to(root)),
        'cameras':names,'original_images_folder':str(folder.relative_to(root)),
        'instruction':'Use this sheet to choose a view. Open that camera original PNG and annotate original pixels; inspect a view where both the contact and attachment edge are visible.',
        'new_sensor_capture':False,'robot_motion_issued':False}


def dispatch(root, operation, arguments):
    root = Path(root).resolve()
    started, wall = time.perf_counter(), time.time()
    response = None
    try:
        if operation not in {"history", "inspect", "overview"}:
            raise ValueError("Use history, inspect or overview; --help describes arguments")
        response = {"ok": True, "result": {"history": history, "inspect": inspect,"overview":overview}[operation](root, arguments)}
    except (ValueError, KeyError, OSError) as error:
        response = {"ok": False, "error_type": type(error).__name__, "error": str(error),
                    "new_sensor_capture": False, "robot_motion_issued": False}
    response["wall_seconds"] = time.perf_counter() - started
    # This local receipt is separate from robot RPC logs and budgets.
    record = {"operation": operation, "arguments": arguments, "response": response,
              "started_monotonic": started, "finished_monotonic": time.perf_counter(), "started_unix": wall}
    log = confined(root, root / "visual-events.jsonl", must_exist=False)
    with log.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return response


def install(workspace):
    """Install a public convenience tool, leaving the numerical service unchanged."""
    root = Path(workspace)
    target = root / "visual.py"
    target.write_text(Path(__file__).read_text())
    (root / "tools").mkdir(exist_ok=True)
    (root / "tools/visual_inspection.txt").write_text(HELP)
    with (root / "tools.txt").open("a") as stream:
        stream.write("\nSaved visual inspection (local, no robot call): python3 visual.py history '{}'\n"
                     "and python3 visual.py inspect '{\"views\":[...]}'. Read tools/visual_inspection.txt\n"
                     "for bounded history, full-frame comparisons and original-pixel crops.\n")
    return {"enabled": True, "entrypoint": "visual.py", "documentation": "tools/visual_inspection.txt",
            "source_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source": "existing public captured PNGs and calibration", "new_sensor_capture": False}


def main():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("operation", choices=["history", "inspect", "overview"])
    parser.add_argument("arguments", nargs="?", default="{}")
    args = parser.parse_args()
    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as error:
        parser.error(str(error))
    response = dispatch(Path(__file__).resolve().parent, args.operation, arguments)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
