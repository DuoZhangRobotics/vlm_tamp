#!/usr/bin/env python3
"""Export published watch data as an offline page and an optional chat fragment.

Reads only the small published JSON index/details and explicitly referenced media.
No experiment imports, GPU work, server, polling, or source-log scans are needed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
VIDEO_PROFILES = ((384, 8, 30), (256, 6, 33), (192, 4, 36), (160, 3, 39), (128, 2, 42), (96, 1, 46))


def asset_path(study: Path, relative: str) -> Path:
    path = (study / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(study.resolve()):
        raise ValueError("Asset is outside this study")
    if not path.is_file():
        raise ValueError("Asset has not been published yet")
    return path


def read_json(path: Path) -> dict:
    if path.stat().st_size > 8_000_000:
        raise ValueError("Published JSON exceeds 8 MB; use the compact watch-data export")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a published JSON object")
    return value


def script_json(value: object) -> str:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def media_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def transcode(source: Path, cache: Path, profile: tuple[int, int, int], *, image: bool = False) -> Path:
    width, fps, quality = profile
    stat = source.stat()
    key = hashlib.sha256(f"v1:{source}:{stat.st_size}:{stat.st_mtime_ns}:{profile}:{image}".encode()).hexdigest()
    output = cache / (key + (".jpg" if image else ".mp4"))
    if output.is_file():
        return output
    cache.mkdir(parents=True, exist_ok=True)
    # Keep decoding, filtering and encoding single-threaded on this shared server.
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source),
               "-filter_threads", "1", "-vf", f"scale='min({width},iw)':-2" + ("" if image else f",fps={fps}"),
               "-an", "-threads", "1"]
    command += ["-frames:v", "1", "-q:v", str(quality)] if image else [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    fd, temporary = tempfile.mkstemp(suffix=output.suffix, dir=cache)
    os.close(fd)
    try:
        subprocess.run(command + ["-y", temporary], check=True, capture_output=True, timeout=90)
        # A recording still being written is not a valid cached snapshot.
        if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
            raise ValueError("Recording changed during export; refresh after it is saved")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def load_study(study: Path) -> tuple[dict, list[tuple[dict, Path]]]:
    status = read_json(study / "watch-status.json")
    data = {"study": str(study), "status": status, "runs": []}
    videos = []
    for row in status.get("runs", []):
        run = {key: row.get(key) for key in ("id", "label", "outcome", "controls", "memory")}
        repetition = re.search(r"(?:^|-)r(\d+)(?:-|$)", row["id"])
        title = row.get("label", row["id"]) + (f" · repetition {repetition[1]}" if repetition else "")
        run.update(title=title, videos=[], images=[], notes=[])
        try:
            detail = read_json(asset_path(study, row["detail"]))
        except (OSError, ValueError, KeyError) as exc:
            detail = {}
            run["detail_error"] = "Details not available at snapshot time: " + str(exc)
        run["detail"] = {key: value for key, value in detail.items() if key not in ("links", "frame", "wrist_view", "contact_overlay", "videos")}
        run["image_paths"] = [(label, detail[key]) for key, label in (
            ("frame", "Recorded robot view"), ("wrist_view", "Wrist camera"), ("contact_overlay", "Contact overlay")) if detail.get(key)]
        for item in row.get("videos", detail.get("videos", [])):
            video = {"label": item.get("label", "Recording"), "partial": item.get("partial", False)}
            try:
                source = asset_path(study, item["path"])
                if source.suffix.lower() != ".mp4":
                    raise ValueError("Only published MP4 recordings are supported")
                video["source_path"] = str(source)
                videos.append((video, source))
            except (OSError, ValueError, KeyError) as exc:
                video["error"] = str(exc)
            run["videos"].append(video)
        data["runs"].append(run)
    # Start with a playable recording, while leaving the current live run selectable.
    data["default_run"] = next((r["id"] for r in reversed(data["runs"]) if r["videos"]), status.get("active_run"))
    return data, videos


def render(data: dict) -> str:
    status = data["status"]
    outcomes = status.get("task_outcomes", {})
    stamp = datetime.fromtimestamp(status["checked_at"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    values = {
        "SUMMARY": f"{outcomes.get('successes', 0)} successes / {outcomes.get('completed', 0)} completed episodes · {status.get('status_text', '')}",
        "STAMP": stamp, "HEALTH": status.get("health_text", ""),
        "STATUS": status.get("message", ""),
    }
    template = (TEMPLATES / "luna_watch_preview.html").read_text()
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", html.escape(str(value)))
    # Insert data last, so model-produced strings cannot be interpreted as placeholders.
    return template.replace("@@DATA@@", script_json(data))


def build(study: Path, cache: Path, max_bytes: int) -> tuple[str, dict]:
    data, videos = load_study(study)
    for run in data["runs"]:
        for label, relative in run.pop("image_paths"):
            try:
                source = asset_path(study, relative)
                if source.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    raise ValueError("Unsupported image type")
                small = transcode(source, cache, (256, 1, 6), image=True)
                run["images"].append({"label": label, "src": media_uri(small, "image/jpeg")})
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                run["notes"].append(f"{label}: preview unavailable ({type(exc).__name__}).")
    base_size = len(render(data).encode())
    if base_size > max_bytes - 2048:
        raise ValueError(f"Analysis and images need {base_size} bytes; increase --max-bytes for a standalone export")
    # Equal per-recording allowance keeps every saved trial represented, even in chat's 1 MB limit.
    allowance = max(0, (max_bytes - base_size - 2048) * 3 // 4 // max(1, len(videos)))
    for video, source in videos:
        try:
            for profile in VIDEO_PROFILES:
                small = transcode(source, cache, profile)
                if small.stat().st_size <= allowance:
                    video["src"] = media_uri(small, "video/mp4")
                    video["preview_width"] = profile[0]
                    video["preview_fps"] = profile[1]
                    break
            else:
                video["error"] = "Recording exceeds this compact preview's size allowance. Ask to show its full-resolution recording."
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            video["error"] = f"Video not ready at snapshot time ({type(exc).__name__}); refresh after recording completes."
    fragment = render(data)
    if len(fragment.encode()) > max_bytes:
        raise ValueError("Preview exceeds the requested size limit")
    return fragment, data


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".watch-preview-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, help="Standalone HTML; defaults to STUDY/watch-chat.html")
    parser.add_argument("--fragment", type=Path, help="Optional fragment for in-conversation viewing (under 1 MB)")
    parser.add_argument("--max-bytes", type=int, default=950_000)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "luna-watch-preview-cache")
    args = parser.parse_args()
    if args.fragment and args.max_bytes >= 1_000_000:
        parser.error("Chat fragments must be smaller than 1 MB")
    study = args.study.resolve()
    fragment, data = build(study, args.cache, args.max_bytes)
    document = (TEMPLATES / "luna_watch_preview_document.html").read_text().replace("@@FRAGMENT@@", fragment)
    output = args.output or study / "watch-chat.html"
    write_atomic(output, document)
    if args.fragment:
        write_atomic(args.fragment, fragment)
    print(json.dumps({"output": str(output), "fragment": str(args.fragment) if args.fragment else None,
                      "fragment_bytes": len(fragment.encode()), "runs": len(data["runs"]),
                      "videos_embedded": sum(bool(v.get("src")) for r in data["runs"] for v in r["videos"]),
                      "videos_total": sum(len(r["videos"]) for r in data["runs"])}))


if __name__ == "__main__":
    main()
