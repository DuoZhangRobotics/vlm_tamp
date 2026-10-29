"""Build a local, self-contained HTML explanation of the RoArtNet benchmark."""
import html
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def link(path, label, run):
    return f'<a href="{html.escape(os.path.relpath(path, run))}">{html.escape(label)}</a>'


def project(points, K):
    v = np.asarray(points) @ np.asarray(K).T
    return v[:, :2] / v[:, 2:3]


def plot_case(row, run):
    item, pred, ref = row["input"], row["prediction"], row["reference"]
    path = run / "plots" / (row["id"] + ".png")
    with np.load(ROOT / item["input"], allow_pickle=False) as data:
        points, rgb = data["point_cloud"], data["rgb"]
    keep = np.linalg.norm(points, axis=1) > .01
    points, rgb = points[keep], rgb[keep]
    pick = np.random.default_rng(0).choice(len(points), min(len(points), 6500), replace=False)
    points, rgb = points[pick], rgb[pick].astype(float)
    if rgb.max() > 1.:
        rgb /= 255.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    if item["kind"] == "local":
        T = np.asarray(item["camera_to_world"])
        q, b = np.asarray(ref["origin"]), np.asarray(ref["axis"])
        p, a = np.asarray(pred["joint_origin_world_m"]), np.asarray(pred["joint_axis_world"])
        axes[0].imshow(Image.open(ROOT / item["source_rgb"]))
        for origin, axis, color, label in [(p, a, "#ee4035", "RoArtNet"), (q, b, "#00bcd4", "Private reference")]:
            xyz = origin + np.linspace(-.15, .15, 50)[:, None] * axis
            optical = (xyz - T[:3, 3]) @ T[:3, :3]
            uv = project(optical, item["intrinsic"])
            axes[0].plot(uv[:, 0], uv[:, 1], color=color, linewidth=2, label=label)
        axes[0].set_xlim(0, 512)
        axes[0].set_ylim(512, 0)
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].set_title("Image projection can hide depth error")
        world = points @ T[:3, :3].T + T[:3, 3]
        axes[1].scatter(world[:, 0], world[:, 1], c=rgb, s=2, alpha=.6)
        intersection = p + a * (np.dot(q - p, b) / np.dot(a, b))
        axes[1].scatter(*intersection[:2], c="#ee4035", marker="x", s=100, label="Predicted axis at reference height")
        axes[1].scatter(*q[:2], c="#00bcd4", marker="+", s=130, label="Reference axis")
        axes[1].plot([q[0], intersection[0]], [q[1], intersection[1]], c="#ee4035", linestyle=":")
        axes[1].set_xlabel("World X (m)")
        axes[1].set_ylabel("World Y (m)")
        axes[1].set_title("Top view: metric position error")
        axes[1].legend(fontsize=7)
    else:
        p, a = np.asarray(pred["joint_origin_optical_m"]), np.asarray(pred["joint_axis_optical"])
        q, b = np.asarray(ref["origin"]), np.asarray(ref["axis"])
        for ax, dims in zip(axes, [(0, 1), (0, 2)]):
            ax.scatter(points[:, dims[0]], points[:, dims[1]], c=rgb, s=2, alpha=.6)
            for origin, axis, color, label in [(p, a, "#ee4035", "RoArtNet"), (q, b, "#00bcd4", "Author reference")]:
                xyz = origin + np.linspace(-.25, .25, 50)[:, None] * axis
                ax.plot(xyz[:, dims[0]], xyz[:, dims[1]], color=color, linewidth=2, label=label)
            ax.set_xlabel("Camera " + "xyz"[dims[0]] + " (m)")
            ax.set_ylabel("Camera " + "xyz"[dims[1]] + " (m)")
            ax.legend(fontsize=8)
        axes[0].set_title("Published cloud (near-zero rows omitted in plot only)")
        axes[1].set_title("Depth projection")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(alpha=.2)
    fig.suptitle(f"{item['id']} / {row['profile']} / seed{row['seed']}\n"
        f"Position {fmt(row['errors']['reference_plane_error_mm'],1)} mm; axis {fmt(row['errors']['axis_angle_deg'],1)}°", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    run = ROOT / "results/roartnet-perception-20261025"
    (run / "plots").mkdir(exist_ok=True)
    rows, summary = read(run / "scored-private.json"), read(run / "summary.json")
    follow, cleanup, motion = [read(run / name) for name in
        ["followup-summary.json", "cleanup/scored-private.json", "motion/scored-private.json"]]
    plots = {r["id"]: plot_case(r, run) for r in rows}
    primary = summary["groups"]["author-eval/local"]
    stages = primary["mean_timing_seconds"]
    colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#e41a1c", "#a65628", "#6baed6", "#e377c2", "#999999", "#66c2a5"]
    contents = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
      "<title>RoArtNet perception: geometry and timing</title><style>body{font:16px system-ui;margin:auto;max-width:1280px;padding:28px;color:#1b2636;background:#f8fafc}h1,h2{color:#17374d}a{color:#075eaa}table{border-collapse:collapse;width:100%;background:white;font-size:14px;margin:18px 0}th,td{padding:9px;border:1px solid #d4dce5;text-align:left}th{background:#e9eff5}.notice{padding:18px;background:#fff0cf;border-left:5px solid #e4a228}.good{background:#e1f4e9;padding:18px}details{border:1px solid #cad3de;background:white;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:600}img{max-width:100%;height:auto}code,pre{font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f6;padding:12px}.bar{display:flex;height:30px;width:100%;margin:12px 0}.small{color:#526274;font-size:14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:16px}button{padding:7px}input{padding:10px;width:95%;max-width:500px}</style>",
      "<h1>RoArtNet: fast hypotheses, motion verification still needed</h1>",
      "<p>25 October 2026 · pinned microwave checkpoint · Blackwell GPU · saved observations only</p>",
      f"<div class='notice'><b>38/38 baseline inferences completed.</b> Median local position error: <b>{fmt(primary['reference_plane_error_mm']['median'],1)} mm</b>; median axis error: <b>{fmt(primary['axis_angle_deg']['median'],1)}°</b>; median inference: <b>{fmt(primary['median_inference_seconds'])} s</b>. This backend has not been connected to robot control or Luna. A numeric output is not proof of a correct hinge.</div>",
      f"<p class='good'><b>Observed motion was more accurate:</b> {follow['motion_accepted']}/{follow['motion_pairs']} frame pairs passed the existing upright geometric fitter with no hinge prior. Median accepted position error was <b>{fmt(follow['motion_error_mm']['median'],1)} mm</b>. This is an exploratory result on one saved trajectory.</p>",
      "<p>RGB target box/points → MobileSAM object mask → calibrated depth cloud → SHOT features → RoArtNet → joint/interaction-point voting → metric hypothesis. The joint count and revolute type are configured. The network does not reconstruct a full mesh or verify a grasp.</p>",
      "<p>" + " · ".join([link(run / p, label, run) for p, label in [("protocol.json", "Original protocol"), ("inputs.json", "Inputs and crop provenance"), ("summary.json", "Summary JSON"), ("timing-and-accuracy.csv", "Timing/accuracy CSV"), ("benchmark.log", "Inference log"), ("source-hashes-benchmark.json", "Source hashes")]]) + "</p>",
      "<h2>Timing breakdown</h2><p>Mean over eight primary local views, including the clipped oblique view and two open-door states. GPU stages synchronize before timing. Imports, weights and segmentation are listed separately.</p><div class='bar'>"]
    keys = [k for k in stages if k != "total_inference_seconds"]
    for k, color in zip(keys, colors):
        contents.append(f"<span title='{html.escape(k)}: {stages[k]:.4f}s' style='width:{100*stages[k]/stages['total_inference_seconds']:.3f}%;background:{color}'></span>")
    contents.append("</div><table><tr><th>Stage</th><th>Mean seconds</th><th>Share</th></tr>")
    for k in keys:
        contents.append(f"<tr><td>{k.removesuffix('_seconds').replace('_',' ')}</td><td>{stages[k]:.4f}</td><td>{100*stages[k]/stages['total_inference_seconds']:.1f}%</td></tr>")
    contents.append(f"<tr><th>Total perception</th><th>{stages['total_inference_seconds']:.3f}</th><th>100%</th></tr></table>")
    local = [r for r in read(run / "inputs.json") if r["kind"] == "local" and r["variant"] == "sam"]
    audit = read(run / "input-audit.json")
    runtime = read(run / "runtime-author-eval.json")
    contents.append(f"<p>RoArtNet imports/device setup {runtime['imports_seconds']:.2f}s, weight loading {runtime['load_weights_seconds']:.3f}s. MobileSAM loading {audit['segmentation_load_seconds']:.2f}s; first segmentation {local[0]['timing']['segmentation_seconds']:.2f}s, subsequent mean {np.mean([r['timing']['segmentation_seconds'] for r in local[1:]]):.3f}s. Sensor file loading and frame checks average {np.mean([r['timing']['sensor_load_and_frame_check_seconds'] for r in local]):.3f}s. Sampled peak GPU0 allocation: {summary['gpu_memory_peak_mib']/1024:.2f} GiB across the full batch, including caches. Neural-network layers themselves average only {(stages['shot_encoder_seconds']+stages['tuple_encoder_seconds'])*1000:.1f}ms; direction voting dominates. No IK, grasp planning, physics or Luna calls occurred.</p>")
    contents.append("<h2>Accuracy and scope</h2><p>Position means axis error at the reference anchor plane; angle ignores axis sign. All other metrics are retained in JSON. Shortest distance between infinite lines can look deceptively small when axes tilt. Five closed front views are nearly identical, and all local scenes reuse one microwave asset. Training-asset overlap is unverified.</p><table><tr><th>Group</th><th>Cases</th><th>Median position mm</th><th>Worst position mm</th><th>Median axis °</th><th>Median seconds</th></tr>")
    for name, group in summary["groups"].items():
        contents.append(f"<tr><td>{name}</td><td>{group['cases']}</td><td>{fmt(group['reference_plane_error_mm']['median'],1)}</td><td>{fmt(group['reference_plane_error_mm']['max'],1)}</td><td>{fmt(group['axis_angle_deg']['median'],1)}</td><td>{fmt(group['median_inference_seconds'])}</td></tr>")
    contents.append("</table><p>The authors’ evaluation settings improve axis direction compared with the coarser saved checkpoint settings. Neither profile resolves the position bias. The six author examples are a runtime reproduction sample, not a reproduction of full-paper aggregate accuracy.</p>")
    contents.append("<h2>Why estimates go wrong</h2><ul><li>Clean frontal estimates have a repeatable position offset around 6cm; changing seed barely moves it. Correct frame round trips and faithful upstream features rule out simple axis swaps or a substituted descriptor. The exact learned-bias/domain-shift contribution remains unproven.</li><li>The oblique object is clipped by the image boundary and its mask omits some handle points; error reaches 171mm.</li><li>Dilating the clean mask by two pixels admits far background (up to3.48m versus target1.07m), expanding the cloud and changing voting. Axis error grows from1.3° to27.6°.</li><li>A largest-component cleanup also discards the recessed window surface. It improves contaminated-axis estimates but worsens clean position estimates; it is not enabled by default.</li></ul>")
    contents.append("<div class='grid'><img src='crop-contact-sheet.png' alt='All RGB-based object masks'><img src='cleanup/removed-target-points.png' alt='Valid recessed window points wrongly removed by largest-component filtering'></div>")
    contents.append("<h2>Cleanup ablation — exploratory, not a new baseline</h2><table><tr><th>Crop</th><th>Before position / axis</th><th>After position / axis</th><th>Points removed</th><th>Cleanup seconds</th></tr>")
    baseline = {r["input_id"]: r for r in rows if r["profile"] == "author-eval" and r["seed"] == 42}
    for r in cleanup:
        old, new = baseline[r["input_id"]]["errors"], r["errors"]
        contents.append(f"<tr><td>{r['input_id'].split('-')[-1]}</td><td>{fmt(old['reference_plane_error_mm'],1)}mm / {fmt(old['axis_angle_deg'],1)}°</td><td>{fmt(new['reference_plane_error_mm'],1)}mm / {fmt(new['axis_angle_deg'],1)}°</td><td>{r['cleanup']['input_points']-r['cleanup']['retained_points']}</td><td>{r['cleanup_seconds']:.3f}</td></tr>")
    contents.append("</table><h2>Motion verification — exploratory</h2><p>Forward/backward optical flow and calibrated depth generate moving correspondences; the existing robust upright SE2 fitter estimates the pivot with <code>prior=None</code>. No simulator coordinates choose the fit. Native references are read only afterward. Nonzero motion and sufficient spatial coverage are required. This did not test RoArtNet-selected physical probes.</p><table><tr><th>Pair</th><th>Result</th><th>Native movement °</th><th>Estimated movement °</th><th>Position mm</th><th>Total s</th></tr>")
    for r in motion:
        fit = r["fit"]
        contents.append(f"<tr><td>{r['id']}</td><td>{html.escape(fit['reason'])}</td><td>{r['native_rotation_deg_private']:.2f}</td><td>{fmt(np.degrees(fit['rotation_rad']) if 'rotation_rad' in fit else None)}</td><td>{fmt(r.get('errors',{}).get('reference_plane_error_mm'),1)}</td><td>{r['timing']['total_seconds']:.3f}</td></tr>")
    contents.append("</table><p>Bootstrap spread is geometric stability, not calibrated uncertainty. Motion estimates assume an upright hinge and one dominant moving rigid part; these examples do not establish transfer to arbitrary articulated objects.</p>")
    contents.append("<h2>Inspect each input and output</h2><input id='filter' placeholder='Filter by state, view, profile or seed' aria-label='Filter cases'><p class='small'>Red: predicted axis. Cyan: private evaluation reference. Reference overlays are observer-only and were never model inputs.</p>")
    for r in rows:
        im = os.path.relpath(plots[r["id"]], run)
        contents.append(f"<details class='case' data-key='{html.escape(r['id'])}'><summary>{html.escape(r['id'])} · {fmt(r['errors']['reference_plane_error_mm'],1)}mm · {fmt(r['errors']['axis_angle_deg'],1)}° · {fmt(r['prediction']['timing']['total_inference_seconds'])}s</summary><img loading='lazy' src='{im}' alt='Predicted and reference hinge geometry'><p>" + link(ROOT / r['input']['input'], "Input XYZ NPZ", run) + " · " + link(run / 'predictions' / (r['id'] + '.json'), "Raw output + stage timing", run) + "</p><pre>" + html.escape(json.dumps(r['prediction'], indent=2)) + "</pre></details>")
    contents.append("<h2>Code and reproducibility</h2><p>" + " · ".join(link(ROOT / p, label, run) for p, label in [
        ("docs/roartnet_perception_20261025.md", "Setup and conclusions"), ("src/vlm_tamp/roartnet_adapter.py", "Perception adapter"),
        ("scripts/prepare_roartnet_inputs.py", "Public input preparation"), ("scripts/run_roartnet_benchmark.py", "Benchmark runner"),
        ("scripts/evaluate_roartnet.py", "Private scorer"), ("scripts/probe_roartnet_motion.py", "Motion probe"),
        ("scripts/probe_roartnet_cleanup.py", "Cleanup probe")]) + "</p><p>" + " · ".join(link(run / p, p, run) for p in ["upstream-compatibility.patch", "pip-freeze.txt", "conda-explicit.txt", "followup-summary.json", "setup-failures/cuda-header-absolute-path.json"]) + "</p>")
    contents.append("<script>document.querySelector('#filter').addEventListener('input',e=>document.querySelectorAll('.case').forEach(x=>x.hidden=!x.dataset.key.includes(e.target.value.trim())));</script></html>")
    (run / "watch.html").write_text("\n".join(contents))
    print(run / "watch.html")


if __name__ == "__main__":
    main()
