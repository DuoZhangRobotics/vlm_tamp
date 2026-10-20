"""Install the pinned, inference-only ReKep/GraspGen worker without altering LIBERO."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CODE_REVISION = "2dd8852e1be60f5f9d277fafcc621835cdf59110"
MODEL_REVISION = "ec1ccbb5eec0680db669246ac312a3636f16ee43"
WEIGHTS = {
    "graspgen_franka_panda_gen.pth": "0597583b89b322d42ceb4e596967d6ed68d1b56cba4039895909ccd5bdc66eff",
    "graspgen_franka_panda_dis.pth": "e47d703c63b54c2d11fbc1effd43898f251b4147250888541e3b16e9c0d19e1c",
    "graspgen_franka_panda.yml": "3b666d28ffb91001ddb6ba24a2e0c11458478a986b808b493cf6fa9a987c2abd",
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", default="/home/duo/miniforge3/bin/conda")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    prefix = ROOT / ".runtime/envs/rekep-libero"
    python = prefix / "bin/python"
    output = ROOT / "results/rekep-libero-setup"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_HOME=str(prefix), CXX="/usr/bin/g++", CC="/usr/bin/gcc",
               TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="4", PYOPENGL_PLATFORM="egl")
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")

    def run(name, command, cwd=ROOT):
        print(name, flush=True)
        with (output / (name + ".log")).open("a") as log:
            subprocess.run([str(x) for x in command], cwd=cwd, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    repo = ROOT / "third_party/GraspGen"
    if not args.verify_only:
        if not repo.exists():
            run("clone-graspgen", ["git", "clone", "https://github.com/NVlabs/GraspGen.git", repo])
        run("pin-graspgen", ["git", "-C", repo, "checkout", "--detach", CODE_REVISION])
        if not python.exists():
            run("create-env", [args.conda, "create", "-y", "-p", prefix, "python=3.10", "pip"])
        if not (prefix / "bin/nvcc").exists():
            run("cuda-toolchain", [args.conda, "install", "-y", "-p", prefix,
                "-c", "nvidia/label/cuda-12.1.1", "cuda-nvcc=12.1", "cuda-cudart-dev=12.1",
                "cuda-driver-dev=12.1", "cuda-cccl=12.1"])
        if not (prefix / "include/cusparse.h").exists():
            run("cuda-library-headers", [args.conda, "install", "-y", "-p", prefix,
                "-c", "nvidia/label/cuda-12.1.1", "nvidia/label/cuda-12.1.1::libcusparse-dev",
                "nvidia/label/cuda-12.1.1::libcublas-dev", "nvidia/label/cuda-12.1.1::libcurand-dev",
                "nvidia/label/cuda-12.1.1::libcusolver-dev"])
        run("install-torch", [python, "-m", "pip", "install", "--cache-dir", ROOT / ".runtime/pip-cache",
             "torch==2.1.0", "torchvision==0.16.0", "--index-url", "https://download.pytorch.org/whl/cu121"])
        run("install-dependencies", [python, "-m", "pip", "install", "--cache-dir", ROOT / ".runtime/pip-cache",
             "-r", ROOT / "requirements-rekep-libero.txt"])
        run("install-pyg-operators", [python, "-m", "pip", "install", "--no-index",
             "-f", "https://data.pyg.org/whl/torch-2.1.0+cu121.html", "torch-scatter==2.1.2", "torch-cluster==1.6.3"])
        run("install-pointnet", [python, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
             repo / "pointnet2_ops"])
        run("install-graspgen", [python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", repo])
        run("install-mobilesam", [python, "-m", "pip", "install", "--no-deps", "-e", ROOT / "third_party/MobileSAM"])
    folder = ROOT / ".runtime/graspgen-models/checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in WEIGHTS.items():
        destination = folder / name
        if not destination.exists() and not args.verify_only:
            url = f"https://huggingface.co/adithyamurali/GraspGenModels/resolve/{MODEL_REVISION}/checkpoints/{name}"
            with urllib.request.urlopen(url, timeout=120) as response, destination.with_suffix(".partial").open("wb") as target:
                while chunk := response.read(4 * 1024 * 1024):
                    target.write(chunk)
            destination.with_suffix(".partial").rename(destination)
        if not destination.exists() or sha(destination) != expected:
            raise RuntimeError("Missing or incorrect grasp checkpoint: " + name)
    actual_revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_revision != CODE_REVISION:
        raise RuntimeError("GraspGen source revision changed")
    run("verify-imports", [python, "-c", "import torch,numpy,pointnet2_ops,torch_scatter,torch_cluster; "
         "from grasp_gen.grasp_server import GraspGenSampler; "
         "assert torch.cuda.is_available(); print(torch.__version__,numpy.__version__,torch.cuda.get_device_name())"])
    installed = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (output / "requirements-installed.txt").write_text(installed)
    manifest = {"graspgen_revision": actual_revision, "model_revision": MODEL_REVISION,
                "weights_sha256": WEIGHTS, "python": str(python), "cuda_arch": "8.9",
                "inference_only": True, "pickle5_omitted": "Upstream already falls back to Python's built-in pickle",
                "original_libero_environment_modified": False}
    (output / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
