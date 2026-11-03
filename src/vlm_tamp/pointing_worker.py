"""Local JSON transport for one Molmo2 model shared by an experiment batch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time
from typing import Any


class PointingClient:
    """Connect to the explicitly configured worker; never start a replacement."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = str(socket_path)

    def point(self, image_paths: list[Path], prompt: str) -> dict[str, Any]:
        """Return the worker response without selecting or repairing a point."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(600)
            connection.connect(self.socket_path)
            with connection.makefile('rwb') as stream:
                request = {'images': [str(path) for path in image_paths], 'prompt': prompt}
                stream.write(json.dumps(request).encode() + b'\n')
                stream.flush()
                raw = stream.readline()
                if not raw:
                    raise RuntimeError('Molmo2 worker closed without a reply; inspect its process log')
                return json.loads(raw)


def serve(socket_path: Path, ready_file: Path, checkpoint: Path) -> None:
    """Load once, then handle independent image-only requests sequentially."""
    from .pointing_molmo2 import Molmo2Pointer

    started = time.perf_counter()
    pointer = Molmo2Pointer(checkpoint)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen()
        metadata = {'backend': 'allenai/Molmo2-8B', 'pid': os.getpid(),
                    'model_load_seconds': pointer.load_seconds,
                    'startup_seconds': time.perf_counter() - started,
                    'checkpoint': str(checkpoint), 'dtype': 'bfloat16',
                    'ready': True, 'socket': str(socket_path)}
        ready_file.write_text(json.dumps(metadata, indent=2) + '\n')
        sequence = 0
        while True:
            connection, _ = server.accept()
            with connection, connection.makefile('rwb') as stream:
                request = json.loads(stream.readline())
                sequence += 1
                result = pointer.point([Path(p) for p in request['images']], request['prompt'])
                result.update(backend=metadata['backend'], worker_pid=os.getpid(),
                              worker_request_sequence=sequence)
                stream.write(json.dumps(result).encode() + b'\n')
                stream.flush()


def main() -> None:
    """Start only the requested local checkpoint and endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    serve(args.socket, args.ready_file, args.checkpoint)


if __name__ == '__main__':
    main()
