"""Explicit Molmo2 multi-image pointing for recorded-image experiments.

This module does not choose a robot action, change a contact, or infer depth.
Coordinates follow the official Molmo2 model card: one-based image indices and
image coordinates scaled by 1000. Model-format failures remain visible.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import time
from typing import Any


def parse_multi_image_points(text: str, sizes: list[tuple[int, int]]) -> dict[str, Any]:
    """Decode native point tags without clipping, repairing, or dropping points."""
    tags = re.findall(r'<points\b[^>]*>', text)
    if not tags:
        return {'status': 'no_point_tag', 'points': [], 'raw_text': text}
    points = []
    for tag in tags:
        coordinates = re.search(r'\bcoords="([^"]*)"', tag)
        if coordinates is None:
            return {'status': 'invalid_point_format', 'points': [], 'raw_text': text}
        for group in re.split(r'[\t:;,]', coordinates.group(1)):
            tokens = group.split()
            if len(tokens) < 4 or len(tokens) % 3 != 1 or any(
                re.fullmatch(r'\d+(?:\.\d+)?', token) is None for token in tokens
            ):
                return {'status': 'invalid_point_format', 'points': [], 'raw_text': text}
            frame = float(tokens[0])
            if not frame.is_integer() or not 1 <= frame <= len(sizes):
                return {'status': 'invalid_image_index', 'points': [], 'raw_text': text}
            width, height = sizes[int(frame) - 1]
            for offset in range(1, len(tokens), 3):
                identity, x, y = map(float, tokens[offset:offset + 3])
                if not identity.is_integer() or not all(math.isfinite(v) for v in [x, y]):
                    return {'status': 'invalid_point_format', 'points': [], 'raw_text': text}
                if not 0 <= x <= 1000 or not 0 <= y <= 1000:
                    return {'status': 'coordinates_out_of_range', 'points': [], 'raw_text': text}
                points.append({'image_index': int(frame), 'point_id': int(identity),
                               'coordinates_1000': [x, y],
                               'pixel_xy': [x * width / 1000, y * height / 1000]})
    return {'status': 'parsed', 'points': points, 'raw_text': text}


class Molmo2Pointer:
    """Keep one explicitly selected checkpoint on a specified CUDA device."""

    def __init__(self, checkpoint: Path, device: str = 'cuda:0') -> None:
        """Load the reviewed local checkpoint in BF16, without a device fallback."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, device_map={'': device}, low_cpu_mem_usage=True)
        self.model.eval()
        torch.cuda.synchronize(self.model.device)
        self.load_seconds = time.perf_counter() - started

    def point(self, image_paths: list[Path], prompt: str) -> dict[str, Any]:
        """Run a fresh greedy request, retaining raw output and timing evidence."""
        import torch
        from PIL import Image

        started = time.perf_counter()
        images = [Image.open(path).convert('RGB') for path in image_paths]
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            *[{'type': 'image', 'image': image} for image in images],
        ]}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors='pt', return_dict=True)
        shapes = {key: list(value.shape) for key, value in inputs.items()}
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        torch.cuda.synchronize(self.model.device)
        prepared = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        torch.cuda.synchronize(self.model.device)
        finished = time.perf_counter()
        output = generated[0, inputs['input_ids'].shape[1]:]
        raw = self.processor.tokenizer.decode(output, skip_special_tokens=True)
        result = parse_multi_image_points(raw, [image.size for image in images])
        result.update(
            prompt=prompt, rendered_prompt=rendered,
            preprocessing_seconds=prepared - started,
            generation_seconds=finished - prepared, total_seconds=finished - started,
            input_tokens=inputs['input_ids'].shape[1], output_tokens=len(output),
            input_shapes=shapes, ended_at_token_limit=len(output) == 512,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(self.model.device),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(self.model.device),
        )
        return result
