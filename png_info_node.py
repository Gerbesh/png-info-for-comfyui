"""ComfyUI backend node for PNG generation metadata."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import comfy.samplers
import folder_paths

from .png_info_parser import MetadataError, parse_png, resolve_assets


def _image_files() -> list[str]:
    input_dir = folder_paths.get_input_directory()
    files = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    return [
        name
        for name in folder_paths.filter_files_content_types(files, ["image"])
        if name.casefold().endswith(".png")
    ]


def safe_png_path(image: str) -> str:
    """Resolve an annotated ComfyUI path without allowing directory traversal."""
    name, base_dir = folder_paths.annotated_filepath(image)
    if base_dir is None:
        base_dir = folder_paths.get_input_directory()
    root = Path(base_dir).resolve()
    candidate = (root / name).resolve()
    if candidate.suffix.casefold() != ".png" or not candidate.is_relative_to(root):
        raise ValueError("Invalid PNG path")
    if not candidate.is_file():
        raise ValueError("PNG file does not exist")
    return str(candidate)


def _checkpoint_choices() -> list[str]:
    return folder_paths.get_filename_list("checkpoints")


def _lora_choices() -> list[str]:
    return ["None", *folder_paths.get_filename_list("loras")]


class PNGInfoForComfyUI:
    """Extract A1111 or ComfyUI settings from an uploaded PNG."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (sorted(_image_files()), {"image_upload": True}),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "FLOAT",
        _checkpoint_choices(),
        _lora_choices(),
        "FLOAT",
        _lora_choices(),
        "FLOAT",
        _lora_choices(),
        "FLOAT",
        _lora_choices(),
        "FLOAT",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "ckpt_name",
        "lora_01",
        "strength_01",
        "lora_02",
        "strength_02",
        "lora_03",
        "strength_03",
        "lora_04",
        "strength_04",
        "positive_prompt",
        "negative_prompt",
        "status",
    )
    FUNCTION = "extract"
    CATEGORY = "PNG Info"
    DESCRIPTION = (
        "Reads generation settings from A1111 or ComfyUI PNG metadata. "
        "Use the frontend Apply button to copy them into a selected KSampler branch."
    )

    def extract(self, image: str):
        image_path = safe_png_path(image)
        try:
            info = resolve_assets(
                parse_png(image_path),
                _checkpoint_choices(),
                folder_paths.get_filename_list("loras"),
            )
            status = _status_text(info.source_format, info.warnings)
        except MetadataError as error:
            raise ValueError(f"PNG Info: {error}") from error

        loras = [*info.loras]
        while len(loras) < 4:
            loras.append({"name": "None", "strength": 1.0})

        return (
            info.seed,
            info.steps,
            info.cfg,
            info.sampler_name,
            info.scheduler,
            info.denoise,
            info.ckpt_name,
            loras[0]["name"],
            loras[0]["strength"],
            loras[1]["name"],
            loras[1]["strength"],
            loras[2]["name"],
            loras[2]["strength"],
            loras[3]["name"],
            loras[3]["strength"],
            info.positive_prompt,
            info.negative_prompt,
            status,
        )

    @classmethod
    def IS_CHANGED(cls, image: str):
        image_path = safe_png_path(image)
        digest = hashlib.sha256()
        with open(image_path, "rb") as file:
            digest.update(file.read())
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image: str):
        try:
            safe_png_path(image)
        except ValueError:
            return f"Invalid image file: {image}"
        return True


def _status_text(source_format: str, warnings: list[str]) -> str:
    if warnings:
        return f"{source_format}: " + " | ".join(warnings)
    return f"{source_format}: metadata parsed successfully"
