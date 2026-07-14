"""Metadata parsers for Automatic1111 and ComfyUI PNG files."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image


class MetadataError(ValueError):
    """Raised when a PNG has no supported generation metadata."""


@dataclass
class GenerationInfo:
    source_format: str
    seed: int = 0
    steps: int = 20
    cfg: float = 7.0
    sampler_name: str = "euler"
    scheduler: str = "normal"
    denoise: float = 1.0
    ckpt_name: str = ""
    loras: list[dict[str, Any]] = field(default_factory=list)
    positive_prompt: str = ""
    negative_prompt: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "seed": self.seed,
            "steps": self.steps,
            "cfg": self.cfg,
            "sampler_name": self.sampler_name,
            "scheduler": self.scheduler,
            "denoise": self.denoise,
            "ckpt_name": self.ckpt_name,
            "loras": self.loras,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "warnings": self.warnings,
        }


_LORA_TAG = re.compile(
    r"<lora:([^:>]+)(?::([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?>",
    flags=re.IGNORECASE,
)

_A1111_SAMPLERS = {
    "euler": "euler",
    "euler a": "euler_ancestral",
    "heun": "heun",
    "lms": "lms",
    "dpm2": "dpm_2",
    "dpm2 a": "dpm_2_ancestral",
    "dpm++ 2s a": "dpmpp_2s_ancestral",
    "dpm++ 2m": "dpmpp_2m",
    "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm++ sde": "dpmpp_sde",
    "dpm fast": "dpm_fast",
    "dpm adaptive": "dpm_adaptive",
    "ddim": "ddim",
    "uni pc": "uni_pc",
    "uni pc bh2": "uni_pc_bh2",
    "lcm": "lcm",
}

_SCHEDULERS = {
    "automatic": "normal",
    "normal": "normal",
    "uniform": "normal",
    "karras": "karras",
    "exponential": "exponential",
    "polyexponential": "polyexponential",
    "sgm uniform": "sgm_uniform",
    "simple": "simple",
    "ddim": "ddim_uniform",
    "ddim uniform": "ddim_uniform",
    "beta": "beta",
}


def read_png_metadata(path: str | Path) -> dict[str, str]:
    """Read textual PNG chunks without decoding image pixels."""
    with Image.open(path) as image:
        metadata: dict[str, str] = {}
        for key, value in {**image.info, **getattr(image, "text", {})}.items():
            if isinstance(value, str):
                metadata[key] = value
        return metadata


def parse_png(path: str | Path) -> GenerationInfo:
    metadata = read_png_metadata(path)
    if metadata.get("prompt"):
        try:
            return parse_comfy_prompt(metadata["prompt"])
        except (json.JSONDecodeError, MetadataError, TypeError, ValueError) as error:
            if not metadata.get("parameters"):
                raise MetadataError(f"Invalid ComfyUI prompt metadata: {error}") from error
    if metadata.get("parameters"):
        return parse_a1111_parameters(metadata["parameters"])
    raise MetadataError("No supported A1111 or ComfyUI metadata found")


def parse_a1111_parameters(parameters: str) -> GenerationInfo:
    match = re.search(r"(?:^|\n)Steps:\s*", parameters)
    if not match:
        raise MetadataError("A1111 metadata has no Steps field")

    prompt_block = parameters[: match.start()].rstrip()
    options_line = parameters[match.start() :].lstrip("\n")
    negative_marker = "\nNegative prompt:"
    if negative_marker in prompt_block:
        positive, negative = prompt_block.split(negative_marker, 1)
        negative = negative.strip()
    else:
        positive, negative = prompt_block, ""

    loras: list[dict[str, Any]] = []
    for lora_match in _LORA_TAG.finditer(positive):
        loras.append(
            {
                "name": lora_match.group(1).strip(),
                "strength": float(lora_match.group(2) or 1.0),
            }
        )
    positive = _clean_prompt(_LORA_TAG.sub("", positive))

    fields = _parse_a1111_fields(options_line)
    sampler_raw = fields.get("sampler", "Euler")
    schedule_raw = fields.get("schedule type", "")
    sampler_name, scheduler = _normalise_a1111_sampler(sampler_raw, schedule_raw)

    warnings: list[str] = []
    if sampler_name == sampler_raw:
        warnings.append(f"Unknown A1111 sampler: {sampler_raw}")

    return _limit_loras(
        GenerationInfo(
            source_format="A1111",
            seed=_to_int(fields.get("seed"), 0),
            steps=_to_int(fields.get("steps"), 20),
            cfg=_to_float(fields.get("cfg scale"), 7.0),
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=_to_float(fields.get("denoising strength"), 1.0),
            ckpt_name=fields.get("model", "").strip(),
            loras=loras,
            positive_prompt=positive,
            negative_prompt=negative,
            warnings=warnings,
        )
    )


def parse_comfy_prompt(prompt_text: str) -> GenerationInfo:
    prompt = json.loads(prompt_text)
    if not isinstance(prompt, dict):
        raise MetadataError("ComfyUI prompt is not an object")

    candidates = [
        (node_id, node)
        for node_id, node in prompt.items()
        if isinstance(node, dict) and _is_sampler_node(node)
    ]
    if not candidates:
        raise MetadataError("No supported KSampler found in ComfyUI prompt")

    selected_id, sampler = _select_sampler(prompt, candidates)
    inputs = sampler.get("inputs", {})
    warnings: list[str] = []
    if len(candidates) > 1:
        warnings.append(
            f"Found {len(candidates)} sampler branches; selected node {selected_id}"
        )

    ckpt_name, loras, model_warnings = _parse_model_path(
        prompt, _linked_node_id(inputs.get("model"))
    )
    warnings.extend(model_warnings)

    positive = _find_text_upstream(
        prompt, _linked_node_id(inputs.get("positive"))
    )
    negative = _find_text_upstream(
        prompt, _linked_node_id(inputs.get("negative"))
    )

    return _limit_loras(
        GenerationInfo(
            source_format="ComfyUI",
            seed=_to_int(inputs.get("seed", inputs.get("noise_seed")), 0),
            steps=_to_int(inputs.get("steps"), 20),
            cfg=_to_float(inputs.get("cfg"), 7.0),
            sampler_name=str(inputs.get("sampler_name", "euler")),
            scheduler=str(inputs.get("scheduler", "normal")),
            denoise=_to_float(inputs.get("denoise"), 1.0),
            ckpt_name=ckpt_name,
            loras=loras,
            positive_prompt=positive,
            negative_prompt=negative,
            warnings=warnings,
        )
    )


def resolve_assets(
    info: GenerationInfo,
    checkpoints: Iterable[str],
    available_loras: Iterable[str],
) -> GenerationInfo:
    """Resolve metadata names to unique installed ComfyUI relative paths."""
    result = GenerationInfo(**info.as_dict())
    result.warnings = list(info.warnings)
    result.loras = [dict(item) for item in info.loras]

    if info.ckpt_name:
        resolved, ambiguous = _resolve_asset_name(info.ckpt_name, checkpoints)
        if resolved:
            result.ckpt_name = resolved
        else:
            result.ckpt_name = ""
            reason = "ambiguous" if ambiguous else "not installed"
            result.warnings.append(f"Checkpoint {reason}: {info.ckpt_name}")

    for lora in result.loras:
        original = str(lora.get("name", ""))
        resolved, ambiguous = _resolve_asset_name(original, available_loras)
        if resolved:
            lora["name"] = resolved
        else:
            lora["name"] = "None"
            reason = "ambiguous" if ambiguous else "not installed"
            result.warnings.append(f"LoRA {reason}: {original}")
    return result


def _parse_a1111_fields(options_line: str) -> dict[str, str]:
    reader = csv.reader(io.StringIO(options_line), skipinitialspace=True)
    fields: dict[str, str] = {}
    for item in next(reader, []):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        fields[key.strip().casefold()] = value.strip()
    return fields


def _normalise_a1111_sampler(sampler: str, schedule: str) -> tuple[str, str]:
    sampler_key = re.sub(r"\s+", " ", sampler.strip()).casefold()
    inferred_schedule = schedule.strip()
    for suffix, value in (
        (" karras", "Karras"),
        (" exponential", "Exponential"),
        (" sgm uniform", "SGM Uniform"),
    ):
        if not inferred_schedule and sampler_key.endswith(suffix):
            sampler_key = sampler_key[: -len(suffix)]
            inferred_schedule = value
            break
    return (
        _A1111_SAMPLERS.get(sampler_key, sampler.strip()),
        _SCHEDULERS.get(inferred_schedule.casefold(), "normal"),
    )


def _is_sampler_node(node: dict[str, Any]) -> bool:
    node_type = str(node.get("class_type", ""))
    inputs = node.get("inputs", {})
    return (
        node_type in {"KSampler", "KSamplerAdvanced"}
        or all(key in inputs for key in ("steps", "cfg", "sampler_name", "scheduler"))
    )


def _select_sampler(
    prompt: dict[str, Any], candidates: list[tuple[str, dict[str, Any]]]
) -> tuple[str, dict[str, Any]]:
    candidate_ids = {str(node_id) for node_id, _ in candidates}
    terminal_ids = [
        str(node_id)
        for node_id, node in prompt.items()
        if str(node.get("class_type", ""))
        in {"SaveImage", "PreviewImage", "SaveAnimatedWEBP", "SaveAnimatedPNG"}
    ]
    for terminal_id in sorted(terminal_ids, key=_natural_key, reverse=True):
        found = _nearest_upstream(prompt, terminal_id, candidate_ids)
        if found:
            return found, prompt[found]
    return sorted(candidates, key=lambda item: _natural_key(str(item[0])))[-1]


def _nearest_upstream(
    prompt: dict[str, Any], start_id: str, wanted: set[str]
) -> str | None:
    queue = [start_id]
    visited: set[str] = set()
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in wanted:
            return node_id
        node = prompt.get(node_id, {})
        for value in node.get("inputs", {}).values():
            linked = _linked_node_id(value)
            if linked:
                queue.append(linked)
    return None


def _parse_model_path(
    prompt: dict[str, Any], start_id: str | None
) -> tuple[str, list[dict[str, Any]], list[str]]:
    ckpt_name = ""
    loras: list[dict[str, Any]] = []
    warnings: list[str] = []
    visited: set[str] = set()

    def walk(node_id: str | None) -> None:
        nonlocal ckpt_name
        if not node_id or node_id in visited:
            return
        visited.add(node_id)
        node = prompt.get(node_id, {})
        inputs = node.get("inputs", {})
        node_type = str(node.get("class_type", ""))

        base_id = _linked_node_id(inputs.get("model"))
        if base_id:
            walk(base_id)

        if "ckpt_name" in inputs and not ckpt_name:
            ckpt_name = str(inputs["ckpt_name"])

        if node_type in {"LoraLoader", "LoraLoaderModelOnly"} or "lora_name" in inputs:
            name = str(inputs.get("lora_name", "None"))
            strength_model = _to_float(inputs.get("strength_model"), 1.0)
            strength_clip = _to_float(inputs.get("strength_clip"), strength_model)
            if strength_clip != strength_model:
                warnings.append(
                    f"LoRA {name} has different model/CLIP strengths; using model strength"
                )
            if name != "None" and strength_model != 0:
                loras.append({"name": name, "strength": strength_model})
        elif "Lora Loader Stack" in node_type:
            for index in range(1, 5):
                name = str(inputs.get(f"lora_{index:02d}", "None"))
                strength = _to_float(inputs.get(f"strength_{index:02d}"), 1.0)
                if name != "None" and strength != 0:
                    loras.append({"name": name, "strength": strength})
        elif "Power Lora Loader" in node_type:
            for key in sorted(inputs, key=_natural_key):
                value = inputs[key]
                if not key.casefold().startswith("lora_") or not isinstance(value, dict):
                    continue
                if value.get("on") and value.get("lora") != "None":
                    strength = _to_float(value.get("strength"), 1.0)
                    if strength != 0:
                        loras.append({"name": str(value["lora"]), "strength": strength})

        if not base_id:
            for value in inputs.values():
                linked = _linked_node_id(value)
                if linked:
                    walk(linked)

    walk(start_id)
    return ckpt_name, loras, warnings


def _find_text_upstream(prompt: dict[str, Any], start_id: str | None) -> str:
    if not start_id:
        return ""
    queue = [start_id]
    visited: set[str] = set()
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        node = prompt.get(node_id, {})
        inputs = node.get("inputs", {})
        if "text" in inputs and isinstance(inputs["text"], str):
            return inputs["text"]
        for value in inputs.values():
            linked = _linked_node_id(value)
            if linked:
                queue.append(linked)
    return ""


def _resolve_asset_name(name: str, available: Iterable[str]) -> tuple[str, bool]:
    values = list(available)
    wanted = _normalise_path(name)
    wanted_path = PurePosixPath(wanted)

    strategies = (
        lambda value: _normalise_path(value) == wanted,
        lambda value: PurePosixPath(_normalise_path(value)).name == wanted_path.name,
        lambda value: PurePosixPath(_normalise_path(value)).stem == wanted_path.stem,
    )
    for predicate in strategies:
        matches = [value for value in values if predicate(value)]
        if len(matches) == 1:
            return matches[0], False
        if len(matches) > 1:
            return "", True
    return "", False


def _normalise_path(value: str) -> str:
    return value.strip().replace("\\", "/").casefold()


def _limit_loras(info: GenerationInfo) -> GenerationInfo:
    if len(info.loras) > 4:
        info.warnings.append(
            f"Found {len(info.loras)} active LoRAs; only the first 4 are used"
        )
        info.loras = info.loras[:4]
    return info


def _linked_node_id(value: Any) -> str | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
    ):
        return str(value[0])
    return None


def _clean_prompt(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
