import json

import pytest
from PIL import Image, PngImagePlugin

from png_info_parser import (
    MetadataError,
    parse_a1111_parameters,
    parse_comfy_prompt,
    parse_png,
    resolve_assets,
)


A1111_PARAMETERS = """imagers <lora:Hyperrealistic_mix:1> <lora:Realism__Semi-Realism_LoRa_Style:0.3>
Negative prompt: asdasdasdasd
Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, Seed: 2650675226, Size: 1024x1024, Model: realcartoonPony_v2"""


def comfy_prompt():
    return {
        "16": {
            "inputs": {"ckpt_name": "ILL/model.safetensors"},
            "class_type": "CheckpointLoaderSimple",
        },
        "25": {
            "inputs": {"text": "positive text", "clip": ["52", 1]},
            "class_type": "CLIPTextEncode",
        },
        "27": {
            "inputs": {"text": "negative text", "clip": ["52", 1]},
            "class_type": "CLIPTextEncode",
        },
        "52": {
            "inputs": {
                "lora_1": {"on": True, "lora": "A.safetensors", "strength": 0.3},
                "lora_2": {"on": False, "lora": "B.safetensors", "strength": 0.5},
                "lora_3": {"on": True, "lora": "C.safetensors", "strength": -2},
                "model": ["16", 0],
                "clip": ["16", 1],
            },
            "class_type": "Power Lora Loader (rgthree)",
        },
        "54:29": {
            "inputs": {
                "seed": 123456789,
                "steps": 50,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "simple",
                "denoise": 0.75,
                "model": ["52", 0],
                "positive": ["25", 0],
                "negative": ["27", 0],
                "latent_image": ["53", 0],
            },
            "class_type": "KSampler",
        },
        "31": {
            "inputs": {"samples": ["54:29", 0]},
            "class_type": "VAEDecode",
        },
        "55": {
            "inputs": {"images": ["31", 0]},
            "class_type": "SaveImage",
        },
    }


def test_parse_a1111_and_strip_lora_tags():
    info = parse_a1111_parameters(A1111_PARAMETERS)
    assert info.source_format == "A1111"
    assert info.seed == 2650675226
    assert info.steps == 20
    assert info.cfg == 7.0
    assert info.sampler_name == "dpmpp_2m"
    assert info.scheduler == "karras"
    assert info.denoise == 1.0
    assert info.ckpt_name == "realcartoonPony_v2"
    assert info.positive_prompt == "imagers"
    assert info.negative_prompt == "asdasdasdasd"
    assert info.loras == [
        {"name": "Hyperrealistic_mix", "strength": 1.0},
        {"name": "Realism__Semi-Realism_LoRa_Style", "strength": 0.3},
    ]


def test_parse_comfy_graph_and_ignore_disabled_lora():
    info = parse_comfy_prompt(json.dumps(comfy_prompt()))
    assert info.source_format == "ComfyUI"
    assert info.seed == 123456789
    assert info.steps == 50
    assert info.sampler_name == "euler_ancestral"
    assert info.scheduler == "simple"
    assert info.denoise == 0.75
    assert info.ckpt_name == "ILL/model.safetensors"
    assert info.positive_prompt == "positive text"
    assert info.negative_prompt == "negative text"
    assert info.loras == [
        {"name": "A.safetensors", "strength": 0.3},
        {"name": "C.safetensors", "strength": -2.0},
    ]


def test_parse_png_prefers_comfy_metadata(tmp_path):
    png = tmp_path / "comfy.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", json.dumps(comfy_prompt()))
    Image.new("RGB", (1, 1)).save(png, pnginfo=metadata)
    assert parse_png(png).source_format == "ComfyUI"


def test_resolve_assets_by_unique_stem_and_skip_missing():
    info = parse_a1111_parameters(A1111_PARAMETERS)
    resolved = resolve_assets(
        info,
        ["PONY/realcartoonPony_v2.safetensors"],
        ["Hyperrealistic_mix.safetensors"],
    )
    assert resolved.ckpt_name == "PONY/realcartoonPony_v2.safetensors"
    assert resolved.loras[0]["name"] == "Hyperrealistic_mix.safetensors"
    assert resolved.loras[1]["name"] == "None"
    assert any("LoRA not installed" in warning for warning in resolved.warnings)


def test_only_first_four_loras_are_used():
    parameters = (
        "prompt "
        + " ".join(f"<lora:Lora{i}:{i / 10}>" for i in range(1, 6))
        + "\nSteps: 10, Sampler: Euler, CFG scale: 5, Seed: 1"
    )
    info = parse_a1111_parameters(parameters)
    assert len(info.loras) == 4
    assert any("only the first 4" in warning for warning in info.warnings)


def test_unsupported_png_raises(tmp_path):
    png = tmp_path / "plain.png"
    Image.new("RGB", (1, 1)).save(png)
    with pytest.raises(MetadataError):
        parse_png(png)
