"""PNG Info for ComfyUI custom node package."""

WEB_DIRECTORY = "./web"

# ComfyUI loads custom-node folders as packages. Pytest may import this file as
# a top-level ``__init__`` module when the checkout directory contains spaces;
# keep that collection-time import free from ComfyUI-only dependencies.
if __package__:
    from .png_info_node import PNGInfoForComfyUI
    from . import routes as _routes  # noqa: F401 - registers the HTTP route

    NODE_CLASS_MAPPINGS = {"PNGInfoForComfyUI": PNGInfoForComfyUI}
    NODE_DISPLAY_NAME_MAPPINGS = {"PNGInfoForComfyUI": "PNG Info for ComfyUI"}
else:  # pragma: no cover - only used by external test collectors
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
