"""HTTP endpoint used by the ComfyUI frontend extension."""

from __future__ import annotations

from aiohttp import web
from server import PromptServer

import folder_paths

from .png_info_parser import MetadataError, parse_png, resolve_assets
from .png_info_node import safe_png_path


@PromptServer.instance.routes.get("/png-info-for-comfyui/parse")
async def parse_png_route(request):
    image = request.rel_url.query.get("image", "")
    try:
        image_path = safe_png_path(image)
    except ValueError:
        return web.json_response({"error": "Invalid image file"}, status=400)

    try:
        info = resolve_assets(
            parse_png(image_path),
            folder_paths.get_filename_list("checkpoints"),
            folder_paths.get_filename_list("loras"),
        )
    except MetadataError as error:
        return web.json_response({"error": str(error)}, status=422)
    except Exception as error:  # Keep malformed files from breaking the UI request.
        return web.json_response({"error": f"Could not read PNG metadata: {error}"}, status=422)

    return web.json_response(info.as_dict())
