# PNG Info for ComfyUI

A ComfyUI custom node that reads generation settings from PNG files created by
Automatic1111 or ComfyUI and reuses them in an editable workflow.

## Features

- Parses the A1111 `parameters` text chunk and ComfyUI `prompt` graph.
- Extracts seed, steps, CFG, sampler, scheduler, denoise, checkpoint, positive
  and negative prompts.
- Extracts up to four enabled LoRAs and their strengths.
- Resolves checkpoint and LoRA names against files installed in ComfyUI.
- Provides typed outputs compatible with core KSampler/Checkpoint nodes,
  `Lora Loader Stack (rgthree)`, and `Power Lora Loader (rgthree)`.
- Remembers a complete five-node target set, removes the temporary data links,
  and keeps every destination widget editable.

## Installation

From Comfy Registry / Manager:

```powershell
comfy node install png-info
```

Or clone the repository manually:

Clone the repository into the `custom_nodes` folder of your ComfyUI installation:

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/Gerbesh/png-info-for-comfyui.git
```

Restart ComfyUI, then add **PNG Info for ComfyUI** from the `PNG Info` category.

## Usage

1. Add a `PNG Info for ComfyUI` node and choose or upload a PNG.
2. Build a normal generation branch containing a core `KSampler`,
   `CheckpointLoaderSimple`, positive/negative `CLIPTextEncode` nodes, and an
   rgthree LoRA loader.
3. For `Lora Loader Stack (rgthree)`, temporarily connect all 17 value outputs:
   six sampler values, checkpoint, four LoRA/strength pairs, and two prompts.
   For `Power Lora Loader (rgthree)`, leave the eight LoRA outputs disconnected;
   the plugin finds the loader connected to the bound KSampler/CLIP branch.
4. Click **Bind all connected nodes & detach**. The node stores the five target
   node IDs in the workflow, removes all temporary links and immediately applies
   the current PNG metadata.
5. Edit any populated widget normally. For another PNG, choose the file and
   click **Apply to bound nodes**; no reconnection is needed.

When applying to Power Lora Loader, the first four rows are populated and
enabled from the PNG metadata. Missing and additional existing rows are disabled
so stale LoRAs do not affect the generation. Separate model/CLIP strength mode
receives the same parsed strength for both values.

Use **Clear binding** before wiring a different target set. If a bound target
node is deleted, Apply reports its type and ID instead of writing elsewhere.
Saved bindings that still point to a removed rgthree LoRA Stack are automatically
migrated to the Power Lora Loader connected to the bound generation branch.

The node also exposes live outputs:

`seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`, `ckpt_name`,
four `lora_XX` / `strength_XX` pairs, `positive_prompt`, `negative_prompt`, and
`status`.

### Matching and fallback behavior

- A1111 `<lora:name:weight>` tags are removed from the positive prompt and
  emitted as separate LoRA values.
- Asset matching uses exact relative path, filename, then unique filename stem;
  it never performs fuzzy substitution.
- A missing checkpoint leaves the target loader unchanged. A missing LoRA is
  set to `None`. Both cases are shown in the node status.
- If more than four enabled LoRAs are found, the first four in model-application
  order are used and a warning is shown.
- For PNGs with multiple sampler branches, the parser selects a branch connected
  to a save/preview output and reports the choice.

## Development

```powershell
python -m pytest
node --check web\png_info.js
```

The test suite creates tiny synthetic PNG files; source images are not committed.

## License

This project is licensed under the [MIT License](LICENSE).
