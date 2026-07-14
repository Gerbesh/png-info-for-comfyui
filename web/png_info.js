import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "gerbesh.png-info-for-comfyui";
const NODE_NAME = "PNGInfoForComfyUI";
const BINDING_PROPERTY = "png_info_binding";
const POWER_LORA_LOADER_TYPE = "Power Lora Loader (rgthree)";
const LORA_OUTPUT_PATTERN = /^(lora|strength)_\d{2}$/;
const REQUIRED_OUTPUTS = [
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
];

function findWidget(node, name) {
  return node?.widgets?.find((widget) => widget.name === name) ?? null;
}

function setWidget(node, name, value, warnings) {
  const widget = findWidget(node, name);
  if (!widget) {
    warnings.push(`${node.title || node.type}: widget '${name}' not found`);
    return false;
  }

  const choices = widget.options?.values;
  if (Array.isArray(choices) && !choices.includes(value)) {
    warnings.push(`${node.title || node.type}: '${value}' is not available for ${name}`);
    return false;
  }

  widget.value = value;
  widget.callback?.(value);
  node.setDirtyCanvas?.(true, true);
  return true;
}

function bindingValue(data, outputName) {
  const loraMatch = outputName.match(/^(lora|strength)_(\d{2})$/);
  if (loraMatch) {
    const index = Number(loraMatch[2]) - 1;
    const lora = data.loras[index] || { name: "None", strength: 1.0 };
    return loraMatch[1] === "lora" ? lora.name || "None" : Number(lora.strength ?? 1.0);
  }
  return data[outputName];
}

function isLoraOutput(outputName) {
  return LORA_OUTPUT_PATTERN.test(outputName);
}

function connectedNodeIds(node) {
  const ids = new Set();
  for (const input of node?.inputs || []) {
    const link = input.link != null ? app.graph.links[input.link] : null;
    if (link) ids.add(link.origin_id);
  }
  for (const output of node?.outputs || []) {
    for (const linkId of output.links || []) {
      const link = app.graph.links[linkId];
      if (link) ids.add(link.target_id);
    }
  }
  return ids;
}

function findPowerLoraTarget(records) {
  const candidates = (app.graph?._nodes || []).filter(
    (candidate) => candidate.type === POWER_LORA_LOADER_TYPE,
  );
  if (!candidates.length) return null;

  const branchIds = new Set(
    ["seed", "positive_prompt", "negative_prompt"]
      .map((name) => records[name]?.node_id)
      .filter((id) => id != null),
  );
  const branchMatches = candidates.filter((candidate) =>
    [...connectedNodeIds(candidate)].some((id) => branchIds.has(id)),
  );
  if (branchMatches.length === 1) return branchMatches[0];
  if (branchMatches.length > 1) {
    throw new Error("Multiple Power Lora Loaders are connected to the bound generation branch");
  }
  if (candidates.length === 1) return candidates[0];
  throw new Error("Multiple Power Lora Loaders found; connect the intended one to the bound branch");
}

function setPowerLoraLoader(node, loras, warnings) {
  if (typeof node?.addNewLoraWidget !== "function") {
    warnings.push(`${node?.title || POWER_LORA_LOADER_TYPE}: incompatible rgthree version`);
    return false;
  }

  let widgets = (node.widgets || []).filter((widget) => /^lora_\d+$/.test(widget.name));
  while (widgets.length < 4) {
    node.addNewLoraWidget();
    widgets = (node.widgets || []).filter((widget) => /^lora_\d+$/.test(widget.name));
  }

  widgets.forEach((widget, index) => {
    const lora = index < 4 ? loras[index] : null;
    const enabled = Boolean(lora?.name && lora.name !== "None");
    const strength = Number(lora?.strength ?? 1.0);
    const previous = typeof widget.value === "object" && widget.value ? widget.value : {};
    widget.value = {
      ...previous,
      on: enabled,
      lora: enabled ? lora.name : previous.lora ?? null,
      strength,
      strengthTwo: previous.strengthTwo == null ? null : strength,
    };
    widget.callback?.(widget.value);
  });
  node.setDirtyCanvas?.(true, true);
  return true;
}

function validateBindingTargets(records) {
  const errors = [];
  const groupedIds = (names) => new Set(names.map((name) => records[name]?.node_id));
  const checkGroup = (names, expectedTypes, label) => {
    const ids = groupedIds(names);
    if (ids.size !== 1 || ids.has(undefined)) {
      errors.push(`${label} outputs must connect to one node`);
      return;
    }
    const target = app.graph.getNodeById([...ids][0]);
    if (!target || !expectedTypes.includes(target.type)) {
      errors.push(`${label} has an unsupported target node`);
    }
  };

  checkGroup(
    ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    ["KSampler", "KSamplerAdvanced"],
    "KSampler",
  );
  checkGroup(["ckpt_name"], ["CheckpointLoaderSimple", "CheckpointLoader"], "Checkpoint");
  checkGroup(
    [
      "lora_01", "strength_01", "lora_02", "strength_02",
      "lora_03", "strength_03", "lora_04", "strength_04",
    ],
    ["Lora Loader Stack (rgthree)", POWER_LORA_LOADER_TYPE],
    "LoRA Stack",
  );
  checkGroup(["positive_prompt"], ["CLIPTextEncode"], "Positive prompt");
  checkGroup(["negative_prompt"], ["CLIPTextEncode"], "Negative prompt");

  if (records.positive_prompt?.node_id === records.negative_prompt?.node_id) {
    errors.push("Positive and negative prompts must connect to different CLIPTextEncode nodes");
  }
  return errors;
}

function captureCurrentConnections(node) {
  const records = {};
  const linkIds = [];
  for (const outputName of REQUIRED_OUTPUTS.filter((name) => !isLoraOutput(name))) {
    const slot = node.outputs?.findIndex((output) => output.name === outputName) ?? -1;
    const links = slot >= 0 ? node.outputs[slot]?.links || [] : [];
    if (links.length !== 1) {
      throw new Error(`Output '${outputName}' must have exactly one connection`);
    }
    const link = app.graph.links[links[0]];
    const target = link && app.graph.getNodeById(link.target_id);
    const input = target?.inputs?.[link.target_slot];
    if (!target || !input) throw new Error(`Cannot resolve target for '${outputName}'`);
    records[outputName] = {
      node_id: target.id,
      input_name: input.widget?.name || input.name,
      node_type: target.type,
    };
    linkIds.push(links[0]);
  }

  const loraOutputs = REQUIRED_OUTPUTS.filter(isLoraOutput);
  const loraConnections = loraOutputs.map((outputName) => {
    const slot = node.outputs?.findIndex((output) => output.name === outputName) ?? -1;
    return { outputName, links: slot >= 0 ? node.outputs[slot]?.links || [] : [] };
  });
  if (loraConnections.every(({ links }) => links.length === 0)) {
    const target = findPowerLoraTarget(records);
    if (!target) {
      throw new Error("Connect all LoRA outputs to a LoRA Stack or add a Power Lora Loader to the branch");
    }
    for (const { outputName } of loraConnections) {
      records[outputName] = {
        node_id: target.id,
        input_name: outputName,
        node_type: target.type,
        adapter: "power_lora",
      };
    }
  } else {
    for (const { outputName, links } of loraConnections) {
      if (links.length !== 1) {
        throw new Error(`LoRA output '${outputName}' must have exactly one connection`);
      }
      const link = app.graph.links[links[0]];
      const target = link && app.graph.getNodeById(link.target_id);
      const input = target?.inputs?.[link.target_slot];
      if (!target || !input) throw new Error(`Cannot resolve target for '${outputName}'`);
      records[outputName] = {
        node_id: target.id,
        input_name: input.widget?.name || input.name,
        node_type: target.type,
      };
      linkIds.push(links[0]);
    }
  }

  const errors = validateBindingTargets(records);
  if (errors.length) throw new Error(errors.join("; "));
  return { records, linkIds };
}

function bindingSummary(binding) {
  if (!binding?.records) return "not bound";
  const unique = [];
  const seen = new Set();
  for (const record of Object.values(binding.records)) {
    if (!seen.has(record.node_id)) {
      seen.add(record.node_id);
      unique.push(`${record.node_type} #${record.node_id}`);
    }
  }
  return unique.join(", ");
}

function bindAndDetach(node) {
  let captured;
  try {
    captured = captureCurrentConnections(node);
  } catch (error) {
    node.__pngInfoStatus.value = `Binding error: ${error.message || error}`;
    return;
  }

  app.graph.beforeChange?.();
  try {
    node.properties ||= {};
    node.properties[BINDING_PROPERTY] = {
      version: 1,
      records: captured.records,
    };
    for (const linkId of captured.linkIds) app.graph.removeLink(linkId);
  } finally {
    app.graph.afterChange?.();
    app.graph.change?.();
    app.canvas.setDirty?.(true, true);
  }

  node.__pngInfoStatus.value = `Bound and detached\n${bindingSummary(node.properties[BINDING_PROPERTY])}`;
  window.setTimeout(() => applyToBoundNodes(node), 0);
}

function clearBinding(node) {
  if (node.properties) delete node.properties[BINDING_PROPERTY];
  node.__pngInfoStatus.value = "Binding cleared; connect all outputs to bind again";
  app.graph.change?.();
  app.canvas.setDirty?.(true, true);
}

function statusText(data, applyWarnings = []) {
  const warnings = [...(data?.warnings || []), ...applyWarnings];
  const prefix = data?.source_format ? `${data.source_format} parsed` : "PNG Info";
  return warnings.length ? `${prefix}\n⚠ ${warnings.join("\n⚠ ")}` : `${prefix}: ready`;
}

async function refreshMetadata(node) {
  const imageWidget = findWidget(node, "image");
  const status = node.__pngInfoStatus;
  if (!imageWidget?.value) {
    if (status) status.value = "Choose a PNG with A1111 or ComfyUI metadata";
    return;
  }

  if (status) status.value = "Reading metadata…";
  try {
    const response = await api.fetchApi(
      `/png-info-for-comfyui/parse?image=${encodeURIComponent(imageWidget.value)}`,
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    node.__pngInfoData = payload;
    if (status) status.value = statusText(payload);
  } catch (error) {
    node.__pngInfoData = null;
    if (status) status.value = `Error: ${error.message || error}`;
  }
  node.setDirtyCanvas?.(true, true);
}

function applyToBoundNodes(node) {
  const data = node.__pngInfoData;
  if (!data) {
    node.__pngInfoStatus.value = "Load a supported PNG before applying settings";
    return;
  }

  const binding = node.properties?.[BINDING_PROPERTY];
  if (!binding?.records) {
    node.__pngInfoStatus.value = "Connect all 17 value outputs, then press Bind & detach";
    return;
  }

  const warnings = [];
  const appliedPowerLoraNodes = new Set();
  app.graph.beforeChange?.();
  try {
    for (const [outputName, record] of Object.entries(binding.records)) {
      const target = app.graph.getNodeById(record.node_id);
      if (!target) {
        warnings.push(`${record.node_type} #${record.node_id} was deleted`);
        continue;
      }
      if (record.adapter === "power_lora") {
        if (!appliedPowerLoraNodes.has(target.id)) {
          setPowerLoraLoader(target, data.loras || [], warnings);
          appliedPowerLoraNodes.add(target.id);
        }
        continue;
      }
      const value = bindingValue(data, outputName);
      if (outputName === "ckpt_name" && !value) {
        warnings.push("Checkpoint is not installed; current selection was kept");
        continue;
      }
      setWidget(target, record.input_name, value, warnings);
    }
  } finally {
    app.graph.afterChange?.();
    app.graph.change?.();
    app.canvas.setDirty?.(true, true);
  }
  node.__pngInfoStatus.value = statusText(data, warnings);
}

app.registerExtension({
  name: EXTENSION_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalCreated?.apply(this, args);
      const imageWidget = findWidget(this, "image");
      if (imageWidget) {
        const originalCallback = imageWidget.callback;
        imageWidget.callback = (...callbackArgs) => {
          const callbackResult = originalCallback?.apply(imageWidget, callbackArgs);
          window.setTimeout(() => refreshMetadata(this), 0);
          return callbackResult;
        };
      }

      this.__pngInfoStatus = this.addWidget(
        "text",
        "status",
        "Choose a PNG with A1111 or ComfyUI metadata",
        () => {},
        { multiline: true },
      );
      this.__pngInfoStatus.serialize = false;
      this.addWidget("button", "Bind all connected nodes & detach", null, () => {
        bindAndDetach(this);
      });
      this.addWidget("button", "Apply to bound nodes", null, () => {
        applyToBoundNodes(this);
      });
      this.addWidget("button", "Clear binding", null, () => {
        clearBinding(this);
      });
      this.setSize([Math.max(this.size[0], 380), Math.max(this.size[1], 230)]);

      window.setTimeout(() => {
        refreshMetadata(this);
        const binding = this.properties?.[BINDING_PROPERTY];
        if (binding?.records) {
          this.__pngInfoStatus.value = `Bound\n${bindingSummary(binding)}`;
        }
      }, 0);
      return result;
    };
  },
});
