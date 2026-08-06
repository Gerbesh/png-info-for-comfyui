import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "gerbesh.png-info-for-comfyui";
const NODE_NAME = "PNGInfoForComfyUI";
const QUICK_NODE_NAME = "PNGInfoQuickApply";
const BINDING_PROPERTY = "png_info_binding";
const CONTROLLER_PROPERTY = "png_info_controller_id";
const QUICK_IMAGE_PROPERTY = "png_info_image";
const QUICK_TITLE = "PNG Info — Быстро";
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

function graphNodes(graph) {
  return graph?.nodes ?? graph?._nodes ?? [];
}

function nodeById(graph, id) {
  return graph?.getNodeById?.(id) ?? graphNodes(graph).find((node) => node.id === id) ?? null;
}

function walkGraph(graph, callback) {
  for (const node of graphNodes(graph)) {
    callback(node, graph);
    if (node.subgraph) walkGraph(node.subgraph, callback);
  }
}

function findNodeInAllGraphs(id, expectedType = null) {
  const matches = [];
  walkGraph(app.graph, (node) => {
    if (node.id === id && (!expectedType || node.type === expectedType)) matches.push(node);
  });
  if (matches.length === 1) return matches[0];
  if (!matches.length && expectedType) {
    walkGraph(app.graph, (node) => {
      if (node.id === id) matches.push(node);
    });
  }
  return matches.length === 1 ? matches[0] : null;
}

function updateStatus(node, value) {
  if (node?.__pngInfoStatus) node.__pngInfoStatus.value = value;
  if (node?.__pngInfoHostStatus) node.__pngInfoHostStatus.value = value;
  node?.setDirtyCanvas?.(true, true);
  node?.__pngInfoSubgraphHost?.setDirtyCanvas?.(true, true);
}

function newControllerId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `png-info-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function ensureControllerId(node) {
  node.properties ||= {};
  if (!node.properties[CONTROLLER_PROPERTY]) {
    node.properties[CONTROLLER_PROPERTY] = newControllerId();
    app.graph.change?.();
  }
  return node.properties[CONTROLLER_PROPERTY];
}

function setQuickTitle(node, suffix = "") {
  node.title = suffix ? `${QUICK_TITLE} — ${suffix}` : QUICK_TITLE;
  node.setDirtyCanvas?.(true, true);
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
  const graph = node?.graph ?? app.graph;
  for (const input of node?.inputs || []) {
    const link = input.link != null ? graph?.links?.[input.link] : null;
    if (link) ids.add(link.origin_id);
  }
  for (const output of node?.outputs || []) {
    for (const linkId of output.links || []) {
      const link = graph?.links?.[linkId];
      if (link) ids.add(link.target_id);
    }
  }
  return ids;
}

function findPowerLoraTarget(records) {
  const candidates = [];
  walkGraph(app.graph, (candidate) => {
    if (candidate.type === POWER_LORA_LOADER_TYPE) candidates.push(candidate);
  });
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

function migrateMissingLoraBindingToPower(binding, warnings) {
  const loraOutputs = REQUIRED_OUTPUTS.filter(isLoraOutput);
  const loraRecords = loraOutputs.map((name) => binding.records[name]).filter(Boolean);
  if (loraRecords.some((record) => record.adapter === "power_lora")) return;

  const hasLiveLoraStack = loraRecords.some((record) => {
    const target = findNodeInAllGraphs(record.node_id, record.node_type);
    return target?.type === "Lora Loader Stack (rgthree)";
  });
  if (hasLiveLoraStack) return;

  let target;
  try {
    target = findPowerLoraTarget(binding.records);
  } catch (error) {
    warnings.push(error.message || String(error));
    return;
  }
  if (!target) return;

  for (const outputName of loraOutputs) {
    binding.records[outputName] = {
      node_id: target.id,
      input_name: outputName,
      node_type: target.type,
      adapter: "power_lora",
    };
  }
  binding.version = 2;
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
    const record = records[names[0]];
    const target = findNodeInAllGraphs([...ids][0], record?.node_type);
    const hasPromotedWidgets = Boolean(target) && names.every((name) => {
      const inputName = records[name]?.input_name;
      return inputName && findWidget(target, inputName);
    });
    if (!target || (!expectedTypes.includes(target.type) && !hasPromotedWidgets)) {
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
  const graph = node?.graph ?? app.graph;
  for (const outputName of REQUIRED_OUTPUTS.filter((name) => !isLoraOutput(name))) {
    const slot = node.outputs?.findIndex((output) => output.name === outputName) ?? -1;
    const links = slot >= 0 ? node.outputs[slot]?.links || [] : [];
    if (links.length !== 1) {
      throw new Error(`Output '${outputName}' must have exactly one connection`);
    }
    const link = graph?.links?.[links[0]];
    const target = link && nodeById(graph, link.target_id);
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
      const link = graph?.links?.[links[0]];
      const target = link && nodeById(graph, link.target_id);
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
    updateStatus(node, `Binding error: ${error.message || error}`);
    return;
  }

  app.graph.beforeChange?.();
  try {
    node.properties ||= {};
    node.properties[BINDING_PROPERTY] = {
      version: 2,
      records: captured.records,
    };
    for (const linkId of captured.linkIds) app.graph.removeLink(linkId);
  } finally {
    app.graph.afterChange?.();
    app.graph.change?.();
    app.canvas.setDirty?.(true, true);
  }

  updateStatus(node, `Bound and detached\n${bindingSummary(node.properties[BINDING_PROPERTY])}`);
  window.setTimeout(() => applyToBoundNodes(node), 0);
}

function clearBinding(node) {
  if (node.properties) delete node.properties[BINDING_PROPERTY];
  updateStatus(node, "Binding cleared; connect all outputs to bind again");
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
  if (!imageWidget?.value) {
    updateStatus(node, "Choose a PNG with A1111 or ComfyUI metadata");
    return;
  }

  updateStatus(node, "Reading metadata…");
  try {
    const response = await api.fetchApi(
      `/png-info-for-comfyui/parse?image=${encodeURIComponent(imageWidget.value)}`,
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    node.__pngInfoData = payload;
    updateStatus(node, statusText(payload));
    if (node.__pngInfoAutoApply && node.properties?.[BINDING_PROPERTY]?.records) {
      applyToBoundNodes(node);
    }
  } catch (error) {
    node.__pngInfoData = null;
    updateStatus(node, `Error: ${error.message || error}`);
  }
}

function applyToBoundNodes(node) {
  const data = node.__pngInfoData;
  if (!data) {
    updateStatus(node, "Load a supported PNG before applying settings");
    return false;
  }

  const binding = node.properties?.[BINDING_PROPERTY];
  if (!binding?.records) {
    updateStatus(node, "Connect all 17 value outputs, then press Bind & detach");
    return false;
  }

  const warnings = [];
  const appliedPowerLoraNodes = new Set();
  app.graph.beforeChange?.();
  try {
    migrateMissingLoraBindingToPower(binding, warnings);
    for (const [outputName, record] of Object.entries(binding.records)) {
      const target = findNodeInAllGraphs(record.node_id, record.node_type);
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
  updateStatus(node, statusText(data, warnings));
  return true;
}

function configuredControllers() {
  const controllers = [];
  walkGraph(app.graph, (node) => {
    if (node.type === NODE_NAME && node.properties?.[BINDING_PROPERTY]?.records) {
      ensureControllerId(node);
      controllers.push(node);
    }
  });
  return controllers;
}

function resolveQuickController(node) {
  const controllers = configuredControllers();
  const savedId = node.properties?.[CONTROLLER_PROPERTY];
  const saved = savedId
    ? controllers.find((candidate) => candidate.properties?.[CONTROLLER_PROPERTY] === savedId)
    : null;
  if (saved) return saved;
  if (controllers.length === 1) {
    node.properties ||= {};
    node.properties[CONTROLLER_PROPERTY] = ensureControllerId(controllers[0]);
    app.graph.change?.();
    return controllers[0];
  }
  if (!controllers.length) {
    throw new Error("сначала настройте основную PNG-ноду");
  }
  throw new Error("найдено несколько настроенных PNG-нод");
}

async function loadQuickMetadata(node) {
  const image = node.properties?.[QUICK_IMAGE_PROPERTY];
  if (!image) throw new Error("сначала загрузите картинку");
  setQuickTitle(node, "читаем PNG…");
  const response = await api.fetchApi(
    `/png-info-for-comfyui/parse?image=${encodeURIComponent(image)}`,
  );
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  node.__pngInfoData = payload;
  setQuickTitle(node, "готово");
  return payload;
}

async function chooseQuickImage(node) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png,.png";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    setQuickTitle(node, "загрузка…");
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("type", "input");
      form.append("overwrite", "true");
      const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      const relativeName = payload.subfolder
        ? `${payload.subfolder}/${payload.name}`
        : payload.name;
      const image = payload.type && payload.type !== "input"
        ? `${relativeName} [${payload.type}]`
        : relativeName;
      node.properties ||= {};
      node.properties[QUICK_IMAGE_PROPERTY] = image;
      node.__pngInfoData = null;
      app.graph.change?.();
      await loadQuickMetadata(node);
    } catch (error) {
      node.__pngInfoData = null;
      setQuickTitle(node, `ошибка: ${error.message || error}`);
    }
  };
  input.click();
}

async function applyQuickImage(node) {
  try {
    const data = node.__pngInfoData || await loadQuickMetadata(node);
    const controller = resolveQuickController(node);
    controller.__pngInfoData = data;
    const applied = applyToBoundNodes(controller);
    setQuickTitle(node, applied ? "применено" : "не применено");
  } catch (error) {
    setQuickTitle(node, `ошибка: ${error.message || error}`);
  }
}

function bridgePngInfoSubgraph(hostNode) {
  if (!hostNode?.subgraph) return;
  const pngNode = graphNodes(hostNode.subgraph).find((node) => node.type === NODE_NAME);
  if (!pngNode) return;

  pngNode.__pngInfoSubgraphHost = hostNode;
  pngNode.__pngInfoAutoApply = true;

  const hostUpload = findWidget(hostNode, "upload");
  const innerUpload = findWidget(pngNode, "upload");
  if (hostUpload && innerUpload?.callback && !hostUpload.__pngInfoBridge) {
    hostUpload.__pngInfoBridge = true;
    hostUpload.callback = (...args) => innerUpload.callback?.apply(innerUpload, args);
  }

  let hostStatus = findWidget(hostNode, "PNG Info status");
  if (!hostStatus) {
    hostStatus = hostNode.addWidget(
      "text",
      "PNG Info status",
      "Choose or upload a PNG",
      () => {},
      { multiline: true },
    );
    hostStatus.serialize = false;
  }
  pngNode.__pngInfoHostStatus = hostStatus;

  if (!findWidget(hostNode, "Apply PNG settings")) {
    const applyWidget = hostNode.addWidget("button", "Apply PNG settings", null, () => {
      applyToBoundNodes(pngNode);
    });
    applyWidget.serialize = false;
  }
}

app.registerExtension({
  name: EXTENSION_NAME,

  async afterConfigureGraph() {
    walkGraph(app.graph, (node) => bridgePngInfoSubgraph(node));
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === QUICK_NODE_NAME) {
      const originalCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function (...args) {
        const result = originalCreated?.apply(this, args);
        setQuickTitle(this);
        const upload = this.addWidget("button", "Загрузить картинку", null, () => {
          chooseQuickImage(this);
        });
        upload.serialize = false;
        const apply = this.addWidget("button", "Применить PNG", null, () => {
          applyQuickImage(this);
        });
        apply.serialize = false;
        this.setSize([260, 92]);
        window.setTimeout(() => {
          if (this.properties?.[QUICK_IMAGE_PROPERTY]) {
            loadQuickMetadata(this).catch((error) => {
              setQuickTitle(this, `ошибка: ${error.message || error}`);
            });
          }
        }, 0);
        return result;
      };
      return;
    }

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
        ensureControllerId(this);
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
