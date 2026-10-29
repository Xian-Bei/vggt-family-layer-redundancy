"use strict";

// Cached scientific results come from the same Python callbacks as analysis.view.
// This module controls presentation; it never selects calibration winners.
const $ = id => document.getElementById(id);
const pages = [
  ["single-overview", "Layer Redundancy", ""],
  ["remaining", "Approximate Additivity", ""],
  ["similarity", "Representation Proxies", ""],
  ["pruning-overview", "Pruning & Generalization", ""],
  ["recovery-overview", "Post-Pruning Recovery", ""],
  ["final-table", "Accuracy & Efficiency", ""],
];
const retiredPages = {"final-overview": "final-table","single-detail": "single-overview", "pruning-detail": "pruning-overview", "recovery-detail": "recovery-overview", "recovery-table": "recovery-overview", "final-detail": "final-table", radar: "final-table"};
const cache = new Map(), states = new Map();
let index, figures, active, currentState, renderSequence = 0, activationSequence = 0;
const unique = xs => [...new Set(xs)].sort((a, b) => String(a).localeCompare(String(b), undefined, {numeric: true}));
const datasetLabel = (x, nameOnly = false) => {
  const base = String(x).match(/^(scannet|nuscenes|7scenes|waymo)(?:_|$)/i)?.[1].toLowerCase();
  if (nameOnly && base) return {scannet: "ScanNet", nuscenes: "nuScenes", "7scenes": "7Scenes", waymo: "Waymo"}[base];
  const names = {
    scannet_dev50: "ScanNet Dev", scannet_test50: "ScanNet Test",
    nuscenes_train: "nuScenes Dev", nuscenes_val: "nuScenes Test",
    "7scenes_train": "7Scenes Dev", "7scenes_test": "7Scenes Test",
    waymo_training: "Waymo Dev", waymo_validation: "Waymo Test",
    same_dataset_testing: "Same-dataset testing average", cross_dataset_testing: "Cross-dataset testing average",
    "same-dataset testing average": "Same-dataset testing average",
    "cross-dataset testing average": "Cross-dataset testing average",
    "two-dataset testing average": "Two-dataset testing average",
    "four-dataset testing average": "Four-dataset testing average",
    "four-dataset average": "Four-dataset average", "training average": "Calibration-set average",
    training: "Calibration-set average", min: "Minimum", mean: "Mean",
  };
  return names[x] || String(x).replaceAll("_", " ").replace(/scannet/gi, "ScanNet").replace(/nuscenes/gi, "nuScenes").replace(/7scenes/gi, "7Scenes").replace(/waymo/gi, "Waymo");
};
const recoveryNames = {recovery_linearpatch: "LinearPatch", recovery_replaceme_cos: "ReplaceMe Cos", recovery_replaceme_ls: "ReplaceMe LS", recovery_streamline_ffn: "Streamline FFN", recovery_ghost: "Ghost", recovery_ghost_interval_sequential_autogroup: "Ours"};
const datasetGroups = [
  ["training", "training average", ["scannet_dev50", "nuscenes_train", "7scenes_train", "waymo_training"]],
  ["same_dataset_testing", "same-dataset testing average", ["scannet_test50", "nuscenes_val"]],
  ["cross_dataset_testing", "cross-dataset testing average", ["7scenes_test", "waymo_validation"]],
];
const datasetOrder = datasetGroups.flatMap(([pruning, recovery, members]) => [pruning, recovery, ...members]);
function groupedDatasetChoices(available, recovery = false) {
  const choices = datasetGroups.flatMap(([pruning, recovered, members]) => [recovery ? recovered : pruning, ...members].map((value, i) => [value, datasetLabel(value), {
    className: i === 0 ? "dataset-average" : "dataset-member", disabled: !available.includes(value),
    title: !available.includes(value) ? (recovery ? "No recovery evaluation for this split in analysis_outputs_v3; separate main_results experiments are available in the final tabs." : "No published results for this dataset split") : i === 0 && value.startsWith("train") ? "Average of ScanNet Dev and nuScenes Dev" : "",
  }]));
  const known = new Set(choices.map(c => c[0]));
  return [...choices, ...available.filter(v => !known.has(v)).map(v => [v, datasetLabel(v)])];
}
const label = x => index.backbone_labels[x] || datasetLabel(x);
const element = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };

async function resource(path, text = false) {
  const key = `${text}:${path}`;
  if (!cache.has(key)) cache.set(key, fetch(`data/${path}`, {cache: "no-cache"}).then(r => {
    if (!r.ok) throw new Error(`Could not load ${path} (${r.status}).`);
    return text ? r.text() : r.json();
  }).catch(error => { cache.delete(key); throw error; }));
  return cache.get(key);
}
function defaults(key, value) { if (!(key in currentState)) currentState[key] = value; return currentState[key]; }
function remember() { states.set(active, currentState); }
function button(title, fn, parent = $("controls")) { const b = element("button", title); b.type = "button"; b.onclick = fn; parent.append(b); return b; }
function toast(text) { $("toast")?.remove(); const t = element("div", text); t.id = "toast"; document.body.append(t); setTimeout(() => t.remove(), 2300); }
function checks(title, key, options, initial, radio = false) {
  const selected = defaults(key, initial), field = element("fieldset"); field.append(element("legend", title));
  options.forEach(option => {
    const [value, titleText] = Array.isArray(option) ? option : [option, label(option)];
    const config = Array.isArray(option) ? option[2] || {} : {};
    const row = element("label", null, "check"), input = document.createElement("input");
    if (config.className) row.classList.add(config.className);
    row.title = config.title || ""; input.disabled = !!config.disabled;
    input.type = radio ? "radio" : "checkbox"; input.name = key; input.value = value;
    input.checked = !input.disabled && (radio ? selected === value : selected.includes(value));
    input.onchange = () => { currentState[key] = radio ? value : [...field.querySelectorAll("input:checked")].map(i => i.value); remember(); render(); };
    row.append(input, document.createTextNode(titleText)); field.append(row);
  }); $("controls").append(field);
}
function select(title, key, options, initial, rebuild = false, disabled = false) {
  const field = element("label", title, "select-label"), input = element("select");
  input.setAttribute("aria-label", title);
  const value = defaults(key, initial);
  options.forEach(option => { const [v, text] = Array.isArray(option) ? option : [option, label(option)]; const o = element("option", text); o.value = v; input.append(o); });
  input.value = options.some(o => (Array.isArray(o) ? o[0] : o) === value) ? value : (Array.isArray(options[0]) ? options[0][0] : options[0]);
  currentState[key] = input.value; input.disabled = disabled;
  input.onchange = () => { currentState[key] = input.value; if (key === "dataset") currentState.scene = "mean"; remember(); rebuild ? activate(active) : render(); };
  field.append(input); $("controls").append(field);
}
function toggle(title, key, initial = false) {
  const row = element("label", null, "check"), input = element("input"); input.type = "checkbox"; input.checked = defaults(key, initial);
  input.onchange = () => { currentState[key] = input.checked; remember(); render(); }; row.append(input, document.createTextNode(title)); $("controls").append(row);
}
async function configure(ticket) {
  $("controls").replaceChildren();
  if (active === "single-overview" || active === "remaining" || active === "similarity") {
    const h = await resource("heatmaps.json");
    if (active === "single-overview") {
      select("Dataset", "dataset", ["mean", ...h.single_datasets].map(v => [v, datasetLabel(v, true)]), "mean");
    } else if (active === "remaining") {
      select("Dataset", "dataset", h.remaining.map(r => [r.dataset, datasetLabel(r.dataset, true)]), "mean");
    } else {
      checks("Backbones", "backbones", h.backbones, h.backbones);
      select("Dataset", "dataset", unique(h.similarity.map(r => r.dataset)).map(v => [v, datasetLabel(v, true)]), "mean", true);
      const entries = h.similarity.filter(r => r.dataset === currentState.dataset);
      currentState.scene = "mean";
      const samples = await Promise.all(entries.map(r => resource(r.file)));
      if (ticket !== activationSequence) return;
      const regions = unique(samples.flatMap(rows => rows.flatMap(row => Object.keys(row.regions))));
      checks("Token 1 − similarity regions", "regions", regions, ["all"]);
    }
  } else if (active.startsWith("pruning")) {
    const p = await resource("pruning.json");
    checks("Backbones", "backbones", ["mean", ...index.backbones], index.backbones);
    checks("Datasets", "datasets", groupedDatasetChoices(p.scopes), ["training", "same_dataset_testing", "cross_dataset_testing"]);
    currentState.merge = true;
    checks("Methods", "method", [["similarity", "Similarity"], ["baseline", "Measured"]], "baseline", true);
  } else if (active.endsWith("table")) {
    // Table 1 is a fixed publication snapshot, with no exploratory filters.
    return;
  } else {
    const r = await resource("recovery.json"), final = active.startsWith("final");
    const entries = r.entries.filter(e => e.final === final && !e.detail);
    checks("Backbones", "backbones", ["mean", ...index.backbones], index.backbones);
    const methods = (r.methods_by_view?.[final ? "final" : "recovery"] || r.methods).filter(m => m !== "recovery_ghost_interval_sequential");
    currentState.methods = defaults("methods", methods).filter(m => methods.includes(m));
    const datasets = unique(entries.map(e => e.dataset)).filter(d => !["7scenes_train", "waymo_training"].includes(d));
    currentState.datasets = defaults("datasets", datasets.filter(d => d.includes("average"))).filter(d => datasets.includes(d));
    if (final) checks("Recovery methods", "methods", methods, ["recovery_ghost_interval_sequential_autogroup"]);
    checks("Datasets", "datasets", final ? datasets : groupedDatasetChoices(datasets, true).filter(([v]) => !["7scenes_train", "waymo_training"].includes(v)), final ? ["four-dataset average"] : datasets.filter(d => d.includes("average")));
    if (!final) checks("Recovery methods", "methods", methods.map(v => [v, recoveryNames[v] || label(v.replace(/^recovery_/, ""))]), methods);
    checks("Prune strategies", "strategies", unique(entries.map(e => e.strategy)).map(s => [s, s === "CKA_min" ? "CKA" : "Measured"]), ["Retention_min"]);
  }
  button("Reset controls", () => { states.delete(active); activate(active); });
  button("Refresh published data", async () => {
    const page = active; cache.clear();
    try { [index, figures] = await Promise.all([resource("index.json"), resource("figures.json")]); await activate(page); }
    catch (error) { $("status").className = "error"; $("status").textContent = error.message; }
  });
}

function plotRow(parent, title, columns) {
  const row = element("div", null, "plot-row"); row.style.setProperty("--columns", String(Math.max(1, columns)));
  if (title) row.append(element("h3", title)); parent.append(row); return row;
}
function download(content, name, type) {
  const url = URL.createObjectURL(new Blob([content], {type})); const a = element("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function heatmap(canvas, data) {
  const titleLines = (data.title || "").split("\n").filter(Boolean);
  // Reserve the same axes, title and colorbar slots for every tile on a page.
  const size = 48, left = 58, top = ["remaining", "similarity"].includes(active) ? 94 : 34, cell = 7.5, side = size * cell, bottom = top + side;
  canvas.width = 500; canvas.height = bottom + 62; canvas.className = "heatmap";
  canvas.dataset.plotBounds = JSON.stringify({x: left, y: top, width: side, height: side});
  const c = canvas.getContext("2d"); c.fillStyle = "white"; c.fillRect(0, 0, canvas.width, canvas.height);
  c.font = "bold 19px sans-serif"; c.fillStyle = "#333"; c.textAlign = "center";
  titleLines.forEach((line, i) => c.fillText(line, left + side / 2, 22 + i * 24, side + 65));
  const values = data.values || Array.from({length: data.n}, () => Array(data.n).fill(null));
  if (data.packed) { let offset = 0; for (let i = 0; i < data.n; i++) for (let j = i; j < data.n; j++) { values[i][j] = values[j][i] = data.packed[offset++]; } }
  const finite = values.flat().filter(v => v != null && Number.isFinite(v));
  const min = data.vmin ?? (finite.length ? Math.min(...finite) : 0), max = data.vmax ?? (finite.length ? Math.max(...finite) : 1);
  const palette = index.palettes[data.cmap] || index.palettes.viridis;
  for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) {
    const v = values[i]?.[j], x = left + i * cell, y = top + (size - 1 - j) * cell;
    if (v == null || !Number.isFinite(v)) {
      c.fillStyle = "#f3f4f6"; c.fillRect(x, y, cell, cell);
      c.strokeStyle = "#d5dce4"; c.beginPath(); c.moveTo(x, y + cell); c.lineTo(x + cell, y); c.stroke();
    } else {
      const t = max === min ? 0 : Math.max(0, Math.min(1, (v - min) / (max - min))), rgb = palette[Math.min(255, Math.floor(t * 256))];
      c.fillStyle = `rgb(${rgb.join(",")})`; c.fillRect(x, y, cell + .2, cell + .2);
    }
  }
  c.strokeStyle = "#adb4bf"; c.strokeRect(left, top, side, side); c.fillStyle = "#444"; c.font = "16px sans-serif";
  const ticks = [1, 8, 16, 24, 32, 40, 48];
  ticks.forEach(v => { if (data.show_xindex !== false) { c.textAlign = "center"; c.fillText(v, left + (v - .5) * cell, bottom + 19); } if (data.show_yindex !== false) { c.textAlign = "right"; c.fillText(v, left - 8, bottom - (v - .5) * cell + 4); } });
  c.textAlign = "center"; if (data.show_xlabel !== false) c.fillText("Layer index i", left + side / 2, bottom + 43);
  if (data.show_ylabel !== false) { c.save(); c.translate(14, top + side / 2); c.rotate(-Math.PI / 2); c.fillText("Layer index j", 0, 0); c.restore(); }
  if (data.show_colorbar !== false) {
    for (let k = 0; k < 256; k++) { const rgb = palette[255 - k]; c.fillStyle = `rgb(${rgb.join(",")})`; c.fillRect(left + side + 14, top + k * side / 256, 11, side / 256 + 1); }
    c.fillStyle = "#555"; c.font = "14px sans-serif"; c.textAlign = "left";
    [0, .25, .5, .75, 1].forEach(t => { const v = max - t * (max - min), multiplier = data.colorbar_tick_scale || 1; c.fillText(`${Number((v * multiplier).toFixed(2))}${data.tick_suffix || ""}`, left + side + 29, top + t * side + 4); });
  }
  const tooltip = element("div", "", "tooltip"); canvas.after(tooltip);
  canvas.onpointermove = e => {
    const box = canvas.getBoundingClientRect(), x = (e.clientX - box.left) * canvas.width / box.width, y = (e.clientY - box.top) * canvas.height / box.height;
    const i = Math.floor((x - left) / cell), j = size - 1 - Math.floor((y - top) / cell);
    const v = values[i]?.[j]; tooltip.textContent = i >= 0 && i < size && j >= 0 && j < size ? `i = ${i + 1}, j = ${j + 1} · ${v == null ? "No measurement" : Number(v).toPrecision(6)}` : "";
  };
}
function selectedMethod(method, labelText, selected) {
  if (!selected) return true;
  if (!method || /no_recovery|^upper$|^lower$/.test(method) || /Full Model|No Recovery/.test(labelText)) return true;
  const norm = s => s.split("__")[0].replace(/^recovery_/, "").toLowerCase();
  return selected.some(s => norm(method) === norm(s));
}
function normalizePlot(svg, page) {
  const rects = new Map();
  svg.querySelectorAll("clipPath").forEach(clip => { const rect = clip.querySelector("rect"); if (rect) rects.set(clip.id, rect); });
  let plotted = 0;
  svg.querySelectorAll('[id^="axes_"]').forEach(axis => {
    const clipped = axis.querySelector("[clip-path]"), match = clipped?.getAttribute("clip-path").match(/url\(#([^)]*)\)/), rect = match && rects.get(match[1]);
    if (!rect) return;
    const x = Number(rect.getAttribute("x")), y = Number(rect.getAttribute("y")), width = Number(rect.getAttribute("width"));
    const scale = 420 / width;
    const tx = 94 - x * scale, ty = 25 - y * scale;
    axis.setAttribute("transform", `translate(${tx} ${ty}) scale(${scale})`); plotted++;
  });
  if (plotted) svg.setAttribute("viewBox", "0 0 540 336");
}
function appendLegend(parent, prepend = false) {
  const legend = element("div", null, "legend"), seen = new Set();
  for (const line of parent._plotLines || []) {
    if (line.label.startsWith("_")) continue;
    let title = line.label.replace(/^upper$/, "Full Model").replace(/^Retention /, "Ours · ").replace(/ \(calibration-selected Single\/Double\)/, "");
    if (active.startsWith("final") && line.method && !/Full Model/.test(title)) title += line.style === "--" ? " · Recovery" : " · Final";
    if (active === "recovery-overview" && recoveryNames[line.method]) title = recoveryNames[line.method];
    const key = `${title}:${line.color}:${line.style}`; if (seen.has(key)) continue; seen.add(key);
    const item = element("span"), stroke = element("i"); stroke.style.setProperty("--line-color", line.color);
    if (line.style === "--") stroke.classList.add("dashed"); if (line.style === ":") stroke.classList.add("dotted");
    item.append(stroke, document.createTextNode(title)); legend.append(item);
  }
  if (legend.children.length) prepend ? parent.prepend(legend) : parent.append(legend);
}
async function cell(parent, ref, caption = "", methods = null) {
  const figure = element("figure", null, "cell"), tools = element("div", null, "cell-tools"); figure.append(tools); parent.append(figure);
  if (ref.kind === "heatmap") {
    const data = await resource(`heatmaps/${ref.key}.json`), canvas = element("canvas"); canvas.setAttribute("aria-label", caption || data.title || "Layer degradation heatmap"); figure.append(canvas); heatmap(canvas, ref.displayTitle ? {...data, title: ref.displayTitle} : data);
    button("PNG ↓", () => { const a = element("a"); a.href = canvas.toDataURL("image/png"); a.download = `${ref.key}.png`; a.click(); }, tools);
  } else if (ref.kind === "figure") {
    let meta = figures[ref.key]; if (!meta) throw new Error(`Plot metadata missing: ${ref.key}`);
    if (meta.metadata) meta = await resource(meta.metadata);
    const source = await resource(meta.file, true), doc = new DOMParser().parseFromString(source, "image/svg+xml");
    if (doc.querySelector("parsererror")) throw new Error(`Invalid plot: ${meta.file}`);
    const svg = document.importNode(doc.documentElement, true); svg.removeAttribute("width"); svg.removeAttribute("height");
    // Matplotlib IDs repeat between figures. Prefix every ID and reference before
    // combining multiple plots in one HTML document.
    const prefix = `p${crypto.randomUUID().replaceAll("-", "")}-`;
    const selectedLines = meta.axes.flatMap(a => a.lines).filter(l => selectedMethod(l.method, l.label, methods));
    for (const axis of meta.axes) for (const line of axis.lines) {
      const node = svg.querySelector(`[id="${line.id}"]`); if (node && !selectedMethod(line.method, line.label, methods)) node.style.display = "none";
    }
    if (methods) {
      // Matplotlib emits scatter markers separately from their Line2D paths.
      // Toggle both artists, including hollow markers that only have a stroke.
      const lines = meta.axes.flatMap(axis => axis.lines);
      const colors = new Map();
      for (const line of lines) {
        const color = line.color.toLowerCase();
        colors.set(color, (colors.get(color) || false) || selectedMethod(line.method, line.label, methods));
      }
      svg.querySelectorAll('[id^="PathCollection_"]').forEach(collection => {
        const used = new Set();
        collection.querySelectorAll("[style]").forEach(node => {
          for (const match of node.getAttribute("style").matchAll(/(?:fill|stroke):\s*(#[\da-f]{6})\b/gi)) {
            const color = match[1].toLowerCase(); if (colors.has(color)) used.add(color);
          }
        });
        if (used.size && ![...used].some(color => colors.get(color))) collection.style.display = "none";
      });
    }
    svg.querySelectorAll('[id^="legend_"]').forEach(legend => legend.remove());
    normalizePlot(svg, active);
    parent._plotLines = [...(parent._plotLines || []), ...selectedLines];
    svg.querySelectorAll("*").forEach(node => { [...node.attributes].forEach(attr => { if (attr.name === "id") node.setAttribute(attr.name, prefix + attr.value); else if (attr.value.includes("url(#")) node.setAttribute(attr.name, attr.value.replaceAll("url(#", `url(#${prefix}`)); else if ((attr.name === "href" || attr.name === "xlink:href") && attr.value.startsWith("#")) node.setAttribute(attr.name, "#" + prefix + attr.value.slice(1)); }); });
    figure.append(svg);
    button("SVG ↓", () => download(new XMLSerializer().serializeToString(svg), `${ref.key}.svg`, "image/svg+xml"), tools);
  } else figure.append(element("div", ref.caption || "No result for this selection.", "empty"));
  const text = caption || ref.caption; if (text && !ref.hideCaption) figure.append(element("figcaption", text));
  return figure;
}
function empty(parent, text = "Select at least one item with available results.") { parent.append(element("div", text, "empty")); }

async function renderHeatmaps(parent) {
  const h = await resource("heatmaps.json"), s = currentState;
  if (active === "single-overview") {
    const datasets = s.dataset === "mean" ? h.single_datasets : [s.dataset];
    const entry = h.single_overview.find(r => r.datasets.length === datasets.length && r.datasets.every(d => datasets.includes(d)));
    if (!entry) { empty(parent); return "No datasets selected."; }
    const grid = element("div", null, "redundancy-grid"); parent.append(grid);
    const domains = s.dataset === "mean" ? ["Indoor", "Outdoor"] : [/nuscenes|waymo/.test(s.dataset) ? "Outdoor" : "Indoor"];
    const offsets = domains.flatMap(d => d === "Indoor" ? [0, 1] : [2, 3]);
    grid.style.setProperty("--heatmap-columns", offsets.length);
    grid.append(element("span"), ...domains.map(d => element("h3", s.dataset === "mean" ? `${d} Normalized Degradation` : datasetLabel(s.dataset, true), "redundancy-domain")));
    const metrics = offsets.map((_, i) => i % 2 ? "Geometry Metrics" : "Trajectory Metrics");
    for (let i = 0; i < h.backbones.length; i++) {
      grid.append(element("h3", label(h.backbones[i])));
      await Promise.all(offsets.map((offset, j) => cell(grid, {...entry.rows[i][offset], hideCaption: true}, `${label(h.backbones[i])} · ${domains[Math.floor(j / 2)]} · ${metrics[j]}`)));
    }
    grid.append(element("span"), ...metrics.map(title => element("div", title, "redundancy-metric")));
    return entry.status;
  }
  if (active === "remaining") {
    const entry = h.remaining.find(r => r.dataset === s.dataset);
    const models = plotRow(parent, "", h.backbones.length);
    for (let i = 0; i < h.backbones.length; i++) {
      const group = element("section", null, "backbone-group"), row = element("div", null, "remaining-grid"); group.append(element("h3", label(h.backbones[i]))); group.append(row); models.append(group);
      const refs = entry.rows[i]; if (!refs.length) { empty(row); continue; }
      await cell(row, refs[0], "Original Degradation");
      if (refs[1]) {
        const parts = refs[1].caption.split("\n");
        const marginal = await cell(row, {...refs[1], displayTitle: parts.slice(1).join("\n")}, "Marginal Degradation");
        marginal.title = `After pruning ${parts[0]} layers`;
      }
    }
    return entry.status;
  }
  let count = 0; const messages = [];
  for (const e of h.similarity.filter(r => r.dataset === s.dataset && s.backbones.includes(r.backbone))) {
    const entries = await resource(e.file), scope = entries.find(r => r.scene === s.scene);
    if (!scope) { empty(parent, `${label(e.backbone)} · scene ${s.scene} is unavailable.`); continue; }
    for (const region of s.regions) {
      const cells = scope.regions[region]; if (!cells) { empty(parent, `${label(e.backbone)} · ${region} is unavailable.`); continue; }
      const row = plotRow(parent, `${label(e.backbone)} · ${region}`, 5), order = [0, 2, 1, 3, 4];
      await Promise.all(order.map((offset, i) => cells[offset] && cell(row, cells[offset], ["Normalized degradation", "1 − CKA", "1 − Cosine", "1 − CCA", "1 − Inv_MSE"][i]))); count++;
    } messages.push(scope.status);
  }
  if (!count) empty(parent); return unique(messages).join("\n");
}

async function renderCurves(parent) {
  const s = currentState; let entries;
  if (active.startsWith("pruning")) {
    const p = await resource("pruning.json"); entries = p.entries.filter(e => !e.detail && e.merge === s.merge && e.method === s.method && s.backbones.includes(e.backbone) && s.datasets.includes(e.dataset));
  } else {
    const p = await resource("recovery.json"), final = active.startsWith("final"); entries = p.entries.filter(e => e.final === final && !e.detail && s.backbones.includes(e.backbone) && s.datasets.includes(e.dataset) && s.strategies.includes(e.strategy));
    if (!s.methods.length) { empty(parent, "Select at least one recovery method."); return "No recovery methods selected."; }
  }
  const rank = value => datasetOrder.includes(value) ? datasetOrder.indexOf(value) : datasetOrder.length;
  const scopes = unique(entries.map(e => e.dataset)).sort((a, b) => rank(a) - rank(b)), messages = [];
  const sharedLegend = active === "pruning-overview" || active === "recovery-overview";
  for (const dataset of scopes) {
    const heading = element("h3", datasetLabel(dataset), "row-heading"); parent.append(heading);
    const scopeEntries = entries.filter(e => e.dataset === dataset);
    for (const strategy of unique(scopeEntries.map(e => e.strategy || ""))) {
      if (strategy && s.strategies.length > 1) parent.append(element("h3", strategy === "CKA_min" ? "CKA" : "Measured", "row-heading"));
      const row = plotRow(parent, "", Math.min(3, Math.max(1, s.backbones.length)));
      row.classList.add("curve-overview");
      for (const backbone of s.backbones) {
        const e = scopeEntries.find(e => e.backbone === backbone && (e.strategy || "") === strategy); if (!e) continue;
        if (!e.cells.length) { empty(row, `${label(e.backbone)}: no matching result.`); messages.push(e.status); continue; }
        for (const ref of e.cells) await cell(row, ref, label(e.backbone), active.startsWith("pruning") ? null : s.methods);
      }
      if (sharedLegend) parent._plotLines = [...(parent._plotLines || []), ...(row._plotLines || [])];
      else appendLegend(row);
    }
  }
  if (sharedLegend) appendLegend(parent, true);
  if (!entries.length) empty(parent);
  return `${entries.length} selected scopes · calibration selection and plotting coordinates match the local dashboard.\n${unique(messages).join("\n")}`;
}

async function renderTable(parent) {
  const t = await resource("../assets/paper-table.json");
  const wrap = element("div", null, "table-wrap publication-table-wrap"), table = element("table", null, "publication-table");
  table.setAttribute("aria-label", "Table 1: accuracy and efficiency averaged over four held-out datasets");
  const head = element("thead"), groups = element("tr"), labels = element("tr");
  [["", 2], ["Trajectory Errors", 4], ["Geometric Errors", 3], ["System Efficiency", 3]].forEach(([name, span]) => { const th = element("th", name); th.colSpan = span; th.scope = "colgroup"; groups.append(th); });
  t.headers.forEach(h => { const th = element("th", h); th.scope = "col"; labels.append(th); });
  head.append(groups, labels); table.append(head);
  for (const backbone of [...new Set(t.rows.map(r => r.backbone))]) {
    const rows = t.rows.filter(r => r.backbone === backbone), body = element("tbody");
    const minima = t.headers.map((_, i) => Math.min(...rows.slice(1).map(r => Number(r.cells[i]))));
    rows.forEach((r, index) => {
      const tr = element("tr", null, index === 0 ? "base-model" : r.cells[0] === "+ Ours" ? "ours-model" : "");
      r.cells.forEach((value, i) => {
        const td = element(i === 0 ? "th" : "td"); if (i === 0) td.scope = "row";
        const bold = i === 0 && (index === 0 || r.cells[0] === "+ Ours") || index > 0 && i >= 1 && i <= 8 && Number(value) === minima[i];
        td.append(element(bold ? "strong" : "span", value));
        if (i >= 9 && r.reductions[i - 9]) td.append(element("small", ` ↓${r.reductions[i - 9]}%`, "efficiency-reduction"));
        tr.append(td);
      }); body.append(tr);
    }); table.append(body);
  }
  wrap.append(table); parent.append(wrap);
  return "Paper Table 1 · four held-out datasets · publication snapshot";
}
async function render() {
  const sequence = ++renderSequence, target = document.createDocumentFragment();
  $("plots").classList.add("loading"); $("status").className = ""; $("status").textContent = "Loading selected results…";
  try {
    let status;
    if (["single-overview", "remaining", "similarity"].includes(active)) status = await renderHeatmaps(target);
    else if (active.endsWith("table")) status = await renderTable(target);
    else status = await renderCurves(target);
    if (sequence !== renderSequence) return;
    $("plots").replaceChildren(target); $("status").textContent = status;
  } catch (error) {
    if (sequence !== renderSequence) return;
    $("plots").replaceChildren(); $("status").className = "error"; $("status").textContent = `${error.message} Use “Refresh published data” to retry.`;
  } finally { if (sequence === renderSequence) $("plots").classList.remove("loading"); }
}
async function activate(id) {
  id = retiredPages[id] || id;
  if (!pages.some(p => p[0] === id)) id = "single-overview";
  const ticket = ++activationSequence;
  ++renderSequence; active = id; currentState = states.get(id) || {}; remember();
  delete currentState.simplify;
  $("panel").classList.toggle("publication-view", active === "final-table");
  $("plots").replaceChildren(); $("plots").classList.add("loading"); $("status").className = ""; $("status").textContent = "Loading selected results…";
  $("panel").setAttribute("aria-labelledby", `tab-${active}`);
  document.querySelectorAll("[role=tab]").forEach(b => { const selected = b.dataset.page === active; b.setAttribute("aria-selected", String(selected)); b.tabIndex = selected ? 0 : -1; });
  const url = new URL(location.href); url.searchParams.delete("view"); url.hash = active; history.replaceState(null, "", url);
  try {
    const file = ["single-overview", "remaining", "similarity"].includes(active) ? "heatmaps.json" : active.startsWith("pruning") ? "pruning.json" : active === "final-table" ? "../assets/paper-table.json" : "recovery.json";
    await resource(file); if (ticket !== activationSequence) return;
    await configure(ticket); if (ticket !== activationSequence) return;
    await render();
  } catch (error) { if (ticket === activationSequence) { $("status").className = "error"; $("status").textContent = error.message; } }
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch { const t = element("textarea"); t.value = text; document.body.append(t); t.select(); document.execCommand("copy"); t.remove(); toast("Copied"); }
}
async function main() {
  [index, figures] = await Promise.all([resource("index.json"), resource("figures.json")]);
  if (index.schema_version !== "dashboard_v2") throw new Error("The dashboard assets need to be refreshed with analysis.results.export_dashboard.");
  $("snapshot").textContent = `Result snapshot · ${new Date(index.exported_at).toLocaleDateString()} · VGGT, Pi3 and VGGT-Omega`;
  pages.forEach(([id, title, kind], i) => {
    const b = element("button", `${i + 1}. ${title}`); b.type = "button"; b.id = `tab-${id}`; b.dataset.page = id; b.setAttribute("role", "tab"); b.setAttribute("aria-controls", "panel");
    if (kind) b.append(element("small", kind)); b.onclick = () => activate(id);
    b.onkeydown = e => { if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return; e.preventDefault(); const next = e.key === "Home" ? 0 : e.key === "End" ? pages.length - 1 : (i + (e.key === "ArrowRight" ? 1 : -1) + pages.length) % pages.length; $("tabs").children[next].focus(); activate(pages[next][0]); };
    $("tabs").append(b);
  });
  $("share").onclick = async () => { const url = new URL(location.href); url.searchParams.set("view", JSON.stringify({page: active, state: currentState})); await copy(url.href); };
  window.addEventListener("hashchange", () => activate(location.hash.slice(1)));
  const shared = new URL(location.href).searchParams.get("view");
  if (shared) { try { const v = JSON.parse(shared); v.page = retiredPages[v.page] || v.page; if (pages.some(p => p[0] === v.page) && v.state && typeof v.state === "object") { states.set(v.page, v.state); await activate(v.page); return; } } catch { /* Invalid shared links fall back to the default page. */ } }
  await activate(location.hash.slice(1));
}
$("copy-citation").onclick = () => copy($("bibtex").textContent);
main().catch(error => { $("snapshot").textContent = "Results could not be loaded."; $("status").className = "error"; $("status").textContent = error.message; });

// Match the teaser and introduction to the title's rendered text width,
// including responsive line wrapping and font changes.
const projectTitle = document.querySelector(".project h1");
function alignIntroduction() {
  const range = document.createRange();
  range.selectNodeContents(projectTitle);
  projectTitle.parentElement.style.setProperty("--intro-width", `${range.getBoundingClientRect().width}px`);
}
new ResizeObserver(alignIntroduction).observe(projectTitle);
document.fonts.ready.then(alignIntroduction);
alignIntroduction();
