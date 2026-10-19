"use strict";
const $ = id => document.getElementById(id);
const names = {vggt: "VGGT", pi3: "π³", vggt_omega: "VGGT-Ω"};
const state = {index: null, tables: new Map(), table: null, rows: [], tableRequest: 0, cloudRequest: 0};
const options = (id, values, preferred) => {
  const select = $(id), previous = preferred ?? select.value;
  select.replaceChildren(...values.map(value => {
    const option = document.createElement("option");
    const pair = Array.isArray(value) ? value : [value, names[value] || value];
    option.value = pair[0]; option.textContent = pair[1]; return option;
  }));
  if ([...select.options].some(o => o.value === previous)) select.value = previous;
};
async function getJSON(file) {
  const response = await fetch(`data/${file}`);
  if (!response.ok) throw new Error(`Could not load ${file}: HTTP ${response.status}`);
  return response.json();
}
function error(message) { $("load-error").hidden = false; $("load-error").textContent = message; }
function modeLabel(row) {
  const groups = {upper: "Original", lower: "Pruning only", recovery: "Recovered"};
  const method = row.recovery_method.replace(/^ghost_interval_sequential_autogroup$/, "LaRe").replaceAll("_", " ");
  const source = row.pruning_sources.join(" / ").replaceAll("retention", "Degradation").replaceAll("_", " ");
  return [groups[row.mode_group] || row.mode_group, method, source, row.compression !== "none" ? row.compression : ""].filter(Boolean).join(" · ");
}
async function loadTable() {
  const request = ++state.tableRequest;
  const view = state.index.views[$("view").value];
  const file = view.files[$("backbone").value];
  $("row-status").textContent = "Loading results…";
  let table = state.tables.get(file);
  if (!table) { table = await getJSON(file); state.tables.set(file, table); }
  if (request !== state.tableRequest) return;
  state.table = table;
  const unique = key => [...new Set(table.rows.map(r => r[key]))].sort();
  options("dataset", unique("dataset"), $("view").value === "final" ? "four-dataset testing average" : "same-dataset testing average");
  const drop = state.index.models.models[$("backbone").value]?.drop_count;
  options("drop", [["all", "All budgets"], ...unique("drop_count").filter(Boolean).sort((a,b)=>Number(a)-Number(b)).map(d => [d,d])], String(drop));
  options("method", [["all", "All methods"], ...unique("recovery_method").filter(Boolean).map(m => [m, m === "ghost_interval_sequential_autogroup" ? "LaRe" : m.replaceAll("_", " ")])], "ghost_interval_sequential_autogroup");
  options("compression", unique("compression"), "none");
  renderTable();
}
function renderTable() {
  if (!state.table) return;
  state.rows = state.table.rows.filter(row => row.dataset === $("dataset").value
    && (row.mode_group === "upper" || $("drop").value === "all" || row.drop_count === $("drop").value)
    && (!row.recovery_method || $("method").value === "all" || row.recovery_method === $("method").value)
    && row.compression === $("compression").value);
  const columns = state.table.columns;
  const header = document.createElement("tr");
  for (const label of ["Model / configuration", ...columns]) { const th = document.createElement("th"); th.scope = "col"; th.textContent = label; header.append(th); }
  $("table-head").replaceChildren(header);
  $("table-body").replaceChildren(...state.rows.map(row => {
    const tr = document.createElement("tr");
    if (row.recovery_method === "ghost_interval_sequential_autogroup") tr.className = "ours";
    const first = document.createElement("td"); first.textContent = modeLabel(row); first.title = row.mode; tr.append(first);
    for (const col of columns) { const td = document.createElement("td"); td.textContent = row.display_metrics[col] ?? "—"; tr.append(td); }
    return tr;
  }));
  $("row-status").textContent = state.rows.length ? `${state.rows.length} configurations · ${names[$("backbone").value]}` : "No results for this filter combination.";
}
function downloadCSV() {
  if (!state.table) return;
  const quote = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = [["Model / configuration", ...state.table.columns], ...state.rows.map(r => [modeLabel(r), ...state.table.columns.map(c => r.display_metrics[c] ?? "")])].map(row => row.map(quote).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], {type:"text/csv;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = `lare_${$("view").value}_${$("backbone").value}.csv`; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function figureChoices() {
  const entries = state.index.figures.filter(f => f.kind === $("figure-kind").value && f.backbone === $("figure-backbone").value);
  options("figure-dataset", [...new Set(entries.map(f=>f.dataset))]);
  const similarity = $("figure-kind").value === "similarity";
  $("region-label").hidden = !similarity; $("metric-label").hidden = !similarity;
  renderFigure();
}
function renderFigure() {
  const item = state.index.figures.find(f => f.kind === $("figure-kind").value && f.backbone === $("figure-backbone").value && f.dataset === $("figure-dataset").value && (f.kind !== "similarity" || f.region === $("figure-region").value && f.metric === $("figure-metric").value));
  $("analysis-image").hidden = !item; $("figure-download").hidden = !item;
  $("figure-caption").textContent = item ? `${names[item.backbone]} · ${item.dataset} · ${item.kind}${item.region ? ` · ${item.region} tokens · raw ${item.metric} similarity` : ""}` : "No figure is available for this selection.";
  if (item) { $("analysis-image").src = `data/${item.file}`; $("analysis-image").alt = $("figure-caption").textContent; $("figure-download").href = `data/${item.file}`; }
}
let cloudViewer;
function createCloudViewer() {
  const canvas = $("point-cloud"), gl = canvas.getContext("webgl", {antialias:true});
  if (!gl) throw new Error("WebGL is unavailable. The numerical results and figures are still available above.");
  const shader = (kind, source) => { const s = gl.createShader(kind); gl.shaderSource(s,source); gl.compileShader(s); if (!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const program = gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,"attribute vec3 p;attribute vec3 c;varying vec3 color;uniform vec3 center;uniform float scale;uniform vec2 angle;uniform float zoom;uniform float aspect;uniform float size;void main(){vec3 v=(p-center)*scale;float cy=cos(angle.x),sy=sin(angle.x),cx=cos(angle.y),sx=sin(angle.y);v=vec3(cy*v.x+sy*v.z,v.y,-sy*v.x+cy*v.z);v=vec3(v.x,cx*v.y-sx*v.z,sx*v.y+cx*v.z);float d=3.0-v.z;gl_Position=vec4(v.x*zoom/aspect,v.y*zoom,-v.z*.3,d);gl_PointSize=size;color=c;}"));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,"precision mediump float;varying vec3 color;void main(){if(distance(gl_PointCoord,vec2(0.5))>0.5)discard;gl_FragColor=vec4(color,1.0);}")); gl.linkProgram(program);
  if (!gl.getProgramParameter(program,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program); gl.enable(gl.DEPTH_TEST); const buffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  for (const [attribute,offset] of [["p",0],["c",12]]) { const loc=gl.getAttribLocation(program,attribute); gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,24,offset); }
  const uniforms=Object.fromEntries(["center","scale","angle","zoom","aspect","size"].map(k=>[k,gl.getUniformLocation(program,k)]));
  let count=0,yaw=-0.3,pitch=0.25,zoom=2,drag=null;
  function draw(){const r=canvas.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(r.width*ratio);canvas.height=Math.round(r.height*ratio);gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.075,.15,.12,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniform2f(uniforms.angle,yaw,pitch);gl.uniform1f(uniforms.zoom,zoom);gl.uniform1f(uniforms.aspect,canvas.width/canvas.height);gl.uniform1f(uniforms.size,Number($("point-size").value)*ratio);gl.drawArrays(gl.POINTS,0,count);}
  canvas.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener("pointerup",()=>drag=null);canvas.addEventListener("pointercancel",()=>drag=null);canvas.addEventListener("pointermove",e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.007;pitch+=(e.clientY-drag[1])*.007;drag=[e.clientX,e.clientY];draw();});
  canvas.addEventListener("wheel",e=>{e.preventDefault();zoom=Math.max(.3,Math.min(10,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
  canvas.addEventListener("keydown",e=>{const steps={ArrowLeft:[-.1,0],ArrowRight:[.1,0],ArrowUp:[0,-.1],ArrowDown:[0,.1]};if(steps[e.key]){e.preventDefault();yaw+=steps[e.key][0];pitch+=steps[e.key][1];draw();}});
  $("reset-camera").onclick=()=>{yaw=-.3;pitch=.25;zoom=2;draw();};$("point-size").oninput=draw;new ResizeObserver(draw).observe(canvas);
  return {load(array){count=array.length/6;const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];for(let i=0;i<array.length;i+=6)for(let k=0;k<3;k++){min[k]=Math.min(min[k],array[i+k]);max[k]=Math.max(max[k],array[i+k]);}gl.bufferData(gl.ARRAY_BUFFER,array,gl.STATIC_DRAW);gl.uniform3fv(uniforms.center,min.map((v,k)=>(v+max[k])/2));gl.uniform1f(uniforms.scale,2/Math.max(...max.map((v,k)=>v-min[k]),.0001));draw();}};
}
async function loadCloud() {
  const request = ++state.cloudRequest;
  const scene=state.index.scenes.find(s=>s.backbone===$("scene-backbone").value);
  if (!scene) {$("scene-status").textContent="No exported scene available.";return;}
  const variant=scene.variants.find(v=>v.label===$("scene-variant").value)||scene.variants[0];
  $("scene-status").textContent="Loading sampled reconstruction…";
  const response=await fetch(`data/${variant.file}`);if(!response.ok)throw new Error(`Could not load scene: HTTP ${response.status}`);
  const bytes=await response.arrayBuffer();if(request!==state.cloudRequest)return;
  if(!cloudViewer)cloudViewer=createCloudViewer();cloudViewer.load(new Float32Array(bytes));
  $("scene-status").textContent=`${names[scene.backbone]} · ${scene.dataset} / ${scene.scene_id} · ${variant.label} · ${variant.points.toLocaleString()} displayed points`;
}
function modelCode(){const bb=$("model-backbone").value,model=state.index.models.models[bb];$("model-code").textContent=`from lare import LaRe\n\nmodel = LaRe.from_pretrained(\n    "${bb}",\n    checkpoint="checkpoints/${bb}/model.pt",\n    device="cuda",\n)\nimages = model.load_images([\n    "image_a.jpg", "image_b.jpg"\n])\npredictions = model(images)\npoints = predictions["points"]`;$("model-details").textContent=`${model.drop_count} dropped units · ${model.selector} · compact recovery ${(model.bytes/1024**2).toFixed(2)} MiB. Model files are distributed separately from this website.`;}
async function init(){
  state.index=await getJSON("index.json");const backbones=Object.keys(state.index.views.final.files);
  for(const id of ["backbone","figure-backbone","model-backbone"])options(id,backbones,"vggt_omega");
  options("scene-backbone",state.index.scenes.map(s=>s.backbone),"vggt_omega");options("scene-variant",["Original","LaRe"],"LaRe");
  $("policy").textContent=state.index.selection_policy;
  for(const id of ["view","backbone"])$(id).addEventListener("change",()=>loadTable().catch(e=>error(e.message)));
  for(const id of ["dataset","drop","method","compression"])$(id).addEventListener("change",renderTable);
  $("download-csv").onclick=downloadCSV;
  for(const id of ["figure-kind","figure-backbone"])$(id).addEventListener("change",figureChoices);
  for(const id of ["figure-dataset","figure-region","figure-metric"])$(id).addEventListener("change",renderFigure);
  for(const id of ["scene-backbone","scene-variant"])$(id).addEventListener("change",()=>loadCloud().catch(e=>$("scene-status").textContent=e.message));
  $("model-backbone").onchange=modelCode;$("copy-code").onclick=async()=>{try{await navigator.clipboard.writeText($("model-code").textContent);$("copy-code").textContent="Copied";setTimeout(()=>$("copy-code").textContent="Copy code",1500);}catch{$("copy-code").textContent="Select the code to copy";}};
  figureChoices();modelCode();await loadTable();loadCloud().catch(e=>$("scene-status").textContent=e.message);
}
init().catch(e=>error(`${e.message}. Serve the web/ directory over HTTP; see docs/webpage.md for setup.`));
