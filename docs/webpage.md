# Static results dashboard

`web/` runs entirely on GitHub Pages. It contains a project introduction and
six tabs featuring the selected parts of `analysis.view` and paper Table 1, named after
the paper's analyses: Layer Redundancy (four panels per backbone: indoor/outdoor
trajectory and geometry metrics), Approximate Additivity (the original and first
marginal landscape per backbone), Representation Proxies, Pruning &
Generalization (normalized degradation), Post-Pruning
Recovery (normalized degradation),
and Accuracy & Efficiency (the fixed paper Table 1). Existing URL identifiers remain
valid, with removed per-metric pages and the former recovery table redirecting
to the corresponding overview; the former radar URL redirects to the final table.
The public dashboard does not include 3D, ablation, attention replay,
model inference, or training.

The project introduction, authors, four findings and BibTeX citation follow
[the paper on arXiv](https://arxiv.org/abs/2609.36842). The paper section,
dashboard and citation use the same centered page container. The title wraps
naturally and the introduction uses the full container width. Author names link
to their personal pages; Xiangyu Sun links to the Google Scholar profile listed
on the [UQMM lab page](https://luoyadan.github.io/uqmm/). The arXiv and GitHub
links appear directly below the author affiliations. Dashboard pages
show controls and figure labels without repeating the tab title above them;
normal diagnostic text stays accessible to
screen readers, and loading failures remain visible.
Dataset names use ScanNet, nuScenes, 7Scenes, and Waymo in controls, row labels,
tables while retaining their original split identifiers
for filtering. Dataset row headings always show the full scope name; the
former simplified-label option is removed, including from restored shared views.
Tabs are numbered 1–6. The first three tabs omit dataset split suffixes;
Layer Redundancy has a single dataset selector (default Mean): Mean retains
indoor/outdoor averages, while a concrete dataset shows its two metric families.
Approximate Additivity and Representation Proxies use all four Dev datasets
(ScanNet, nuScenes, 7Scenes, Waymo) for Mean, matching the local dashboard and
publication PDF exports. Representation Proxies always uses the mean scene and
has no scene selector.
Pruning and recovery group dataset controls and rows by calibration, same-dataset
test, and cross-dataset test, with consistent Dev/Test suffixes. The recovery
dataset selector precedes the method selector. Unavailable recovery splits
(7Scenes Dev and Waymo Dev in `analysis_outputs_v3`) are omitted;
these splits do have separate final-experiment results under `main_results`. The calibration average remains
ScanNet Dev + nuScenes Dev, as defined in the experiment.
Pruning and recovery default to calibration-set,
same-dataset test, and cross-dataset test averages, with one shared legend above
all rows. ReplaceMe merging is always enabled for pruning; recovery initially
selects every available method. Reset controls restores these defaults.

The project header uses centered author links and dark, rounded resource
buttons, following the common research-project layout shown by
[VGGT](https://vgg-t.github.io/). Introductory paragraphs span the same content
width as the findings and dashboard. CSS and JavaScript URLs include content
hashes to bypass stale browser caches after an update. When editing either
asset, refresh its `?v=` value in `web/index.html` using the first 12 characters
of its SHA-256 digest. Restarting `python -m http.server` alone does not clear
the browser's cached assets.

Heatmap axes share fixed display slots. Curve axes are scaled uniformly to
common plot dimensions. Shared legends replace repeated
per-figure legends; method filtering toggles both line and scatter artists.
Column captions use 17px text and curve backbone captions use 19px text.
These are presentation changes; the cached values and axis scales are unchanged.

## Refresh the cached results

From the repository root, with the existing local experiment results:

```bash
conda run --no-capture-output -n vggt python -m analysis.results.export_dashboard
```

The exporter reads `analysis_outputs_v3`; the final table is fixed publication data
in `web/assets/paper-table.json` and does not need `main_results`. The default command
builds a complete snapshot in a temporary directory, then replaces `web/data/`
only after export succeeds. It does not rerun experiments or modify result
roots. `analysis.results.export_web` remains an alias for this command.

For a targeted refresh of an existing snapshot, use `--sections heatmaps`,
or `--sections pruning recovery`. These operate
in place; use the full command for publication. `--analysis-root`,
and `--output` can override the input/output locations.

The cache contains only:

- Final display matrices for the mean Similarity scene and each physical
  token region, plus Single-interval and Remaining heatmaps. Symmetric
  matrices are stored as inclusive upper triangles without reducing precision.
- Matplotlib SVG curves plus the method IDs and legend/filter styling used by the UI.
- The fixed paper table lives separately in `web/assets/`, outside the result export.
- Plot palettes and an asset checksum manifest.

The exporter calls the same callbacks and renderers as Gradio. Calibration
winner selection, normalization profiles, fixed dataset averages, heatmap
transformations, and MAE/Pearson/Spearman statistics are computed in Python.
Single-interval overview includes only individual datasets and the all-dataset mean;
its four columns use the callback's `split_pose_metrics=False` option, averaging
the four trajectory metrics together and the three geometry metrics together.
The browser does not approximate an average by blending finished images.

JavaScript filters the cached scopes and table rows, arranges panels, and
shows/hides methods in the original SVG curves. Method visibility keeps the
exported axis limits fixed. Tables compute absolute/percentage changes from
the same displayed numbers used by the Gradio table; absolute changes take
precedence when both switches are enabled. Empty selections and unavailable
results remain explicit.

Matrices use the original Matplotlib palettes, one-based layer tick labels,
lower-left coordinate origin, and the shared 48-layer canvas. Hover exposes
matrix values; PNG/SVG downloads and shareable
control selections work without a Python service.

`Refresh published data` reloads the published snapshot. It cannot inspect
new results on the author's computer; refresh the export and redeploy to
publish new measurements. `Load training eval best` exposes only training
results present at export time; it never runs training.

Weights, activation caches, original images, point clouds, backup files, and
complete experiment trees are excluded. Model export is independent of the
webpage and is not required to refresh it.

## Local preview

```bash
conda run --no-capture-output -n vggt python -m http.server 8000 --directory web
```

Open <http://localhost:8000>. Direct `file://` loading cannot fetch the cache.
Stop the server when finished. The frontend has no package-manager build or
external browser-library dependency.

## GitHub Pages deployment

Repository: <https://github.com/Xian-Bei/vggt-family-layer-redundancy>

Project URL: <https://xian-bei.github.io/vggt-family-layer-redundancy/>

1. Commit and push the source plus the complete `web/` directory to `main`.
2. In Settings → Pages, choose **GitHub Actions** as the deployment source.
3. In Actions, run **Project webpage** on `main`.
4. Wait for the deployment to succeed and open the project URL.

The existing `.github/workflows/pages.yml` uploads only `web/`. It never reads
local result roots or uploads model checkpoints. All asset paths are relative
and support the repository URL prefix. No Hugging Face Space, Python server,
GPU service, custom domain, or separate `xian-bei.github.io` repository is
required. Pushing alone does not trigger this manual deployment workflow.

The final tab displays the 17 rows of paper Table 1, averaged over four held-out
splits, without exploratory filters. Publication values, units (including latency
in seconds), and efficiency reductions are preserved in `web/assets/paper-table.json`.
This fixed snapshot is independent of live result exports. The former compressed
models tab redirects here. The teaser below the project links is rendered from
`docs/teaser.pdf`, with a link to the PDF copy in `web/assets/`.
Recovery controls use publication method names, omit Current Target (the plain
sequential variant), and label the autogroup variant Ours. Both pruning selectors
call the measured criterion Measured. Remaining plots also discover completed
points from validated result summaries when an older index omits them.

The exporter omits retired final curves and exploratory tables, unmerged pruning
curves, arbitrary dataset subsets, and the hidden sequential recovery method.
Finalization removes unreachable plot/matrix files and regenerates checksums,
including when refreshing an older snapshot.

SVG cleanup removes the hidden sequential-recovery artists and embedded legends
(the UI builds one shared legend). Duplicate coordinate arrays are omitted from
JSON because the visible paths already reside in SVG. A reachability audit checks
that all data files serve a current control selection or asset integrity check.
