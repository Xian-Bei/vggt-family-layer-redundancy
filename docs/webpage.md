# Static project webpage

`web/` is a buildless HTML/CSS/JavaScript site. It runs on GitHub Pages without
Gradio, a Python server, a package manager, or third-party browser libraries.
All paths are relative, so repository project URLs are supported.

## Refresh visualization data

From the repository root:

```bash
conda run --no-capture-output -n vggt python -m lare.export
conda run --no-capture-output -n vggt python -m analysis.results.export_web
```

The export reads `analysis_outputs_v3` and `main_results`. It emits only:

- Result tables split by view and backbone, including existing display values
  and raw seven-metric errors.
- Pruning-curve payloads, similarity mean matrices, and analysis figures.
- A selected held-out ScanNet scene per backbone, with Original/LaRe point
  samples of at most 30,000 points each.
- Small manifests with sizes and SHA-256 checksums.

The table export reuses the Python aggregation and calibration winner rules
used by Gradio. Compression variants follow the uncompressed recovery winner.
Final recovery figures reuse the analysis normalization profiles. The browser
only filters and displays the exported results; it does not compute scientific
averages, normalize errors, or select pruning winners.

Similarity panels display raw mean similarity, explicitly labeled as such.
The local Gradio interface additionally supports transformed views and other
research controls. Point samples use a fixed seed and confidence filtering;
they are display assets, not the quantitative evaluation input.

Large model weights, full-resolution point clouds, activation caches, and
complete experiment output trees are excluded from `web/`.

## Local preview

```bash
conda run --no-capture-output -n vggt python -m http.server 8000 --directory web
```

Open `http://localhost:8000`. Direct `file://` loading cannot fetch the data.
Stop the server after use. The 3D viewer supports pointer orbit, wheel zoom,
arrow-key orbit, point-size adjustment, and reset.

## GitHub Pages

Repository: <https://github.com/Xian-Bei/vggt-family-layer-redundancy>

Project webpage after successful deployment:
<https://xian-bei.github.io/vggt-family-layer-redundancy/>

1. Commit and push the prepared public source to the repository's `main`
   branch. The website deployment requires `.github/workflows/pages.yml`
   and the complete `web/` directory, including `web/data/` and `.nojekyll`.
   The model setup link also expects the project README on that branch.
   Include the `.gitignore` exception for `web/data/scenes/*.bin`: these six
   sampled point clouds must be committed with the rest of the website data.
2. Open [Settings → Pages](https://github.com/Xian-Bei/vggt-family-layer-redundancy/settings/pages).
   Under **Build and deployment → Source**, select **GitHub Actions**.
   The existing workflow is sufficient; no new template is needed.
3. Open [Actions](https://github.com/Xian-Bei/vggt-family-layer-redundancy/actions),
   select **Project webpage**, click **Run workflow**, choose `main`, and
   confirm **Run workflow**. The workflow file must be on the default branch
   before its manual trigger is available.
4. Wait for the `deploy` job to succeed. Its deployment URL should be the
   project webpage shown above. Open that URL after the job completes.

The workflow is manual: pushing changes does not publish the site by itself.
For subsequent updates, push the changed `web/` files and run the same
workflow again. Editing only HTML/CSS/JavaScript does not require re-exporting
the numerical data; rerun the export when experiment results change.

This is a repository project site. It does not require a separate
`xian-bei.github.io` repository or a custom domain/CNAME. The site uses relative
asset URLs, which already support the `/vggt-family-layer-redundancy/` prefix.

The workflow uploads only `web/`. It never exports research outputs or uploads
model checkpoints from the working machine. Model download URLs must be added
after the artifacts have actually been published.

The workflow follows the official
[GitHub Pages custom-workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
