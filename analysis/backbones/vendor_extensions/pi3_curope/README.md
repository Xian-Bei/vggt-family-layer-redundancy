# Pi3 cuRoPE compatibility extension

Pi3 imports the optional CUDA RoPE implementation as `models.curope`.
LaRe keeps this compatibility package outside `backbones/Pi3` so the Pi3
repository can remain a clean, read-only upstream checkout.

The source files are copied from the cuRoPE extension used by the current Pi3
setup. If the extension is not compiled, Pi3 falls back to its slower PyTorch
RoPE implementation.

To build the optional CUDA extension locally:

```bash
cd analysis/backbones/vendor_extensions/pi3_curope/models/curope
python setup.py build_ext --inplace
```

Build products such as `*.so` and `build/` are intentionally ignored by Git.
