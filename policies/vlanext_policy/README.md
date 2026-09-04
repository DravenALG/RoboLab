# VLANeXt

VLANeXt uses the same WebSocket + msgpack/NumPy transport as OpenPI, while
serving DROID absolute end-effector actions instead of joint positions.

Start the model server from the `codebase` conda environment:

```bash
cd /data/NTU_slab/draven/proj/codebase
python scripts/serve_vlanext.py \
  --checkpoint /data/NTU_slab/draven/checkpoints/codebase/codebase_droid/steps100k_bs128_lr1e-5_qwen3vl2b_new_policy
```

Then start RoboLab in its own terminal:

```bash
cd /data/NTU_slab/draven/proj/RoboLab
uv run python policies/vlanext_policy/run.py --headless
```

Both sides default to port 8000. The client reads image size, history length,
view mode, input modality, and action horizon from the
checkpoint metadata. `--open-loop-horizon` can request more frequent replans.
