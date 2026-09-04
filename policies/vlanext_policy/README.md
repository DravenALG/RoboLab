# VLANeXt

VLANeXt uses the same WebSocket + msgpack/NumPy transport as OpenPI. Cartesian
checkpoints predict absolute end-effector actions. Joint checkpoints predict
joint deltas relative to the latest observation; the server converts them to
absolute joint-position targets before sending them to RoboLab. New DROID
checkpoints use OpenPI-style q01/q99 action normalization.

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

# For a Cartesian checkpoint (joint-position is the default):
uv run python policies/vlanext_policy/run.py --action-mode cartesian --headless
```

Both sides default to port 8000. The client reads image size, history length,
view mode, input modality, and action horizon from the
checkpoint metadata. `--action-mode` must match the served checkpoint, and
`--open-loop-horizon` can request more frequent replans.
