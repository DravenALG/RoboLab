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

For deterministic crop comparisons, pass `--center-crop-ratio 0.95`. After
the usual resize, both camera views retain the centered 95% of each dimension
and are resized back with bilinear interpolation, preserving the image shape.
The default `1.0` leaves images uncropped. This setting is independent of the
checkpoint's training augmentation configuration.

Inference sweeps can override denoising without changing the training schedule:

```bash
python scripts/serve_vlanext.py --checkpoint /path/to/checkpoint \
  --num-inference-timesteps 20 --port 8001 --device cuda:1
```

The client can connect with `--remote-port 8001 --inference-batch-size 20`
to batch parallel environments on servers that advertise batch support.
Batching defaults to 1 for compatibility with existing servers. Use
`--open-loop-horizon 4` to replan after four actions from each predicted chunk.
For joint policies, `--gripper-threshold 0.3` changes the threshold for closing
the gripper (default 0.5; larger values close less readily). Keep these settings
fixed when comparing denoising steps, and restart the experiment server with
the same `--seed` for each comparison.
