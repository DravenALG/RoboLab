# FLUX-action

This client evaluates a FLUX-action DROID checkpoint using the standard RoboLab
runner. It shares Cosmos3's camera composite, absolute joint action conversion,
gripper binarization and 32-step open-loop horizon. Model inference runs in a
separate Python 3.12 environment; RoboLab uses its Python 3.11 environment.

With RoboLab and `openpi-client` installed, start the server from the sibling
`flux-action` checkout (see its `docs/setup.md` for installation):

```bash
cd ../flux-action
CUDA_VISIBLE_DEVICES=1 .venv/bin/flux-action serve-robolab \
  --checkpoint black-forest-labs/flux-3-action-droid --port 8000
```

In a separate terminal, from the RoboLab root:

```bash
.venv/bin/python policies/flux_action/run.py \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 --headless --device cuda:0 \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --output-folder-name BananaInBowlTask_flux_action_test
```

Omit `--task` to evaluate all registered benchmark tasks. Reuse the output folder
and the original `--num-envs` to resume. `--model-name` changes the policy label
(default: `flux_action`); select the checkpoint or variant on the server.

Each task directory includes `server_metadata.json` and
`inference_requests.jsonl`, alongside the standard scores, trajectories and videos.
RPC timing includes transport and retries; `server_timing.infer_ms` measures
server inference. Only actual server queries are logged, not cached action steps.

Client regression tests do not require a simulator or checkpoint:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest --noconftest tests/test_flux_action_policy.py -q
```
