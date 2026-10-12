# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check per-task duration scaling, saved metadata, and simulator step limits."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from isaaclab.envs import ManagerBasedRLEnvCfg

from robolab.core.environments import runtime


@pytest.mark.parametrize("scene_kind", ["name", "config"])
@pytest.mark.parametrize("default_duration", [50.0, 120.0])
@pytest.mark.parametrize("scale", [None, 1.0, 0.5, 2.0])
def test_episode_length_scaled_before_construction_and_saved(
    scene_kind, default_duration, scale, tmp_path, monkeypatch
):
    cfg = ManagerBasedRLEnvCfg(episode_length_s=default_duration, decimation=8, seed=0)
    cfg.instruction = "Test instruction"
    cfg.scene = SimpleNamespace()
    expected = default_duration * (1.0 if scale is None else scale)
    env = SimpleNamespace(scene=SimpleNamespace(articulations={"robot": object()}), sim=SimpleNamespace())
    env.unwrapped = env

    def construct(env_cfg):
        assert env_cfg.episode_length_s == expected
        return env

    monkeypatch.setattr(runtime.omni.usd, "get_context", Mock())
    monkeypatch.setattr(runtime, "parse_env_cfg", Mock(return_value=cfg))
    monkeypatch.setattr(runtime.gym, "make", lambda scene, cfg: construct(cfg))
    monkeypatch.setattr(runtime, "RobolabEnv", construct)
    monkeypatch.setattr(runtime, "get_output_dir", lambda: str(tmp_path))
    scene = "TestTask" if scene_kind == "name" else cfg

    kwargs = {} if scale is None else {"episode_length_scale": scale}
    # Reusing a configuration must start from its original duration each time.
    for _ in range(2 if scene_kind == "config" else 1):
        actual_env, actual_cfg = runtime.create_env(scene, **kwargs)
        assert actual_env is env
        assert actual_cfg.episode_length_s == expected
        saved_cfg = json.loads((tmp_path / "env_cfg.json").read_text())
        assert saved_cfg["episode_length_s"] == expected
        if scene_kind == "config":
            assert actual_cfg is not cfg
            assert cfg.episode_length_s == default_duration


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_invalid_episode_length_scale_rejected_before_stage_creation(scale, monkeypatch):
    get_context = Mock()
    monkeypatch.setattr(runtime.omni.usd, "get_context", get_context)

    with pytest.raises(ValueError, match="episode_length_scale must be finite and positive"):
        runtime.create_env("TestTask", episode_length_scale=scale)

    get_context.assert_not_called()


@pytest.mark.parametrize("scale,expected_steps", [(1.0, 750), (0.5, 375), (2.0, 1500), (0.501, 376)])
def test_droid_episode_length_scale_controls_step_limit_and_timeout(scale, expected_steps, tmp_path, monkeypatch):
    import torch
    from isaaclab.envs import mdp

    import robolab.constants as constants
    from robolab.core.environments.factory import get_envs
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for the Droid environment")

    monkeypatch.setattr(constants, "_output_dir", str(tmp_path))
    auto_register_droid_envs(task=["BananaInBowlTask"])
    env_name = get_envs(task="BananaInBowlTask")[0]
    env, cfg = runtime.create_env(env_name, num_envs=1, episode_length_scale=scale)
    try:
        assert cfg.episode_length_s == 50 * scale
        assert env.max_episode_length == expected_steps
        saved_cfg = json.loads((tmp_path / "env_cfg.json").read_text())
        assert saved_cfg["episode_length_s"] == 50 * scale

        env.episode_length_buf.fill_(expected_steps - 1)
        assert not mdp.time_out(env).any()
        env.episode_length_buf.fill_(expected_steps)
        assert mdp.time_out(env).all()
    finally:
        env.close()
