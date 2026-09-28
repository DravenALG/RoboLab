# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check duration overrides before environment construction and in saved metadata."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from isaaclab.envs import ManagerBasedRLEnvCfg

from robolab.core.environments import runtime


@pytest.mark.parametrize("scene_kind", ["name", "config"])
@pytest.mark.parametrize("duration", [None, 100.0, 0.5])
def test_episode_length_applied_before_construction_and_saved(scene_kind, duration, tmp_path, monkeypatch):
    cfg = ManagerBasedRLEnvCfg(episode_length_s=50.0, decimation=8, seed=0)
    cfg.instruction = "Test instruction"
    cfg.scene = SimpleNamespace()
    expected = 50.0 if duration is None else duration
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

    actual_env, actual_cfg = runtime.create_env(scene, episode_length_s=duration)

    assert actual_env is env
    assert actual_cfg.episode_length_s == expected
    saved_cfg = json.loads((tmp_path / "env_cfg.json").read_text())
    assert saved_cfg["episode_length_s"] == expected


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_invalid_episode_length_rejected_before_stage_creation(duration, monkeypatch):
    get_context = Mock()
    monkeypatch.setattr(runtime.omni.usd, "get_context", get_context)

    with pytest.raises(ValueError, match="episode_length_s must be finite and positive"):
        runtime.create_env("TestTask", episode_length_s=duration)

    get_context.assert_not_called()


def test_droid_episode_length_controls_step_limit_and_timeout(tmp_path, monkeypatch):
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
    env, cfg = runtime.create_env(env_name, num_envs=1, episode_length_s=100)
    try:
        assert cfg.episode_length_s == 100
        assert env.max_episode_length == 1500
        saved_cfg = json.loads((tmp_path / "env_cfg.json").read_text())
        assert saved_cfg["episode_length_s"] == 100

        env.episode_length_buf.fill_(1499)
        assert not mdp.time_out(env).any()
        env.episode_length_buf.fill_(1500)
        assert mdp.time_out(env).all()
    finally:
        env.close()
