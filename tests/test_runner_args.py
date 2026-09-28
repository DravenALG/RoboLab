# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the eval runner's argparse wiring against AppLauncher collisions.

The per-policy runners build their parser as::

    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)

IsaacLab 2.3's ``add_app_launcher_args`` raises ``ValueError`` if the parser
already declares a field it owns (e.g. ``rendering_mode``). This never shows up
in the other tests because they drive ``AppLauncher`` directly rather than
through the runner's parser — so it's asserted explicitly here.
"""

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from isaaclab.app import AppLauncher

from robolab.eval.runner import add_common_eval_args, run_evaluation


def test_caught_system_exit_does_not_request_simulator_shutdown(monkeypatch):
    import omni.kit.app

    kit_app = Mock()
    monkeypatch.setattr(omni.kit.app, "get_app", lambda: kit_app)
    with pytest.raises(SystemExit) as error:
        sys.exit(2)
    assert error.value.code == 2
    kit_app.post_quit.assert_not_called()


def test_common_eval_args_do_not_collide_with_app_launcher():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    # Must not raise: AppLauncher owns `rendering_mode`, so add_common_eval_args
    # must not declare it (or any other AppLauncher-owned field).
    AppLauncher.add_app_launcher_args(parser)

    dests = {action.dest for action in parser._actions}
    # `renderer` and `rendering_type` are ours; `rendering_mode` is owned by
    # AppLauncher. run_evaluation() consumes args.renderer and args.rendering_type,
    # so both of ours must be present, and AppLauncher's field must coexist.
    assert "renderer" in dests
    assert "rendering_type" in dests
    assert "rendering_mode" in dests
    assert "enable_gt_state" in dests
    assert "episode_length_s" in dests


def test_episode_length_defaults_to_task_duration():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    assert parser.parse_args([]).episode_length_s is None


@pytest.mark.parametrize("flag", ["--episode-length-s", "--episode_length_s"])
@pytest.mark.parametrize("value", ["100", "0.5"])
def test_episode_length_accepts_positive_seconds(flag, value):
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    assert parser.parse_args([flag, value]).episode_length_s == float(value)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "invalid"])
def test_episode_length_rejects_invalid_seconds(value):
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args([f"--episode-length-s={value}"])
    assert error.value.code == 2


@pytest.mark.parametrize("duration", [None, 100.0])
def test_runner_passes_episode_length_to_every_task(duration, tmp_path, monkeypatch):
    import robolab.constants as constants
    from robolab.core.environments import factory, runtime
    from robolab.core.logging import results
    from robolab.core.utils import print_utils
    from robolab.eval import episode, summarize

    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    args = parser.parse_args([] if duration is None else ["--episode-length-s", str(duration)])
    args.device = "cuda:0"
    args.headless = True
    env = Mock()
    env_cfg = SimpleNamespace(instruction="Test task", subtasks=[])
    create_env = Mock(return_value=(env, env_cfg))
    monkeypatch.setattr(constants, "PACKAGE_DIR", str(tmp_path))
    monkeypatch.setattr(constants, "_output_dir", None)
    monkeypatch.setattr(factory, "get_envs", Mock(return_value=["TaskA", "TaskB"]))
    monkeypatch.setattr(runtime, "create_env", create_env)
    monkeypatch.setattr(results, "init_experiment", Mock(return_value=("results.json", {})))
    monkeypatch.setattr(results, "check_all_episodes_complete", Mock(return_value=False))
    monkeypatch.setattr(results, "check_run_complete", Mock(return_value=False))
    monkeypatch.setattr(results, "summarize_experiment_results", Mock())
    monkeypatch.setattr(print_utils, "print_experiment_summary", Mock())
    monkeypatch.setattr(episode, "run_episode", Mock(return_value=({}, [], {})))
    monkeypatch.setattr(summarize, "summarize_run", Mock(return_value={}))

    run_evaluation(args, policy="test", client_factory=Mock())

    assert [call.args[0] for call in create_env.call_args_list] == ["TaskA", "TaskB"]
    assert all(call.kwargs["episode_length_s"] == duration for call in create_env.call_args_list)
