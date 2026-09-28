# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Pytest configuration for RoboLab install-verification tests.

Boots Isaac Sim once at conftest load so test files can freely import
isaaclab/robolab modules at their top level. Auto-accepts the Omniverse
EULA so a fresh install works headless without any prompts.
"""

import os
import sys

# Accept the Omniverse EULA non-interactively. Must be set before any
# isaaclab import. setdefault → user can still override.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import cv2  # noqa: E402, F401  must be imported before isaaclab

from isaaclab.app import AppLauncher  # noqa: E402

# Launch Isaac Sim once for the whole pytest session. Carb's logger is told to
# only emit warnings or worse so install-verification output isn't drowned in
# Isaac Sim's [Info] startup/shutdown chatter.
_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    # Full plugin unload in Isaac Sim 5.0 leaves native UI subscriptions alive
    # until Python GC, where EditorMenu.__del__ can segfault. Keep Kit's default
    # fast shutdown, but let pytest own SystemExit and the final process status.
    fast_shutdown=True,
    kit_args="--/app/python/interceptSysExit=false",
    carb_settings={
        "/log/level": "warn",
        "/log/outputStreamLevel": "warn",
        "/log/fileLogLevel": "warn",
    },
)
simulation_app = _launcher.app


def pytest_addoption(parser):
    parser.addoption(
        "--task",
        default="BananaInBowlTask",
        help="Task class name for test_run_empty (default: BananaInBowlTask). "
             "The test runs the first registered env matching this task.",
    )
    parser.addoption(
        "--env-name",
        default=None,
        help="Full registered env name for test_run_empty (e.g. BananaInBowlTaskHomeOffice). "
             "If set, overrides --task.",
    )


import pytest


@pytest.fixture
def task_arg(request):
    return request.config.getoption("--task")


@pytest.fixture
def env_name_arg(request):
    return request.config.getoption("--env-name")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Write pytest reports before Kit exits, preserving success and failure."""
    outcome = yield
    outcome.get_result()
    # Fast shutdown exits from inside close(), so run this after all other
    # sessionfinish hooks (including terminal/JUnit reporting), and flush first.
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.app.post_quit(int(session.exitstatus))
    simulation_app.close()
