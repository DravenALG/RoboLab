"""Evaluate a VLANeXt DROID absolute-EEF policy in RoboLab."""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must be imported before IsaacLab
from isaaclab.app import AppLauncher

POLICY = "vlanext"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--remote-host", "--remote_host", default="localhost")
parser.add_argument("--remote-port", "--remote_port", default=8000, type=int)
parser.add_argument("--remote-uri", "--remote_uri", default=None)
parser.add_argument(
    "--open-loop-horizon",
    "--open_loop_horizon",
    default=None,
    type=int,
    help="Actions executed per server query; defaults to the checkpoint horizon.",
)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.vlanext_policy.client import VLANeXtDroidEEFClient  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_abs_ik import (  # noqa: E402
    auto_register_droid_abs_ik_envs,
)

auto_register_droid_abs_ik_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)


def make_client(args: argparse.Namespace) -> VLANeXtDroidEEFClient:
    return VLANeXtDroidEEFClient(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
    )


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\033[96m[RoboLab] Terminated with error: {error}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
