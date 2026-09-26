"""Evaluate a VLANeXt DROID Cartesian or joint-position policy in RoboLab."""

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
parser.add_argument("--inference-batch-size", type=int, default=1,
                    help="Maximum environments per inference request; requires a batch-capable server.")
parser.add_argument("--gripper-threshold", type=float, default=0.5,
                    help="Binarization threshold for joint-policy gripper commands.")
parser.add_argument(
    "--action-mode",
    "--action_mode",
    choices=("cartesian", "joint"),
    default="joint",
    help="DROID control space; must match the served checkpoint.",
)
parser.add_argument(
    "--center-crop-ratio",
    type=float,
    default=1.0,
    help="Centered fraction of each image dimension after resize; 1.0 disables cropping.",
)
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
if not 0.0 < args_cli.center_crop_ratio <= 1.0:
    parser.error("--center-crop-ratio must be in (0, 1]")
if args_cli.inference_batch_size < 1:
    parser.error("--inference-batch-size must be positive")
if not 0.0 < args_cli.gripper_threshold < 1.0:
    parser.error("--gripper-threshold must be in (0, 1)")
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.vlanext_policy.client import VLANeXtDroidClient  # noqa: E402

if args_cli.action_mode == "joint":
    from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
        auto_register_droid_envs,
    )

    auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)
else:
    from robolab.registrations.droid.auto_env_registrations_abs_ik import (  # noqa: E402
        auto_register_droid_abs_ik_envs,
    )

    auto_register_droid_abs_ik_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)


def make_client(args: argparse.Namespace) -> VLANeXtDroidClient:
    return VLANeXtDroidClient(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
        action_mode=args.action_mode,
        center_crop_ratio=args.center_crop_ratio,
        inference_batch_size=args.inference_batch_size,
        gripper_threshold=args.gripper_threshold,
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
