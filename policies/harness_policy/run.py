"""Evaluate the visual EEF harness once per task with RoboLab absolute IK."""

import argparse
from pathlib import Path
import sys

import cv2  # noqa: F401 -- must be imported before IsaacLab

# Support direct execution from either repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policies.harness_policy.client import HarnessClient, validate_eval_args
from robolab.eval.runner import add_common_eval_args, run_evaluation


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="localhost")
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--request-timeout", type=float, default=180,
                        help="WebSocket response timeout; must exceed the model API timeout.")
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        validate_eval_args(args)
        if args.request_timeout <= 0:
            raise ValueError("request-timeout must be positive")
    except ValueError as error:
        parser.error(str(error))
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app
    clients = []
    try:
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs

        auto_register_droid_abs_ik_envs(task_dirs=args.task_dirs, task=args.task)

        def make_client(eval_args):
            # The shared runner constructs one client per task but doesn't close
            # its connections; close the preceding task's client here.
            if clients:
                clients.pop().close()
            client = HarnessClient(eval_args.remote_host, eval_args.remote_port, eval_args.request_timeout)
            clients.append(client)
            return client

        run_evaluation(args, policy="harness", client_factory=make_client)
    finally:
        for client in clients:
            client.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
