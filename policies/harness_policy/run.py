"""Evaluate independent harness runs with negotiated EEF or hybrid EEF/joint control."""

import argparse
from pathlib import Path
import sys

import cv2  # noqa: F401 -- must be imported before IsaacLab

# Support direct execution from either repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policies.harness_policy.client import HarnessClient, validate_eval_args
from policies.harness_policy.resume import prepare_output
from policies.harness_policy.batch_resume import read_episode_plan
from robolab.constants import PACKAGE_DIR
from robolab.eval.runner import add_common_eval_args, run_evaluation


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use --resume with an existing output directory and the original evaluation settings. "
               "Completed episodes (including failures) are skipped; interrupted episodes restart.",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Continue --output-folder-name; --num-runs remains the target total per task.")
    parser.add_argument("--episode-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--remote-host", default="localhost")
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--request-timeout", type=float, default=2400,
                        help="Time to receive the next action or episode-end acknowledgement; allow for model retries.")
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        validate_eval_args(args)
        selection = read_episode_plan(args.episode_plan, args.num_runs) if args.episode_plan else None
        if selection is not None and (not args.task or not set(args.task) <= {t for t, _ in selection}):
            raise ValueError("Selected tasks must be present in the episode plan")
        if args.request_timeout <= 0:
            raise ValueError("request-timeout must be positive")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app
    clients = []
    try:
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs

        # Negotiate the controller before the runner constructs any environment.
        client = HarnessClient(args.remote_host, args.remote_port, args.request_timeout)
        clients.append(client)
        actions_cfg = None
        if client.hybrid:
            from policies.harness_policy.hybrid_actions import DroidHarnessActionCfg
            actions_cfg = DroidHarnessActionCfg()
        auto_register_droid_abs_ik_envs(task_dirs=args.task_dirs, task=args.task, actions_cfg=actions_cfg)

        if args.resume:
            from robolab.core.environments.factory import get_envs

            if args.task:
                tasks = get_envs(task=args.task)
            elif args.tag:
                tasks = get_envs(tag=args.tag)
            else:
                tasks = get_envs()
            archive = prepare_output(Path(PACKAGE_DIR) / "output" / args.output_folder_name,
                                     tasks, args.num_runs)
            if archive is not None:
                print(f"Archived interrupted output: {archive}")

        def make_client(eval_args):
            # Episode hooks reset state and create independent Codex sessions;
            # reuse the negotiated connection across tasks.
            return client

        run_evaluation(args, policy="harness", client_factory=make_client, episode_selection=selection)
    finally:
        for client in clients:
            client.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
