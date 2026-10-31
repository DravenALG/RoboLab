"""Evaluate a FLUX-action DROID server with RoboLab's standard evaluation loop."""

import argparse
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    # Conda's Mesa-only vendor directory can prevent NVIDIA EGL initialization.
    nvidia_egl = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if nvidia_egl.is_file():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(nvidia_egl))

    import cv2  # noqa: F401 -- must precede Isaac Lab
    from isaaclab.app import AppLauncher

    from robolab.eval.runner import add_common_eval_args, run_evaluation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--model-name", default="flux_action", help="Policy label in evaluation results.")
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.multi_gpu = False
    simulation_app = AppLauncher(args).app
    client = None
    exit_code = 0

    try:
        import robolab.constants
        from policies.flux_action.client import FluxActionClient
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args.enable_subtask
        auto_register_droid_envs(task_dirs=args.task_dirs, task=args.task, cameras=WRIST_LEFT_RIGHT_HEAD)

        def make_client(options):
            nonlocal client
            # The current runner constructs a client per task but does not close
            # it. Keep that lifecycle local to this entrypoint.
            if client is not None:
                client.close()
            client = FluxActionClient(
                remote_host=options.remote_host,
                remote_port=options.remote_port,
                model_name=options.model_name,
                log_dir=robolab.constants.get_output_dir(),
            )
            return client

        run_evaluation(args, policy=args.model_name, client_factory=make_client)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if client is not None:
            client.close()
        # Isaac Sim's fast shutdown may exit before main returns.
        simulation_app.app.post_quit(exit_code)
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
