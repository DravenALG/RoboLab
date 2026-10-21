"""RoboLab adapter for EEF or hybrid EEF/joint harness actions and public observations."""

from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from openpi_client import image_tools, msgpack_numpy
from PIL import Image
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

from robolab.constants import PACKAGE_DIR
from robolab.eval.base_client import InferenceClient
from robolab.eval.errors import PolicyExecutionError
from policies.harness_policy.resume import inspect_output


ACTION_REPRESENTATION = "robolab_base_link_absolute_xyz_quat_wxyz_gripper"
HYBRID_ACTION_REPRESENTATION = "robolab_mode_absolute_xyz_quat_wxyz_or_jointpos_gripper"


def validate_eval_args(args) -> None:
    """Evaluate a fixed number of independent, sequential runs with normal scoring."""
    if args.num_envs != 1:
        raise ValueError("The harness requires --num-envs 1")
    if isinstance(args.num_runs, bool) or not isinstance(args.num_runs, int) or args.num_runs < 1:
        raise ValueError("The harness requires a positive integer for --num-runs")
    if args.num_episodes_adaptive is not None:
        raise ValueError("Adaptive additional episodes are disabled for the harness")
    if args.enable_gt_state:
        raise ValueError("Ground-truth input is disabled for the visual harness")
    if not args.enable_subtask:
        raise ValueError("Keep the benchmark's subtask evaluation enabled")
    resume = getattr(args, "resume", False)
    if resume and args.output_folder_name is None:
        raise ValueError("--resume requires an explicit --output-folder-name")
    if args.output_folder_name is not None:
        if not args.output_folder_name.strip():
            raise ValueError("--output-folder-name must not be empty")
        output_dir = Path(PACKAGE_DIR) / "output" / args.output_folder_name
        if resume:
            if not output_dir.is_dir() or output_dir.is_symlink():
                raise ValueError(f"--resume requires an existing directory: {output_dir}")
            progress = inspect_output(output_dir)
            progress.pending(getattr(args, "task", None) or [], args.num_runs)
        elif output_dir.exists() or output_dir.is_symlink():
            raise ValueError(
                f"Output path already exists: {output_dir}. "
                "Choose a new --output-folder-name or pass --resume; resume is disabled by default."
            )


class HarnessClient(InferenceClient):
    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000, timeout: float = 2400) -> None:
        super().__init__()
        self.timeout = timeout
        self._url = f"ws://{remote_host}:{remote_port}"
        self._packer = msgpack_numpy.Packer()
        self._episode_id = None
        self._result_episode_id = None
        self._decision_step = 0
        self._control_step = 0
        self._ended = True
        self._reconnect = False
        self._resources = ExitStack()
        self._open_connection()

    def _open_connection(self) -> None:
        # Only a new episode may replace a deliberately discarded connection.
        # Never reconnect/resend an interrupted decision.
        previous = getattr(self, "metadata", None)
        deadline = time.monotonic() + 10
        try:
            while True:
                self._connection = self._resources.enter_context(connect(
                    self._url, compression=None, max_size=None,
                    open_timeout=10, proxy=None, ping_interval=None,
                ))
                try:
                    self.metadata = msgpack_numpy.unpackb(self._connection.recv(timeout=self.timeout))
                    break
                except ConnectionClosed as error:
                    self._resources.close()
                    if not self._reconnect or error.rcvd is None or error.rcvd.code != 1013 or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)  # Previous episode's server cleanup is finishing.
            if previous is not None and self.metadata != previous:
                raise ValueError("Harness metadata changed between episodes")
            version = self.metadata.get("protocol_version")
            if version not in (2, 3):
                raise ValueError("Harness protocol version 2 or 3 is required; update client and server together")
            self.hybrid = version == 3
            self.vla_enabled = self.metadata.get("vla_enabled", self.hybrid)
            representation = HYBRID_ACTION_REPRESENTATION if self.hybrid else ACTION_REPRESENTATION
            self.action_dim = 9 if self.hybrid else 8
            if self.metadata.get("action_representation") != representation:
                raise ValueError("Server action representation does not match the harness protocol")
            if self.metadata.get("action_dim") != self.action_dim:
                raise ValueError(f"Server action_dim must be {self.action_dim}")
            self.image_size = int(self.metadata["image_size"])
            if self.image_size < 1:
                raise ValueError("Server image_size must be positive")
        except Exception:
            self.close()
            raise
        self._reconnect = False

    def begin_episode(self, episode_idx: int, *, max_steps: int | None = None,
                      control_dt: float | None = None) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("Harness episodes require a positive integer max_steps")
        if control_dt is None or not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("Harness episodes require positive finite control_dt")
        super().begin_episode(episode_idx, max_steps=max_steps, control_dt=control_dt)
        self.reset()
        if self._reconnect:
            self._open_connection()
        self._max_steps, self._control_dt = max_steps, control_dt
        self._episode_id = f"{uuid4().hex}-{episode_idx}"
        self._result_episode_id = self._episode_id
        self._ended = False

    def end_episode(self, observation, *, actual_steps: int, reason: str) -> None:
        if self._episode_id is None or self._ended:
            return
        try:
            # No server-side episode exists before the first successful inference.
            if self._decision_step:
                self._query_server({
                    "type": "end_episode", "episode_id": self._episode_id, "env_id": 0,
                    "decision_step": self._decision_step, "control_step": actual_steps,
                    "observation": self._extract_observation(observation), "reason": reason,
                })
        finally:
            self._ended = True
            self._chunks.clear()
            self._counters.clear()

    def episode_result_metadata(self) -> dict:
        # run_episode resets action state before the runner commits its result.
        return {"harness_episode_id": self._result_episode_id} if self._result_episode_id else {}

    def reset(self, *, env_id: int | None = None) -> None:
        if self._episode_id is not None and not self._ended:
            raise RuntimeError("end_episode must be called before reset")
        super().reset(env_id=env_id)
        self._episode_id = None
        self._decision_step = 0
        self._control_step = 0

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= len(self._chunks[env_id])

    def _next_action(self, env_id: int) -> np.ndarray:
        if self._ended or self._control_step >= self._max_steps:
            raise RuntimeError("Episode has ended or exhausted its control steps")
        action = super()._next_action(env_id)
        self._control_step += 1
        return action

    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        if env_id != 0:
            raise ValueError("The minimal harness supports only env_id=0")
        proprio = raw_obs["proprio_obs"]
        observation = {
            "position": self._to_numpy(proprio["ee_pos"], env_id),
            "quaternion": self._to_numpy(proprio["ee_quat"], env_id),
            "joint_positions": self._to_numpy(proprio["arm_joint_pos"], env_id),
            "gripper": float(self._to_numpy(proprio["gripper_pos"], env_id).reshape(-1)[0]),
        }
        if self.vla_enabled:
            observation["vla_images"] = {}
        for output_name, camera in (("exterior_image", "over_shoulder_left_camera"), ("wrist_image", "wrist_cam")):
            raw_image = self._to_numpy(raw_obs["image_obs"][camera], env_id)
            if self.vla_enabled:
                observation["vla_images"][output_name] = image_tools.resize_with_pad(raw_image, 224, 224)
            image = Image.fromarray(raw_image)
            image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            observation[output_name] = np.asarray(image)
        return observation

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        if self._episode_id is None:
            raise RuntimeError("begin_episode must be called before inference")
        return {
            "type": "infer",
            "episode_id": self._episode_id,
            "env_id": 0,
            "decision_step": self._decision_step,
            "control_step": self._control_step,
            "max_steps": self._max_steps,
            "control_dt": self._control_dt,
            "instruction": instruction,
            "observation": extracted_obs,
        }

    def _query_server(self, request: dict) -> dict:
        self._connection.send(self._packer.pack(request))
        response = self._connection.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Harness execution error (no retry):\n{response}")
        try:
            result = msgpack_numpy.unpackb(response)
            if not isinstance(result, dict):
                raise ValueError("Harness response must be an object")
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self._policy_failure(str(error), reconnect=True)
        if "error" in result:
            error = result["error"]
            if (not isinstance(error, dict) or error.get("kind") != "policy_execution"
                    or not isinstance(error.get("type"), str) or not isinstance(error.get("message"), str)
                    or result.get("episode_id") != self._episode_id):
                self._policy_failure("Invalid harness error response", reconnect=True)
            self._policy_failure(error["message"], error_type=error["type"])
        return result

    def _policy_failure(self, message: str, *, error_type: str = "PolicyProtocolError", reconnect=False):
        self._ended = True
        self._chunks.clear()
        self._counters.clear()
        if reconnect:
            self.close()
            self._reconnect = True
        raise PolicyExecutionError(message, error_type=error_type)

    def _unpack_response(self, response: dict) -> np.ndarray:
        try:
            return self._validate_response(response)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self._policy_failure(str(error), reconnect=True)

    def _validate_response(self, response: dict) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=np.float32)
        if (actions.ndim != 2 or actions.shape[1] != self.action_dim
                or not 1 <= len(actions) <= self._max_steps - self._control_step
                or not np.isfinite(actions).all()):
            raise ValueError("Invalid harness action chunk")
        if self.hybrid:
            if not np.isin(actions[:, 0], [0, 1]).all() or not np.all(actions[:, 0] == actions[0, 0]):
                raise ValueError("A harness chunk must contain one valid control mode (0=EEF, 1=joints)")
            poses = actions[actions[:, 0] == 0, 1:]
        else:
            poses = actions
        if not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, atol=1e-4):
            raise ValueError("Harness action quaternions must be normalized")
        if not np.isin(actions[:, -1], [0, 1]).all():
            raise ValueError("Harness gripper actions must be binary")
        self._decision_step += 1
        return actions

    def _build_visualization(self, extracted_obs):
        return extracted_obs["exterior_image"]

    def close(self) -> None:
        self._resources.close()
