"""Thin RoboLab adapter for the visual EEF harness; no privileged observations."""

from __future__ import annotations

from contextlib import ExitStack
from uuid import uuid4

import numpy as np
from openpi_client import msgpack_numpy
from PIL import Image
from websockets.sync.client import connect

from robolab.eval.base_client import InferenceClient


ACTION_REPRESENTATION = "robolab_base_link_absolute_xyz_quat_wxyz_gripper"


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
    if args.output_folder_name is not None:
        raise ValueError("The minimal harness requires a fresh output directory; resume is disabled")


class HarnessClient(InferenceClient):
    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000, timeout: float = 2400) -> None:
        super().__init__()
        self.timeout = timeout
        # Connect once, and never reconnect/resend a decision automatically.
        self._resources = ExitStack()
        self._connection = self._resources.enter_context(connect(
            f"ws://{remote_host}:{remote_port}", compression=None, max_size=None,
            open_timeout=10, proxy=None, ping_interval=None,
        ))
        self._packer = msgpack_numpy.Packer()
        try:
            self.metadata = msgpack_numpy.unpackb(self._connection.recv(timeout=timeout))
            if self.metadata.get("action_representation") != ACTION_REPRESENTATION:
                raise ValueError("Server must provide RoboLab base_link absolute quaternion actions")
            if self.metadata.get("action_dim") != 8:
                raise ValueError("Server action_dim must be 8")
            self.open_loop_horizon = int(self.metadata["action_horizon"])
            self.image_size = int(self.metadata["image_size"])
            if self.open_loop_horizon < 1 or self.image_size < 1:
                raise ValueError("Server horizon and image_size must be positive")
        except Exception:
            self.close()
            raise
        self._episode_id = None
        self._decision_step = 0

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self.reset()
        self._episode_id = f"{uuid4().hex}-{episode_idx}"

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        self._episode_id = None
        self._decision_step = 0

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
        for output_name, camera in (("exterior_image", "over_shoulder_left_camera"), ("wrist_image", "wrist_cam")):
            image = Image.fromarray(self._to_numpy(raw_obs["image_obs"][camera], env_id))
            image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            observation[output_name] = np.asarray(image)
        return observation

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        if self._episode_id is None:
            raise RuntimeError("begin_episode must be called before inference")
        return {
            "episode_id": self._episode_id,
            "env_id": 0,
            "decision_step": self._decision_step,
            "instruction": instruction,
            "observation": extracted_obs,
        }

    def _query_server(self, request: dict) -> dict:
        self._connection.send(self._packer.pack(request))
        response = self._connection.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Harness execution error (no retry):\n{response}")
        return msgpack_numpy.unpackb(response)

    def _unpack_response(self, response: dict) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (self.open_loop_horizon, 8) or not np.isfinite(actions).all():
            raise ValueError("Invalid harness action chunk")
        if not np.allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1, atol=1e-4):
            raise ValueError("Harness action quaternions must be normalized")
        if not np.isin(actions[:, 7], [0, 1]).all():
            raise ValueError("Harness gripper actions must be binary")
        self._decision_step += 1
        return actions

    def _build_visualization(self, extracted_obs):
        return extracted_obs["exterior_image"]

    def close(self) -> None:
        self._resources.close()
