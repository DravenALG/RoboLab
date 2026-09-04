"""RoboLab client for a VLANeXt DROID absolute-EEF policy server."""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any
from urllib.parse import urlparse

import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image
from scipy.spatial.transform import Rotation

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)

# DROID Cartesian actions target a TCP that differs from RoboLab's controlled
# ``base_link``.  The transform is expressed in the base_link frame.
DROID_TCP_IN_BASE_POS = np.array([-0.018174022, 0.0, 0.0], dtype=np.float64)
DROID_TCP_IN_BASE_QUAT_WXYZ = np.array(
    [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=np.float64
)


def _rotation_from_wxyz(quaternion: np.ndarray) -> Rotation:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat(quaternion[..., [1, 2, 3, 0]])


def robolab_pose_to_droid(
    base_position: np.ndarray,
    base_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Convert one RoboLab base_link pose to DROID TCP xyz + Euler XYZ."""
    base_position = np.asarray(base_position, dtype=np.float64).reshape(3)
    base_rotation = _rotation_from_wxyz(base_quaternion_wxyz)
    tcp_in_base = _rotation_from_wxyz(DROID_TCP_IN_BASE_QUAT_WXYZ)

    tcp_position = base_position + base_rotation.apply(DROID_TCP_IN_BASE_POS)
    tcp_euler = (base_rotation * tcp_in_base).as_euler("xyz")
    return np.concatenate([tcp_position, tcp_euler]).astype(np.float32)


def droid_actions_to_robolab(actions: np.ndarray) -> np.ndarray:
    """Convert DROID ``xyz + Euler XYZ + gripper`` to RoboLab absolute IK."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected DROID actions with shape (T, 7), got {actions.shape}")

    tcp_rotation = Rotation.from_euler("xyz", actions[:, 3:6])
    tcp_in_base = _rotation_from_wxyz(DROID_TCP_IN_BASE_QUAT_WXYZ)
    base_rotation = tcp_rotation * tcp_in_base.inv()
    base_position = actions[:, :3] - base_rotation.apply(DROID_TCP_IN_BASE_POS)
    base_quat_xyzw = base_rotation.as_quat()
    base_quat_wxyz = base_quat_xyzw[:, [3, 0, 1, 2]]
    gripper = (actions[:, 6:7] > 0.5).astype(np.float64)
    return np.concatenate([base_position, base_quat_wxyz, gripper], axis=1).astype(np.float32)


def resize_image(image: np.ndarray, image_size: int | tuple[int, int] | list[int] | None) -> np.ndarray:
    """Resize an RGB image directly, matching DROID training preprocessing."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 image, got {image.shape}")
    if image_size is None:
        return image
    if isinstance(image_size, bool):
        raise ValueError(f"Invalid server image_size metadata: {image_size}")
    if isinstance(image_size, int):
        width = image_size
        height = max(1, round(image.shape[0] * width / image.shape[1]))
    elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        height, width = map(int, image_size)
    else:
        raise ValueError(f"Invalid server image_size metadata: {image_size}")
    if height <= 0 or width <= 0:
        raise ValueError(f"Server image_size metadata must be positive, got {image_size}")
    if image.shape[:2] == (height, width):
        return image
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


class VLANeXtDroidEEFClient(InferenceClient):
    """Stateful OpenPI-protocol client for VLANeXt's DROID action space."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int | None = None,
        remote_uri: str | None = None,
    ) -> None:
        super().__init__()
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._remote_uri = remote_uri
        self._display = remote_uri or f"{remote_host}:{remote_port}"

        print(f"[{self.__class__.__name__}] Waiting for VLANeXt at {self._display}...")
        self._remote_policy = self._connect()
        self.metadata = self._remote_policy.get_server_metadata()
        self._configure_from_metadata(open_loop_horizon)
        self._reset_histories()
        print(
            f"[{self.__class__.__name__}] Connected: horizon={self.action_horizon}, "
            f"history={self.history_len}, modality={self.input_modality}, views={self.view_mode}."
        )

    def _connect(self) -> websocket_client_policy.WebsocketClientPolicy:
        host = urlparse(self._remote_uri).hostname if self._remote_uri else self._remote_host
        if host in {"localhost", "127.0.0.1", "::1"}:
            for variable in ("NO_PROXY", "no_proxy"):
                entries = {entry for entry in os.environ.get(variable, "").split(",") if entry}
                entries.update({"localhost", "127.0.0.1", "::1"})
                os.environ[variable] = ",".join(sorted(entries))
        if self._remote_uri is not None:
            return websocket_client_policy.WebsocketClientPolicy(self._remote_uri)
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _configure_from_metadata(self, requested_horizon: int | None) -> None:
        expected = "droid_absolute_eef_xyz_euler_xyz_gripper"
        representation = self.metadata.get("action_representation")
        if representation != expected:
            raise ValueError(f"Server action representation is {representation!r}, expected {expected!r}")

        self.action_horizon = int(self.metadata["action_horizon"])
        self.history_len = int(self.metadata["history_len"])
        self.image_size = self.metadata.get("image_size")
        self.input_modality = self.metadata.get("input_modality", "image")
        self.view_mode = self.metadata.get("view_mode", "single")
        if int(self.metadata.get("action_dim", 0)) != 7:
            raise ValueError(f"Server action_dim must be 7, got {self.metadata.get('action_dim')}")

        if self.input_modality not in {"image", "video"}:
            raise ValueError(f"Unsupported input modality: {self.input_modality!r}")
        if self.view_mode not in {"single", "multi"}:
            raise ValueError(f"Unsupported view mode: {self.view_mode!r}")

        self.open_loop_horizon = int(requested_horizon or self.action_horizon)
        if not 1 <= self.open_loop_horizon <= self.action_horizon:
            raise ValueError(
                f"open_loop_horizon must be in [1, {self.action_horizon}], "
                f"got {self.open_loop_horizon}"
            )

    def _reset_histories(self, env_id: int | None = None) -> None:
        if not hasattr(self, "_state_history") or env_id is None:
            self._state_history: dict[int, deque[np.ndarray]] = {}
            self._image_history: dict[int, deque[np.ndarray]] = {}
            self._wrist_history: dict[int, deque[np.ndarray]] = {}
            return
        self._state_history.pop(env_id, None)
        self._image_history.pop(env_id, None)
        self._wrist_history.pop(env_id, None)

    def _history(self, histories: dict[int, deque[np.ndarray]], env_id: int) -> deque[np.ndarray]:
        if env_id not in histories:
            histories[env_id] = deque(maxlen=max(1, self.history_len))
        return histories[env_id]

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self.reset()

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        self._reset_histories(env_id)

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image = self._to_numpy(raw_obs["image_obs"]["over_shoulder_left_camera"], env_id)
        wrist = self._to_numpy(raw_obs["image_obs"]["wrist_cam"], env_id)
        proprio = raw_obs["proprio_obs"]
        base_position = self._to_numpy(proprio["ee_pos"], env_id).reshape(3)
        base_quaternion = self._to_numpy(proprio["ee_quat"], env_id).reshape(4)
        gripper = float(self._to_numpy(proprio["gripper_pos"], env_id).reshape(-1)[0])
        droid_pose = robolab_pose_to_droid(base_position, base_quaternion)

        image = resize_image(image, self.image_size)
        wrist = resize_image(wrist, self.image_size)
        state = np.concatenate([droid_pose, [np.clip(gripper, 0.0, 1.0)]]).astype(
            np.float32
        )
        self._history(self._state_history, env_id).append(state)
        self._history(self._image_history, env_id).append(image)
        self._history(self._wrist_history, env_id).append(wrist)
        return {"env_id": env_id, "image": image, "wrist": wrist, "state": state}

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        env_id = extracted_obs["env_id"]
        exterior: np.ndarray = extracted_obs["image"]
        wrist: np.ndarray = extracted_obs["wrist"]
        if self.input_modality == "video":
            exterior = np.stack(self._image_history[env_id])
            wrist = np.stack(self._wrist_history[env_id])

        request = {
            "observation/exterior_image_1_left": exterior,
            "observation/state_history": np.stack(self._state_history[env_id]),
            "prompt": instruction,
        }
        if self.view_mode == "multi":
            request["observation/wrist_image_left"] = wrist
        return request

    def _query_server(self, request: dict) -> dict:
        return self._infer_with_retry(request)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                return self._remote_policy.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as error:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "VLANeXt connection lost (%s); reconnecting (%d/%d)",
                    error,
                    attempt + 1,
                    max_retries,
                )
                self._remote_policy = self._connect()
                self._chunks.clear()
                self._counters.clear()

        raise RuntimeError("VLANeXt inference retry loop exited unexpectedly")

    def _unpack_response(self, response: dict) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (self.action_horizon, 7):
            raise ValueError(
                f"Server returned actions {actions.shape}, expected ({self.action_horizon}, 7)"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Server returned NaN or infinity")
        return actions

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return droid_actions_to_robolab(chunk)

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        if self.view_mode == "multi":
            return np.concatenate([extracted_obs["image"], extracted_obs["wrist"]], axis=1)
        return extracted_obs["image"]
