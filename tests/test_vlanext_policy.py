from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from policies.vlanext_policy.client import (
    DROID_TCP_IN_BASE_POS,
    DROID_TCP_IN_BASE_QUAT_WXYZ,
    VLANeXtDroidClient,
    droid_actions_to_robolab,
    resize_image,
    robolab_pose_to_droid,
)


def test_droid_eef_action_is_converted_to_base_link_absolute_ik() -> None:
    offset_xyzw = DROID_TCP_IN_BASE_QUAT_WXYZ[[1, 2, 3, 0]]
    euler = Rotation.from_quat(offset_xyzw).as_euler("xyz")
    droid_action = np.array([[0.4, -0.2, 0.3, *euler, 0.7]], dtype=np.float32)

    action = droid_actions_to_robolab(droid_action)[0]

    np.testing.assert_allclose(
        action[:3], droid_action[0, :3] - DROID_TCP_IN_BASE_POS, atol=1e-6
    )
    np.testing.assert_allclose(action[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert action[7] == 1.0


def test_droid_and_robolab_pose_transforms_are_inverses() -> None:
    base_position = np.array([0.325, 0.117, 0.500])
    base_rotation = Rotation.from_euler("xyz", [-0.925, 1.350, -0.908])
    base_xyzw = base_rotation.as_quat()
    base_wxyz = base_xyzw[[3, 0, 1, 2]]

    droid_pose = robolab_pose_to_droid(base_position, base_wxyz)
    droid_action = np.concatenate([droid_pose, [0.0]])[None]
    recovered = droid_actions_to_robolab(droid_action)[0]

    np.testing.assert_allclose(recovered[:3], base_position, atol=1e-6)
    recovered_rotation = _rotation_from_wxyz(recovered[3:7])
    rotation_error = (recovered_rotation.inv() * base_rotation).as_rotvec()
    np.testing.assert_allclose(rotation_error, np.zeros(3), atol=1e-6)


def _rotation_from_wxyz(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]])


def test_resize_image_uses_server_shape_without_padding() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :10, 0] = 255
    resized = resize_image(image, [8, 12])
    assert resized.shape == (8, 12, 3)
    assert resized[:, 0, 0].mean() > resized[:, -1, 0].mean()


def test_integer_resize_uses_target_width_and_preserves_aspect_ratio() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    resized = resize_image(image, 10)

    assert resized.shape == (5, 10, 3)


@pytest.mark.parametrize("image_size", [True, 0, -1, [0, 8], [8, 0], [1, 2, 3], "8"])
def test_resize_image_rejects_invalid_metadata(image_size) -> None:
    with pytest.raises(ValueError, match="image_size"):
        resize_image(np.zeros((10, 20, 3), dtype=np.uint8), image_size)


class _FakeServerClient:
    def __init__(self, *_args, **_kwargs) -> None:
        self.requests = []
        self.metadata = {
            "action_representation": "droid_absolute_eef_xyz_euler_xyz_gripper",
            "model_action_representation": "droid_absolute_eef_xyz_euler_xyz_gripper",
            "action_normalization": "quantile_01_99",
            "action_horizon": 2,
            "action_dim": 7,
            "history_len": 2,
            "image_size": 8,
            "input_modality": "video",
            "view_mode": "multi",
        }

    def get_server_metadata(self) -> dict:
        return self.metadata

    def infer(self, request: dict) -> dict:
        self.requests.append(request)
        actions = np.zeros((2, 7), dtype=np.float32)
        actions[:, :3] = [0.4, 0.0, 0.3]
        actions[:, 6] = [0.2, 0.8]
        return {"actions": actions, "normalized_actions": actions.copy()}


def _observation() -> dict:
    return {
        "image_obs": {
            "over_shoulder_left_camera": np.zeros((1, 12, 16, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((1, 12, 16, 3), dtype=np.uint8),
        },
        "proprio_obs": {
            "ee_pos": np.array([[0.4, 0.0, 0.3]], dtype=np.float32),
            "ee_quat": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "gripper_pos": np.array([[0.0]], dtype=np.float32),
        },
    }


def test_client_uses_metadata_and_tracks_per_step_histories() -> None:
    with patch(
        "policies.vlanext_policy.client.websocket_client_policy.WebsocketClientPolicy",
        _FakeServerClient,
    ):
        client = VLANeXtDroidClient(action_mode="cartesian", open_loop_horizon=2)

    first = client.infer(_observation(), "pick up the object")
    second = client.infer(_observation(), "pick up the object")
    third = client.infer(_observation(), "pick up the object")

    assert first["action"].shape == (8,)
    assert first["action"][-1] == 0.0
    assert second["action"][-1] == 1.0
    assert third["action"][-1] == 0.0
    assert len(client._remote_policy.requests) == 2
    first_request, second_request = client._remote_policy.requests
    assert first_request["observation/exterior_image_1_left"].shape == (1, 6, 8, 3)
    assert second_request["observation/exterior_image_1_left"].shape == (2, 6, 8, 3)
    assert second_request["observation/state_history"].shape == (2, 7)
    assert "observation/action_history" not in first_request
    assert "observation/action_history" not in second_request


@pytest.mark.parametrize("crop_ratio", [1.0, 0.95])
def test_client_crop_preserves_shape_and_removes_borders_in_both_views(crop_ratio) -> None:
    with patch(
        "policies.vlanext_policy.client.websocket_client_policy.WebsocketClientPolicy",
        _FakeServerClient,
    ):
        client = VLANeXtDroidClient(action_mode="cartesian", center_crop_ratio=crop_ratio)
    client.image_size = None
    observation = _observation()
    for image_key in ("over_shoulder_left_camera", "wrist_cam"):
        image = np.zeros((1, 80, 160, 3), dtype=np.uint8)
        image[:, :2] = 255
        image[:, -2:] = 255
        image[:, :, :4] = 255
        image[:, :, -4:] = 255
        observation["image_obs"][image_key] = image

    client.infer(observation, "pick up the object")
    request = client._remote_policy.requests[0]
    for key in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
        images = request[key]
        assert images.shape == (1, 80, 160, 3)
        assert images.dtype == np.uint8
        if crop_ratio == 1.0:
            np.testing.assert_array_equal(images, image)
        else:
            assert not images.any()


class _FakeJointServerClient(_FakeServerClient):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.metadata.update(
            action_mode="joint",
            action_representation="droid_absolute_joint_position_gripper",
            model_action_representation="droid_delta_joint_position_gripper",
            action_dim=8,
        )

    def infer(self, request: dict) -> dict:
        self.requests.append(request)
        actions = np.tile(np.arange(8, dtype=np.float32), (2, 1))
        actions[:, -1] = [0.2, 0.8]
        return {"actions": actions, "normalized_actions": actions.copy()}


def test_joint_client_packs_joint_state_and_executes_joint_targets() -> None:
    with patch(
        "policies.vlanext_policy.client.websocket_client_policy.WebsocketClientPolicy",
        _FakeJointServerClient,
    ):
        client = VLANeXtDroidClient(open_loop_horizon=2)

    assert client.action_mode == "joint"
    observation = _observation()
    observation["proprio_obs"]["arm_joint_pos"] = np.arange(7, dtype=np.float32)[None]
    first = client.infer(observation, "pick up the object")
    second = client.infer(observation, "pick up the object")

    np.testing.assert_array_equal(first["action"], [0, 1, 2, 3, 4, 5, 6, 0])
    np.testing.assert_array_equal(second["action"], [0, 1, 2, 3, 4, 5, 6, 1])
    request = client._remote_policy.requests[0]
    np.testing.assert_array_equal(
        request["observation/state_history"][0],
        [0, 1, 2, 3, 4, 5, 6, 0],
    )


def test_client_rejects_action_mode_mismatch() -> None:
    with patch(
        "policies.vlanext_policy.client.websocket_client_policy.WebsocketClientPolicy",
        _FakeJointServerClient,
    ), pytest.raises(ValueError, match="action representation"):
        VLANeXtDroidClient(action_mode="cartesian")


def test_batched_client_keeps_environment_chunks_and_refreshes_only_when_needed() -> None:
    class BatchServer(_FakeJointServerClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metadata["supports_batch_inference"] = True

        def infer(self, request):
            self.requests.append(request)
            responses = []
            for item in request["requests"]:
                state = item["observation/state_history"][-1]
                actions = np.tile(state, (2, 1))
                actions[:, -1] = 0.4
                responses.append({"actions": actions})
            return {"responses": responses}

    with patch("policies.vlanext_policy.client.websocket_client_policy.WebsocketClientPolicy", BatchServer):
        client = VLANeXtDroidClient(inference_batch_size=2, gripper_threshold=0.3)
    observation = _observation()
    for group in observation.values():
        for key, value in group.items():
            group[key] = np.repeat(value, 3, axis=0)
    observation["proprio_obs"]["arm_joint_pos"] = np.tile(np.arange(3)[:, None], (1, 7)).astype(np.float32)
    first = client.infer_batch(observation, "move", env_ids=[2, 0])
    client.infer_batch(observation, "move", env_ids=[0])
    third = client.infer_batch(observation, "move", env_ids=[2, 0])
    assert [len(r["requests"]) for r in client._remote_policy.requests] == [2, 1]
    for result in (first, third):
        for env_id in (2, 0):
            np.testing.assert_array_equal(result[env_id]["action"][:7], env_id)
            assert result[env_id]["action"][-1] == 1.0
