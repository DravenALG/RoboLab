"""CPU client tests; use pytest --noconftest to skip Isaac Sim startup."""

import json

import numpy as np
import pytest
import torch

from policies.cosmos3 import client as cosmos_client
from policies.flux_action.client import FluxActionClient


class FakeTransport:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.requests = []
        self.closed = False
        self.error = None

    def get_server_metadata(self):
        return {"serving_setup": {"compile_dit": True}}

    def infer(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        actions = np.repeat(request["observation/joint_position"][None], 32, axis=0)
        gripper = np.resize(np.array([0.2, 0.5, 0.8], np.float32), (32, 1))
        return {"action": np.concatenate((actions, gripper), axis=1), "server_timing": {"infer_ms": 12.0}}

    def close(self):
        self.closed = True


@pytest.fixture
def transports(monkeypatch):
    connections = []

    def connect(host, port):
        transport = FakeTransport(host, port)
        connections.append(transport)
        return transport

    monkeypatch.setattr(cosmos_client.websocket_client_policy, "WebsocketClientPolicy", connect)
    return connections


@pytest.fixture
def observation():
    return {
        "image_obs": {
            "wrist_cam": torch.full((2, 36, 64, 3), 30, dtype=torch.uint8),
            "over_shoulder_left_camera": torch.full((2, 36, 64, 3), 90, dtype=torch.uint8),
            "over_shoulder_right_camera": torch.full((2, 36, 64, 3), 150, dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.arange(14, dtype=torch.float32).reshape(2, 7),
            "gripper_pos": torch.tensor([[0.25], [0.75]]),
        },
    }


def test_request_camera_layout_state_and_instruction(transports, observation):
    client = FluxActionClient(remote_host="example", remote_port=8123)
    client.begin_episode(4)
    result = client.infer(observation, "put the banana in the bowl", env_id=1)
    remote = transports[0]
    request = remote.requests[0]
    image = request["observation/image"]
    assert (remote.host, remote.port) == ("example", 8123)
    assert image.shape == (540, 640, 3) and image.dtype == np.uint8
    np.testing.assert_array_equal(image[:360], 30)
    np.testing.assert_array_equal(image[360:, :320], 90)
    np.testing.assert_array_equal(image[360:, 320:], 150)
    np.testing.assert_array_equal(request["observation/joint_position"], np.arange(7, 14))
    np.testing.assert_array_equal(request["observation/gripper_position"], [0.75])
    assert request["prompt"] == "put the banana in the bowl"
    assert request["session_id"] == "robolab-episode-4-env-1"
    np.testing.assert_array_equal(result["action"], [7, 8, 9, 10, 11, 12, 13, 0])
    assert result["viz"].shape == (360, 1920, 3)
    client.close()


def test_chunk_horizon_gripper_and_per_env_reset(transports, observation):
    client = FluxActionClient()
    actions = [client.infer(observation, "move", env_id=0)["action"] for _ in range(32)]
    assert len(transports[0].requests) == 1
    assert [action[-1] for action in actions[:3]] == [0, 0, 1]
    np.testing.assert_array_equal(actions[-1][:7], np.arange(7))
    client.infer(observation, "move", env_id=1)
    assert len(transports[0].requests) == 2
    client.infer(observation, "move", env_id=0)
    assert len(transports[0].requests) == 3
    client.reset(env_id=0)
    client.infer(observation, "move", env_id=1)
    assert len(transports[0].requests) == 3
    client.infer(observation, "move", env_id=0)
    assert len(transports[0].requests) == 4
    client.reset()
    assert not client._chunks and not client._counters
    client.close()


def test_metadata_and_success_timing_are_saved(transports, observation, tmp_path):
    client = FluxActionClient(model_name="flux_base", log_dir=tmp_path)
    client.infer(observation, "move")
    client.infer(observation, "move")
    assert json.loads((tmp_path / "server_metadata.json").read_text()) == transports[0].get_server_metadata()
    records = [json.loads(line) for line in (tmp_path / "inference_requests.jsonl").read_text().splitlines()]
    assert len(records) == 1  # cached actions must not create fake RPC measurements
    assert records[0]["ok"] and records[0]["model"] == "flux_base"
    assert records[0]["server_timing"] == {"infer_ms": 12.0}
    assert records[0]["rpc_ms"] >= 0 and records[0]["task_env"] == tmp_path.name
    client.close()
    client.close()
    assert transports[0].closed


def test_server_error_is_logged_and_propagated(transports, observation, tmp_path):
    client = FluxActionClient(log_dir=tmp_path)
    transports[0].error = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        client.infer(observation, "move")
    record = json.loads((tmp_path / "inference_requests.jsonl").read_text())
    assert not record["ok"] and record["error"] == "RuntimeError: inference failed"
    assert record["rpc_ms"] >= 0 and not client._chunks
    client.close()


def test_reconnection_closes_old_socket_and_invalidates_all_chunks(transports, observation):
    client = FluxActionClient()
    client.infer(observation, "move", env_id=0)
    transports[0].error = OSError("connection lost")
    client.infer(observation, "move", env_id=1)
    assert len(transports) == 2 and transports[0].closed
    assert set(client._chunks) == {1}
    client.infer(observation, "move", env_id=0)
    assert len(transports[1].requests) == 2
    client.close()


def test_close_supports_bundled_openpi_private_socket():
    class Socket:
        closed = False

        def close(self):
            self.closed = True

    class BundledClient:
        def __init__(self):
            self._ws = Socket()

    remote = BundledClient()
    FluxActionClient._close_transport(remote)
    assert remote._ws.closed
