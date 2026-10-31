"""FLUX-action's DROID protocol with per-request evaluation timing."""

import json
import logging
import time
from pathlib import Path

from policies.cosmos3.client import Cosmos3Client

logger = logging.getLogger(__name__)


class FluxActionClient(Cosmos3Client):
    """Reuse Cosmos3's camera layout, absolute joint actions and 32-step chunks.

    Gripper observations are closed fractions (0=open, 1=closed). FLUX-action
    handles its training-time flip internally; the client only binarizes the
    returned gripper command, exactly as the Cosmos3 client does.
    """

    def __init__(
        self,
        remote_host: str = "127.0.0.1",
        remote_port: int = 8000,
        *,
        model_name: str = "flux_action",
        log_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(remote_host=remote_host, remote_port=remote_port)

    @staticmethod
    def _close_transport(client) -> None:
        # The bundled openpi-client has no public close method. Its underlying
        # synchronous WebSocket owns the connection; newer clients may expose close.
        transport = client if callable(getattr(client, "close", None)) else getattr(client, "_ws", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                logger.warning("Could not close FLUX-action connection", exc_info=True)

    def _connect(self):
        previous = getattr(self, "client", None)
        if previous is not None:
            self._close_transport(previous)
        client = super()._connect()
        try:
            if self.log_dir is not None:
                (self.log_dir / "server_metadata.json").write_text(
                    json.dumps(client.get_server_metadata(), indent=2) + "\n"
                )
        except Exception:
            self._close_transport(client)
            raise
        return client

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        extracted = super()._extract_observation(raw_obs, env_id=env_id)
        extracted["session_id"] = f"robolab-episode-{self._eval_episode_idx}-env-{env_id}"
        return extracted

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        request = super()._pack_request(extracted_obs, instruction)
        request["session_id"] = extracted_obs["session_id"]
        return request

    def _query_server(self, request: dict) -> dict:
        started = time.perf_counter()
        record = {
            "model": self.model_name,
            "task_env": self.log_dir.name if self.log_dir is not None else None,
            "session_id": request.get("session_id"),
            "timestamp": time.time(),
        }
        try:
            response = super()._query_server(request)
            record["ok"] = True
            record["server_timing"] = response.get("server_timing", {})
            return response
        except Exception as error:
            record["ok"] = False
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["rpc_ms"] = (time.perf_counter() - started) * 1000
            if self.log_dir is not None:
                # Open per record so errors, task changes and interrupted runs
                # never leave a buffered timing log behind.
                with (self.log_dir / "inference_requests.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            self._close_transport(client)
            self.client = None
        self.reset()
