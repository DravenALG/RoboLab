"""Sample whole-device GPU usage independently of the simulation step loop."""

import logging
import math
import subprocess
import threading

import torch


logger = logging.getLogger(__name__)


class GpuUsageMonitor:
    """One monitor per run; peaks are sampled, not allocator high-water marks.

    UUID lookup respects CUDA_VISIBLE_DEVICES. Whole-device memory includes
    PhysX, rendering, and other processes, unlike PyTorch allocator counters.
    Parallel environments sharing a run also share these device-level peaks.
    """

    def __init__(self, device):
        self.device = device
        self.data = {
            "scope": "whole_device",
            "sample_interval_s": 1.0,
            "gpu_uuid": None,
            "peak_memory_used_mib": None,
            "peak_utilization_percent": None,
            "samples": 0,
            "sampling_errors": 0,
            "complete": False,
        }
        self._stop = threading.Event()
        self._thread = None

    def _record_error(self, error):
        if not self.data["sampling_errors"]:
            logger.warning("GPU usage statistics unavailable or partial: %s", error)
        self.data["sampling_errors"] += 1
        self.data["error"] = str(error)

    def _sample(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", self.data["gpu_uuid"],
                 "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=2,
            )
            memory, utilization = map(float, result.stdout.strip().split(","))
            if not (math.isfinite(memory) and memory >= 0
                    and math.isfinite(utilization) and 0 <= utilization <= 100):
                raise ValueError("Invalid nvidia-smi GPU usage sample")
            for key, value in (("peak_memory_used_mib", memory),
                               ("peak_utilization_percent", utilization)):
                previous = self.data[key]
                self.data[key] = value if previous is None else max(previous, value)
            self.data["samples"] += 1
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self._record_error(error)

    def _poll(self):
        while not self._stop.wait(self.data["sample_interval_s"]):
            self._sample()

    def __enter__(self):
        try:
            device = torch.device(self.device)
            if device.type != "cuda":
                self.data["error"] = "Not a CUDA device"
                return self
            # Recent PyTorch exposes the actual CUDA device UUID, independent
            # of the physical indices reported by nvidia-smi.
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            self.data["gpu_uuid"] = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
            self._sample()
            self._thread = threading.Thread(target=self._poll, name="gpu-usage", daemon=True)
            self._thread.start()
        except Exception as error:
            # Monitoring must never prevent an evaluation from running.
            self._record_error(error)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join()
            self._sample()
        self.data["complete"] = self.data["samples"] > 0 and self.data["sampling_errors"] == 0
