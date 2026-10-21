"""Offline progress inspection and recovery for sequential harness evaluations.

Only benchmark result records commit an episode. Model logs, videos and HDF5
files can exist before that commit, so they must never be used to skip work.
This module deliberately uses only the standard library, including when loaded
by the batch launcher from the server's Python environment.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4


def _directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"Expected a directory, not a file or symlink: {path}")


@dataclass
class Assignment:
    gpu: int
    seconds: float
    tasks: list[str]


def read_assignments(run_dir: Path, gpus: list[int]) -> list[Assignment]:
    """Read the original mapping; extra, unused trailing GPUs are harmless."""
    _directory(run_dir)
    path = run_dir / "assignments.tsv"
    if path.is_symlink():
        raise ValueError(f"Assignments must not be a symlink: {path}")
    assignments, seen_tasks, ports = [], set(), []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["gpu", "port", "estimated_seconds", "tasks"]:
            raise ValueError(f"Invalid assignments header: {path}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Incomplete assignment in {path}")
            gpu, port = int(row["gpu"]), int(row["port"])
            seconds, tasks = float(row["estimated_seconds"]), row["tasks"].split()
            if gpu < 0 or not 1 <= port <= 65535 or not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"Invalid GPU, port or estimate in {path}")
            if not tasks or any(not task.isidentifier() for task in tasks):
                raise ValueError(f"Invalid tasks in {path}")
            if len(set(tasks)) != len(tasks) or seen_tasks.intersection(tasks):
                raise ValueError(f"Duplicate task assignment in {path}")
            seen_tasks.update(tasks)
            ports.append(port)
            assignments.append(Assignment(gpu, seconds, tasks))
    saved_gpus = [assignment.gpu for assignment in assignments]
    if not assignments or len(set(saved_gpus)) != len(saved_gpus):
        raise ValueError(f"Empty or duplicate GPU assignments in {path}")
    if ports != list(range(ports[0], ports[0] + len(ports))):
        raise ValueError(f"Invalid original port order in {path}")
    if gpus[:len(saved_gpus)] != saved_gpus:
        raise ValueError(f"--gpus must preserve the original worker GPUs and order: {saved_gpus}")
    return assignments


@dataclass
class Progress:
    output: Path
    records: dict[tuple[str, int], dict]
    raw: bytes = b""
    tail_offset: int | None = None
    legacy: bool = False

    def pending(self, tasks: list[str], runs: int) -> list[tuple[str, int]]:
        if type(runs) is not int or runs < 1:
            raise ValueError("runs must be a positive integer")
        if any(episode >= runs for _, episode in self.records):
            raise ValueError(f"Saved episode exceeds --runs; keep the original run count: {self.output}")
        if any(not task.isidentifier() for task in tasks):
            raise ValueError("Invalid task name")
        return [(task, episode) for task in tasks for episode in range(runs)
                if (task, episode) not in self.records]

    def artifacts(self, tasks: list[str], runs: int) -> list[Path]:
        """Find only files belonging to missing episodes, never completed ones."""
        artifacts = []
        for task, episode in self.pending(tasks, runs):
            scene = self.output / task
            _directory(scene)
            if not scene.exists():
                continue
            names = {f"run_{episode}.hdf5", f"run_{episode}.hdf5_failed.hdf5",
                     f"log_{episode}.json", f"log_{episode}_env0.json"}
            video = re.compile(rf"_{episode}(?:_env0)?(?:_viewport)?\.mp4$")
            for path in scene.iterdir():
                if path.name in names or video.search(path.name):
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(f"Expected an episode artifact file: {path}")
                    artifacts.append(path)
        return sorted(artifacts)


def inspect_output(output: Path) -> Progress:
    """Read without writing. Only an unterminated, malformed last line is recoverable."""
    output = Path(output)
    _directory(output)
    progress = Progress(output, {})
    path = output / "episode_results.jsonl"
    legacy = output / "episode_results.json"
    if path.is_symlink() or legacy.is_symlink():
        raise ValueError(f"Result files must not be symlinks: {output}")
    rows = []
    if path.exists():
        progress.raw = path.read_bytes()
        lines = progress.raw.splitlines(keepends=True)
        offset = 0
        for number, line in enumerate(lines, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    if number == len(lines) and not line.endswith(b"\n"):
                        progress.tail_offset = offset
                    else:
                        raise ValueError(f"Malformed result at {path}:{number}") from error
            offset += len(line)
    elif legacy.exists():
        progress.legacy = True
        progress.raw = legacy.read_bytes()
        rows = json.loads(progress.raw)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list of episode results in {legacy}")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid episode result in {output}")
        task, episode = row.get("env_name"), row.get("episode")
        if (not isinstance(task, str) or not task.isidentifier()
                or type(episode) is not int or episode < 0
                or type(row.get("success")) is not bool):
            raise ValueError(f"Invalid episode identity or success field in {output}")
        if (row.get("policy", "harness") != "harness"
                or row.get("task_name", task) != task
                or type(row.get("run", episode)) is not int or row.get("run", episode) != episode
                or type(row.get("env_id", 0)) is not int or row.get("env_id", 0) != 0):
            raise ValueError(f"Conflicting harness episode identity in {output}: {task}/{episode}")
        key = (task, episode)
        if key in progress.records:
            raise ValueError(f"Duplicate episode result in {output}: {task}/{episode}")
        progress.records[key] = row
    return progress


def inspect_worker(run_dir: Path, assignment: Assignment, runs: int) -> Progress:
    worker = run_dir / f"gpu_{assignment.gpu}"
    _directory(worker)
    progress = inspect_output(worker / "robolab")
    if any(task not in assignment.tasks for task, _ in progress.records):
        raise ValueError(f"Results do not match the saved task assignment: {progress.output}")
    progress.artifacts(assignment.tasks, runs)  # Validate recovery before starting any workers.
    return progress


def prepare_output(output: Path, tasks: list[str], runs: int) -> Path | None:
    """Archive interrupted artifacts and repair the ledger before appending to it.

    Call only with exclusive access. Archives live beside the output directory,
    outside the benchmark result tree. Repeated preparation is safe.
    """
    progress = inspect_output(output)
    if not progress.pending(tasks, runs):
        return None
    artifacts = progress.artifacts(tasks, runs)
    raw = progress.raw
    if progress.legacy:
        repaired = b"".join((json.dumps(row) + "\n").encode() for row in progress.records.values())
    else:
        repaired = raw if progress.tail_offset is None else raw[:progress.tail_offset]
        if repaired and not repaired.endswith(b"\n"):
            repaired += b"\n"
    repair = progress.legacy or repaired != raw
    if not artifacts and not repair:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_root = output.parent / f"{output.name}.resume"
    _directory(archive_root)
    archive = archive_root / f"{stamp}_{uuid4().hex[:8]}"
    archive.mkdir(parents=True)
    if repair:
        name = "episode_results.json" if progress.legacy else "episode_results.jsonl"
        (archive / f"{name}.bak").write_bytes(raw)
        # Atomic replacement keeps all committed rows intact if recovery itself stops.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output, prefix=".resume-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(repaired)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(output / "episode_results.jsonl")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    for path in artifacts:
        destination = archive / path.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)
    return archive
