"""Persisted batch generations and GPU-independent episode recovery.

Only result ledgers commit work. A new GPU layout writes a new generation;
historical ledgers and artifacts stay at their original paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile

from policies.harness_policy.resume import (
    Assignment, Progress, _directory, inspect_output, prepare_output, read_assignments,
)

MANIFEST = "batch_manifest.json"


def validate_gpus(gpus: list[int]) -> None:
    if (not isinstance(gpus, list) or not gpus
            or any(type(g) is not int or g < 0 for g in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError("GPU indices must be nonnegative and unique")


def _json(path: Path):
    if path.is_symlink():
        raise ValueError(f"Metadata must not be a symlink: {path}")
    return json.loads(path.read_text())


def read_episode_plan(path: Path, runs: int) -> frozenset[tuple[str, int]]:
    data = _json(path)
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or type(data.get("runs")) is not int or data["runs"] != runs
            or not isinstance(data.get("episodes"), list)):
        raise ValueError(f"Invalid episode plan or changed --runs: {path}")
    selected = set()
    for item in data["episodes"]:
        if (not isinstance(item, list) or len(item) != 2
                or not isinstance(item[0], str) or not item[0].isidentifier()
                or type(item[1]) is not int or not 0 <= item[1] < runs):
            raise ValueError(f"Invalid episode selection in {path}")
        key = tuple(item)
        if key in selected:
            raise ValueError(f"Duplicate episode selection in {path}: {key}")
        selected.add(key)
    if not selected:
        raise ValueError(f"Empty episode plan: {path}")
    return frozenset(selected)


@dataclass
class Worker:
    directory: Path
    assignment: Assignment
    selection: frozenset[tuple[str, int]] | None = None

    @property
    def gpu(self) -> int:
        return self.assignment.gpu

    def validate(self, progress: Progress, runs: int) -> None:
        progress.pending(self.assignment.tasks, runs)
        if any(task not in self.assignment.tasks for task, _ in progress.records):
            raise ValueError(f"Results do not match the saved task assignment: {progress.output}")
        if self.selection is not None and not progress.records.keys() <= self.selection:
            raise ValueError(f"Result is outside the saved episode plan: {progress.output}")

    def progress(self, runs: int, *, artifacts=False) -> Progress:
        _directory(self.directory)
        progress = inspect_output(self.directory / "robolab")
        self.validate(progress, runs)
        if artifacts:
            progress.artifacts(self.assignment.tasks, runs)
        return progress

    def pending(self, progress: Progress, runs: int) -> list[tuple[str, int]]:
        return [key for key in progress.pending(self.assignment.tasks, runs)
                if self.selection is None or key in self.selection]


@dataclass
class Generation:
    relative: str
    gpus: list[int]
    workers: list[Worker]


@dataclass
class Batch:
    root: Path
    runs: int
    costs: dict[str, float]
    generations: list[Generation]
    persisted: bool = False
    task_set: str = "full"

    @property
    def current(self) -> Generation:
        return self.generations[-1]

    @property
    def workers(self) -> list[Worker]:
        return [worker for generation in self.generations for worker in generation.workers]

    def committed(self, *, artifacts=False) -> dict[tuple[str, int], dict]:
        found = {}
        for worker in self.workers:
            progress = worker.progress(self.runs, artifacts=artifacts)
            for key, row in progress.records.items():
                if key in found:
                    raise ValueError(f"Duplicate episode result across batches: {key[0]}/{key[1]}")
                found[key] = row
        return found

    def document(self) -> dict:
        return dict(schema_version=1, runs=self.runs, task_set=self.task_set, task_costs=self.costs,
                    generations=[dict(path=g.relative, gpus=g.gpus) for g in self.generations])


def _generation(root: Path, relative: str, gpus: list[int], runs: int) -> Generation:
    if relative != "." and not re.fullmatch(r"resume_batches/[0-9]{6,}", relative):
        raise ValueError(f"Invalid batch path: {relative!r}")
    directory = root
    for part in Path(relative).parts:
        directory /= part
        _directory(directory)
    assignments = read_assignments(directory, gpus)
    workers = []
    for assignment in assignments:
        path = directory / f"gpu_{assignment.gpu}"
        _directory(path)
        selection = None
        if relative != ".":
            selection = read_episode_plan(path / "episode_plan.json", runs)
            if {task for task, _ in selection} != set(assignment.tasks):
                raise ValueError(f"Episode plan does not match assignments: {path}")
        workers.append(Worker(path, assignment, selection))
    return Generation(relative, gpus, workers)


def read_batch(root: Path, runs: int | None = None) -> Batch:
    root = Path(root)
    _directory(root)
    if runs is not None and (type(runs) is not int or runs < 1):
        raise ValueError("runs must be a positive integer")
    path = root / MANIFEST
    if not path.exists() and not path.is_symlink():
        # Legacy assignments have no target run count or per-task estimates.
        runs = runs or 1
        assignments = read_assignments(root, None)
        costs = {task: (a.seconds / (len(a.tasks) * runs) if a.seconds > 0 else 1.0)
                 for a in assignments for task in a.tasks}
        gpus = [a.gpu for a in assignments]
        return Batch(root, runs, costs, [_generation(root, ".", gpus, runs)])
    data = _json(path)
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or type(data.get("runs")) is not int or data["runs"] < 1):
        raise ValueError(f"Invalid batch manifest: {path}")
    if runs is not None and runs != data["runs"]:
        raise ValueError(f"Keep the original --runs total ({data['runs']})")
    runs = data["runs"]
    task_set = data.get("task_set", "full")
    if task_set not in ("full", "sub"):
        raise ValueError(f"Invalid task set in {path}")
    costs = data.get("task_costs")
    if (not isinstance(costs, dict) or not costs
            or any(not isinstance(t, str) or not t.isidentifier()
                   or type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for t, v in costs.items())):
        raise ValueError(f"Invalid task costs in {path}")
    entries = data.get("generations")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Missing batch generations in {path}")
    generations, seen = [], set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid generation in {path}")
        relative, gpus = entry.get("path"), entry.get("gpus")
        validate_gpus(gpus)
        if (not isinstance(relative, str) or relative in seen
                or (index == 0 and relative != ".") or (index > 0 and relative == ".")):
            raise ValueError(f"Invalid or duplicate batch path in {path}")
        generation = _generation(root, relative, gpus, runs)
        tasks = {t for w in generation.workers for t in w.assignment.tasks}
        if not tasks <= costs.keys() or (index == 0 and tasks != costs.keys()):
            raise ValueError(f"Generation tasks do not match the batch catalog: {relative}")
        generations.append(generation)
        seen.add(relative)
    return Batch(root, runs, costs, generations, persisted=True, task_set=task_set)


def worker_directories(root: Path) -> list[Path]:
    """Also support historical timing audits that have no assignment file."""
    root = Path(root)
    if (root / MANIFEST).exists() or (root / MANIFEST).is_symlink():
        return [w.directory for w in read_batch(root).workers]
    return sorted(root.glob("gpu_*"))


@dataclass
class ResumePlan:
    batch: Batch
    generation: Generation
    completed: dict[tuple[str, int], dict]
    changed: bool

    @property
    def remaining(self) -> int:
        return len(self.batch.costs) * self.batch.runs - len(self.completed)

    def pending(self, worker: Worker) -> list[tuple[str, int]]:
        return [(task, run) for task in worker.assignment.tasks
                for run in range(self.batch.runs)
                if (task, run) not in self.completed
                and (worker.selection is None or (task, run) in worker.selection)]


def plan_resume(root: Path, gpus: list[int], runs: int, *, task_set: str | None = None) -> ResumePlan:
    validate_gpus(gpus)
    batch = read_batch(root, runs)
    if task_set is not None and task_set != batch.task_set:
        raise ValueError(f"Keep the original --set {batch.task_set}; received {task_set}")
    completed = batch.committed(artifacts=True)
    remaining = {t: [r for r in range(runs) if (t, r) not in completed] for t in batch.costs}
    remaining = {t: ids for t, ids in remaining.items() if ids}
    if gpus == batch.current.gpus or not remaining:
        # Every missing episode must still be owned by the current generation.
        assigned = {key for w in batch.current.workers
                    for t in w.assignment.tasks for key in ((t, r) for r in range(runs))
                    if w.selection is None or key in w.selection}
        missing = {(t, r) for t, ids in remaining.items() for r in ids}
        if not missing <= assigned:
            raise ValueError("Current batch assignments omit pending episodes")
        return ResumePlan(batch, batch.current, completed, False)
    parent = batch.root / "resume_batches"
    _directory(parent)
    # Never reuse an unpublished directory left by a crashed preparation.
    numbers = [int(p.name) for p in parent.iterdir() if re.fullmatch(r"[0-9]{6,}", p.name)] if parent.exists() else []
    relative = f"resume_batches/{max(numbers, default=0) + 1:06d}"
    loads, tasks = [0.0] * len(gpus), [[] for _ in gpus]
    for task in sorted(remaining, key=lambda t: (-batch.costs[t] * len(remaining[t]), t)):
        index = min(range(len(gpus)), key=lambda i: loads[i])
        tasks[index].append(task)
        loads[index] += batch.costs[task] * len(remaining[task])
    workers = [Worker(batch.root / relative / f"gpu_{gpu}", Assignment(gpu, loads[i], tasks[i]),
                      frozenset((t, r) for t in tasks[i] for r in remaining[t]))
               for i, gpu in enumerate(gpus) if tasks[i]]
    return ResumePlan(batch, Generation(relative, list(gpus), workers), completed, True)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"Metadata must not be a symlink: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".batch-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, data: dict) -> None:
    _atomic_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())


def initialize_batch(root: Path, gpus: list[int], runs: int, costs: dict[str, float],
                     *, task_set: str = "full") -> None:
    """Called with the run-directory lock held, after writing initial assignments."""
    validate_gpus(gpus)
    if task_set not in ("full", "sub"):
        raise ValueError("Task set must be full or sub")
    batch = read_batch(root, runs)
    if batch.persisted:
        raise ValueError("Batch manifest already exists")
    if set(costs) != set(batch.costs):
        raise ValueError("Task costs do not match initial assignments")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in costs.values()):
        raise ValueError("Task costs must be finite and positive")
    batch.costs = costs
    batch.task_set = task_set
    batch.current.gpus = list(gpus)
    _atomic_json(batch.root / MANIFEST, batch.document())


def commit_resume(plan: ResumePlan, base_port: int) -> None:
    """Publish a fully prepared generation, under the caller's exclusive lock."""
    if not plan.remaining:
        return
    batch, generation = plan.batch, plan.generation
    if type(base_port) is not int or not 1 <= base_port <= 65536 - len(generation.gpus):
        raise ValueError("Invalid batch port range")
    if plan.changed:
        directory = batch.root / generation.relative
        _directory(directory.parent)
        directory.parent.mkdir(exist_ok=True)
        directory.mkdir()  # No reuse or overwrite of an abandoned generation.
        rows = ["gpu\tport\testimated_seconds\ttasks\n"]
        for worker in generation.workers:
            worker.directory.mkdir()
            _atomic_json(worker.directory / "episode_plan.json",
                         dict(schema_version=1, runs=batch.runs, episodes=sorted(worker.selection)))
            index = generation.gpus.index(worker.gpu)
            rows.append(f"{worker.gpu}\t{base_port + index}\t{worker.assignment.seconds:.3f}\t"
                        + " ".join(worker.assignment.tasks) + "\n")
        _atomic_bytes(directory / "assignments.tsv", "".join(rows).encode())
        _sync_directory(directory.parent)
        # No processes may start before this atomic publication succeeds.
        document = batch.document()
        document["generations"].append(dict(path=generation.relative, gpus=generation.gpus))
        _atomic_json(batch.root / MANIFEST, document)
    else:
        if not batch.persisted:
            _atomic_json(batch.root / MANIFEST, batch.document())
        for worker in generation.workers:
            if plan.pending(worker):
                archive = prepare_output(worker.directory / "robolab", worker.assignment.tasks, batch.runs)
                if archive is not None:
                    print(f"Archived interrupted output: {archive}")
