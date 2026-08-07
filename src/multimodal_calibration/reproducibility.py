"""Small, dependency-light provenance helpers for experiment drivers.

Recorded runs must be attributable to one source tree and one campaign. This
module deliberately records both the Git identity and a hash of the working
source files: a dirty checkout is therefore visible and cannot be mistaken for
the recorded commit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Mapping


PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "torch",
    "joblib",
    "threadpoolctl",
    "pyarrow",
    "matplotlib",
    "pillow",
    "openml",
)
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def validate_campaign_tag(tag: str | None, *, required: bool = False) -> str | None:
    """Validate a filesystem-safe, human-readable campaign identifier."""
    if tag is None or not tag.strip():
        if required:
            raise ValueError(
                "recorded runs require --campaign-tag (or --tag) so "
                "artifacts cannot be mixed with an unrelated campaign"
            )
        return None
    tag = tag.strip()
    if not TAG_RE.fullmatch(tag):
        raise ValueError(
            "campaign tag must start with an alphanumeric character and contain "
            "only letters, digits, '.', '_' or '-' (maximum 80 characters)"
        )
    return tag


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_tree_paths(root: Path) -> tuple[Path, ...]:
    """Return the executable-source inventory bound to recorded results."""
    roots = (
        root / "src",
        root / "tests",
    )
    paths = {
        p
        for base in roots
        if base.exists()
        for p in base.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".sh", ".toml", ".yaml", ".yml", ".json"}
    }
    # Dependency pins and documented commands are part of the executable
    # reproduction definition, not ancillary prose.
    paths.update(
        path
        for path in (
            root / "requirements.txt",
            root / "README.md",
            root / "pytest.ini",
        )
        if path.is_file()
    )
    return tuple(sorted(paths))


def source_tree_sha256(root: Path) -> str:
    """Hash exact executable-source paths and bytes in the worktree."""
    digest = hashlib.sha256()
    for path in source_tree_paths(root):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, object]:
    """Return a content identity for an exact binary artifact."""
    path = Path(path)
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        initial_stat = os.fstat(stream.fileno())
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        final_stat = os.fstat(stream.fileno())
    if (
        initial_stat.st_size != byte_count
        or final_stat.st_size != byte_count
    ):
        raise RuntimeError(f"{path} changed while its identity was computed")
    return {
        "filename": path.name,
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def validate_artifact_identity(
    path: Path,
    expected: Mapping[str, object] | None,
) -> dict[str, object]:
    """Verify that ``path`` is exactly the artifact recorded in a payload."""
    if not isinstance(expected, Mapping):
        raise RuntimeError(f"{path}: missing artifact identity")
    actual = artifact_identity(path)
    comparable = {
        "filename": expected.get("filename"),
        "size_bytes": expected.get("size_bytes"),
        "sha256": expected.get("sha256"),
    }
    if actual != comparable:
        raise RuntimeError(
            f"{path}: artifact identity mismatch "
            f"(expected {comparable}, found {actual})"
        )
    return actual


def atomic_write_binary(
    path: Path,
    writer: Callable[[BinaryIO], object],
) -> dict[str, object]:
    """Publish a binary file atomically after the writer fully flushes it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with tmp.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Record the bytes produced by *this* writer before publication. If a
        # duplicate worker later replaces the same destination, the completion
        # JSON will retain this generation's hash and downstream verification
        # will fail closed instead of authenticating the other worker's bytes.
        identity = artifact_identity(tmp)
        identity["filename"] = path.name
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return identity


def atomic_write_json(
    path: Path,
    payload: object,
    *,
    default: Callable[[object], object] | None = None,
) -> dict[str, object]:
    """Publish a JSON completion record atomically."""

    def write(stream: BinaryIO) -> None:
        encoded = (
            json.dumps(payload, indent=2, default=default, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        stream.write(encoded)

    return atomic_write_binary(path, write)


def source_identity(root: Path) -> dict:
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "tracked_worktree_dirty": bool(status) if status is not None else None,
        "source_tree_sha256": source_tree_sha256(root),
    }


def _package_versions(names: Iterable[str] = PACKAGE_NAMES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_identity() -> dict:
    """Return the cross-worker runtime fields that must remain identical."""
    return {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "machine": platform.machine(),
        "packages": _package_versions(),
    }


def thread_identity() -> dict[str, str | None]:
    """Return producer thread policy separately from immutable runtime core."""
    return {name: os.environ.get(name) for name in THREAD_ENV_VARS}


def normalize_run_config(
    run_config: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize a manifest configuration to deterministic JSON values."""
    if run_config is None:
        return {}
    if not isinstance(run_config, Mapping):
        raise TypeError("run_config must be a mapping")

    def normalize(value):
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [normalize(item) for item in value]
            return sorted(
                normalized,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        if isinstance(value, Path):
            return value.as_posix()
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            if not (float("-inf") < value < float("inf")):
                raise ValueError("run_config cannot contain NaN or infinity")
            return value
        raise TypeError(
            f"run_config contains unsupported value {value!r} "
            f"of type {type(value).__name__}"
        )

    normalized = normalize(run_config)
    # A canonical serialization is a final guard against non-JSON values.
    json.dumps(normalized, sort_keys=True, allow_nan=False)
    return normalized


def runtime_provenance(
    root: Path,
    *,
    seed: int | None,
    campaign_tag: str | None,
    requested_device: str | None = None,
) -> dict:
    """Return JSON-serializable run, source, thread, and device provenance."""
    device: dict[str, object] = {"requested": requested_device}
    try:
        import torch

        device.update(
            {
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.is_available()
                    else None
                ),
                "gpu_names": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
            }
        )
    except (ImportError, RuntimeError):
        device["torch_cuda_available"] = None

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_tag": campaign_tag,
        "seed": seed,
        "command": list(sys.argv),
        "source": source_identity(root),
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "platform_system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "packages": _package_versions(),
        },
        "threads": thread_identity(),
        "device": device,
    }


def attach_run_provenance(
    payload: dict,
    root: Path,
    *,
    seed: int | None,
    campaign_tag: str | None,
    requested_device: str | None = None,
) -> dict:
    """Attach provenance in-place and return ``payload`` for convenient use."""
    payload["provenance"] = runtime_provenance(
        root,
        seed=seed,
        campaign_tag=campaign_tag,
        requested_device=requested_device,
    )
    return payload


def validate_campaign_payloads(
    payloads: Iterable[tuple[str, dict]],
    root: Path,
    *,
    campaign_tag: str | None,
    requested_device: str | None = None,
    run_config: Mapping[str, object] | None = None,
    producer_threads: Mapping[str, object] | None = None,
) -> str:
    """Require payloads from one current-source/config/runtime campaign."""
    tag = validate_campaign_tag(campaign_tag, required=campaign_tag is not None)
    current_source = source_identity(root)
    current_environment = environment_identity()
    expected_run_config = normalize_run_config(run_config)
    expected_threads = (
        normalize_run_config(producer_threads)
        if producer_threads is not None
        else None
    )
    devices: set[str] = set()
    thread_configs: set[str] = set()
    errors: list[str] = []
    seen = 0
    for label, payload in payloads:
        seen += 1
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{label}: missing run provenance")
            continue
        if provenance.get("campaign_tag") != tag:
            errors.append(f"{label}: campaign tag mismatch")
        payload_seed = payload.get("config", {}).get("seed")
        if payload_seed is not None:
            try:
                seed_matches = (
                    provenance.get("seed") == int(payload_seed)
                )
            except (TypeError, ValueError):
                seed_matches = False
            if not seed_matches:
                errors.append(f"{label}: provenance seed mismatch")
        if provenance.get("source") != current_source:
            errors.append(f"{label}: source identity mismatch")
        runtime = provenance.get("runtime", {})
        payload_environment = {
            "python": runtime.get("python"),
            "platform_system": (
                runtime.get("platform_system")
                or (
                    str(runtime.get("platform", "")).split("-", 1)[0]
                    if runtime.get("platform")
                    else None
                )
            ),
            "machine": runtime.get("machine"),
            "packages": runtime.get("packages"),
        }
        if payload_environment != current_environment:
            errors.append(f"{label}: runtime package/environment mismatch")
        payload_run_config = payload.get("config", {}).get("run_config")
        if normalize_run_config(payload_run_config) != expected_run_config:
            errors.append(f"{label}: run configuration mismatch")
        payload_threads = normalize_run_config(provenance.get("threads") or {})
        thread_configs.add(
            json.dumps(payload_threads, sort_keys=True, separators=(",", ":"))
        )
        if expected_threads is not None and payload_threads != expected_threads:
            errors.append(f"{label}: producer thread policy mismatch")
        device = provenance.get("device", {}).get("requested")
        if not isinstance(device, str) or not device:
            errors.append(f"{label}: requested device missing from provenance")
        else:
            devices.add(device)
    if seen == 0:
        errors.append("no payloads supplied")
    if len(devices) > 1:
        errors.append("mixed requested devices: " + ", ".join(sorted(devices)))
    if len(thread_configs) > 1:
        errors.append("mixed producer thread policies")
    if errors:
        raise RuntimeError(
            "campaign payload validation failed: " + "; ".join(errors)
        )
    if len(devices) != 1:
        raise RuntimeError("campaign payload validation found no common device")
    common_device = next(iter(devices))
    if requested_device is not None and common_device != requested_device:
        raise RuntimeError(
            "campaign payload validation failed: requested device "
            f"{common_device!r} does not match expected {requested_device!r}"
        )
    return common_device


def write_campaign_manifest(
    out_dir: Path,
    root: Path,
    *,
    campaign_tag: str,
    datasets: Iterable[str],
    seeds: Iterable[int],
    methods: Iterable[str] = (),
    requested_device: str | None = None,
    run_config: Mapping[str, object] | None = None,
    producer_threads: Mapping[str, object] | None = None,
) -> Path:
    """Create or verify the immutable identity file for one result directory.

    An existing directory is rejected when its tag, source hash, or expected
    grid differs. This prevents an aggregator from silently combining legacy
    and current reproduction runs.
    """
    tag = validate_campaign_tag(campaign_tag, required=True)
    dataset_values = tuple(sorted(set(datasets)))
    seed_values = tuple(sorted(set(int(seed) for seed in seeds)))
    method_values = tuple(sorted(set(methods)))
    normalized_config = normalize_run_config(run_config)
    normalized_threads = normalize_run_config(producer_threads)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "campaign_manifest.json"
    identity = source_identity(root)
    expected = {
        "schema_version": 2,
        "campaign_tag": tag,
        "source": identity,
        "environment": environment_identity(),
        "datasets": list(dataset_values),
        "seeds": list(seed_values),
        "methods": list(method_values),
        "requested_device": requested_device,
        "run_config": normalized_config,
        "producer_threads": normalized_threads,
    }
    expected["created_utc"] = datetime.now(timezone.utc).isoformat()
    # The UUID also separates worker threads in one process; the PID alone is
    # sufficient only for process-based array jobs.
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    comparable_keys = (
        "schema_version",
        "campaign_tag",
        "source",
        "environment",
        "datasets",
        "seeds",
        "methods",
        "requested_device",
        "run_config",
        "producer_threads",
    )
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(expected, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link gives atomic create-if-absent semantics. Unlike
            # os.replace, two incompatible writers cannot both report success
            # or overwrite the winner on Windows.
            os.link(tmp, path)
        except FileExistsError:
            pass
    finally:
        tmp.unlink(missing_ok=True)

    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read completed campaign manifest {path}") from error
    mismatches = [
        key for key in comparable_keys if actual.get(key) != expected.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            f"{path} belongs to a different campaign; mismatched fields: "
            + ", ".join(mismatches)
        )
    return path
