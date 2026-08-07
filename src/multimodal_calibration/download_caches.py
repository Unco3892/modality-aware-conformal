"""Install the audited frozen-encoder cache archives.

The tracked JSON manifest fixes each archive's name, byte size, SHA-256 digest,
allowed members, member sizes, and member digests.  It deliberately contains
no public release URL while redistribution review is pending.  A caller must
therefore provide either an explicit local archive directory or an explicit
HTTP(S) base URL.  The historical ``--base-url`` plus ``--manifest
SHA256SUMS.txt`` invocation remains supported.

Only the Python standard library is used, so the installer can run before the
environment in ``requirements.txt`` is installed.

Examples::

    python src/multimodal_calibration/download_caches.py --archive-dir /path/to/cache_release

    python src/multimodal_calibration/download_caches.py --base-url https://github.com/Unco3892/modality-aware-conformal/releases/download/CACHE_TAG
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Mapping

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))
from multimodal_calibration.experiment_config import PAPER_DATASETS  # noqa: E402

DATASETS = PAPER_DATASETS
DEFAULT_MANIFEST = ROOT / "data" / "cache_manifest.json"
CHUNK_BYTES = 1 << 20
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class CacheInstallError(RuntimeError):
    """Raised when a cache source, manifest, or installed file is invalid."""


@dataclass(frozen=True)
class MemberSpec:
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ArchiveSpec:
    dataset: str
    filename: str
    bytes: int
    sha256: str
    target_subdir: str
    members: Mapping[str, MemberSpec]


@dataclass(frozen=True)
class CacheManifest:
    schema_version: int
    bundle_version: str
    base_url: str | None
    archives: Mapping[str, ArchiveSpec]


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheInstallError(f"{label} must be a positive integer")
    return value


def _sha256_value(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value.lower()) is None:
        raise CacheInstallError(f"{label} must be a 64-character SHA-256 digest")
    return value.lower()


def _canonical_relative_path(
    value: object,
    label: str,
    *,
    flat: bool = False,
) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CacheInstallError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise CacheInstallError(f"{label} must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise CacheInstallError(f"{label} is not a canonical relative path: {value!r}")
    if any(part in ("", ".", "..") or ":" in part for part in path.parts):
        raise CacheInstallError(f"{label} contains an unsafe component: {value!r}")
    if flat and len(path.parts) != 1:
        raise CacheInstallError(f"{label} must be a flat archive member name")
    return value


def _manifest_from_json(raw: object, source: Path) -> CacheManifest:
    if not isinstance(raw, dict):
        raise CacheInstallError(f"cache manifest must be a JSON object: {source}")
    if raw.get("schema_version") != 1:
        raise CacheInstallError(
            f"unsupported cache-manifest schema in {source}: "
            f"{raw.get('schema_version')!r}"
        )
    bundle_version = raw.get("bundle_version")
    if not isinstance(bundle_version, str) or not bundle_version.strip():
        raise CacheInstallError(f"bundle_version is missing from {source}")

    release = raw.get("release")
    if not isinstance(release, dict):
        raise CacheInstallError(f"release metadata is missing from {source}")
    base_url = release.get("base_url")
    if base_url is not None and (
        not isinstance(base_url, str) or not base_url.strip()
    ):
        raise CacheInstallError("release.base_url must be null or a non-empty URL")

    archive_rows = raw.get("archives")
    if not isinstance(archive_rows, list) or not archive_rows:
        raise CacheInstallError(f"archives must be a non-empty list in {source}")

    archives: dict[str, ArchiveSpec] = {}
    filenames: set[str] = set()
    for index, row in enumerate(archive_rows):
        label = f"archives[{index}]"
        if not isinstance(row, dict):
            raise CacheInstallError(f"{label} must be an object")
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or dataset not in DATASETS:
            raise CacheInstallError(f"{label}.dataset is not a paper dataset")
        if dataset in archives:
            raise CacheInstallError(f"duplicate archive dataset: {dataset}")

        filename = _canonical_relative_path(
            row.get("filename"), f"{label}.filename", flat=True
        )
        if filename != f"{dataset}_embeddings.zip":
            raise CacheInstallError(
                f"{label}.filename must be {dataset}_embeddings.zip"
            )
        if filename in filenames:
            raise CacheInstallError(f"duplicate archive filename: {filename}")
        filenames.add(filename)

        target_subdir = _canonical_relative_path(
            row.get("target_subdir"), f"{label}.target_subdir"
        )
        if target_subdir != f"{dataset}/embeddings":
            raise CacheInstallError(
                f"{label}.target_subdir must be {dataset}/embeddings"
            )

        member_rows = row.get("members")
        if not isinstance(member_rows, dict) or not member_rows:
            raise CacheInstallError(f"{label}.members must be a non-empty object")
        members: dict[str, MemberSpec] = {}
        for member_name, member_row in member_rows.items():
            name = _canonical_relative_path(
                member_name, f"{label}.members key", flat=True
            )
            if not name.endswith(".npy"):
                raise CacheInstallError(f"cache member must end in .npy: {name}")
            if not isinstance(member_row, dict):
                raise CacheInstallError(f"{label}.members[{name!r}] is invalid")
            members[name] = MemberSpec(
                bytes=_positive_int(
                    member_row.get("bytes"), f"{label}.members[{name!r}].bytes"
                ),
                sha256=_sha256_value(
                    member_row.get("sha256"),
                    f"{label}.members[{name!r}].sha256",
                ),
            )

        archives[dataset] = ArchiveSpec(
            dataset=dataset,
            filename=filename,
            bytes=_positive_int(row.get("bytes"), f"{label}.bytes"),
            sha256=_sha256_value(row.get("sha256"), f"{label}.sha256"),
            target_subdir=target_subdir,
            members=members,
        )

    return CacheManifest(
        schema_version=1,
        bundle_version=bundle_version,
        base_url=base_url,
        archives=archives,
    )


def _load_json_manifest(path: Path) -> CacheManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CacheInstallError(f"cache manifest not readable: {path}") from error
    except json.JSONDecodeError as error:
        raise CacheInstallError(f"cache manifest is not valid JSON: {path}") from error
    return _manifest_from_json(raw, path)


def _legacy_sha256s(text: str, source: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise CacheInstallError(
                f"malformed SHA256SUMS line {line_number} in {source}"
            )
        digest = _sha256_value(parts[0], f"{source}:{line_number}")
        filename = parts[1].lstrip("*")
        filename = _canonical_relative_path(
            filename, f"{source}:{line_number}", flat=True
        )
        if filename in values:
            raise CacheInstallError(f"duplicate SHA256SUMS filename: {filename}")
        values[filename] = digest
    if not values:
        raise CacheInstallError(f"SHA256SUMS manifest is empty: {source}")
    return values


def load_manifest(
    path: Path = DEFAULT_MANIFEST,
    *,
    fallback_json: Path = DEFAULT_MANIFEST,
) -> CacheManifest:
    """Load the tracked JSON schema or a legacy two-column SHA256SUMS file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CacheInstallError(f"cache manifest not readable: {path}") from error
    if text.lstrip().startswith("{"):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise CacheInstallError(
                f"cache manifest is not valid JSON: {path}"
            ) from error
        return _manifest_from_json(raw, path)

    base = _load_json_manifest(fallback_json)
    legacy = _legacy_sha256s(text, path)
    expected_names = {spec.filename for spec in base.archives.values()}
    if set(legacy) != expected_names:
        missing = sorted(expected_names - set(legacy))
        extra = sorted(set(legacy) - expected_names)
        raise CacheInstallError(
            f"legacy SHA256SUMS does not match the tracked archive set; "
            f"missing={missing}, extra={extra}"
        )
    archives = {
        dataset: replace(spec, sha256=legacy[spec.filename])
        for dataset, spec in base.archives.items()
    }
    return replace(base, archives=archives)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while chunk := source.read(CHUNK_BYTES):
        destination.write(chunk)
        digest.update(chunk)
        total += len(chunk)
    return total, digest.hexdigest()


def _validate_archive_file(path: Path, spec: ArchiveSpec) -> None:
    if not path.is_file():
        raise CacheInstallError(f"cache archive not found: {path}")
    size = path.stat().st_size
    if size != spec.bytes:
        raise CacheInstallError(
            f"archive size mismatch for {spec.filename}: "
            f"expected {spec.bytes}, got {size}"
        )
    digest = sha256(path)
    if digest != spec.sha256:
        raise CacheInstallError(
            f"archive checksum mismatch for {spec.filename}: "
            f"expected {spec.sha256}, got {digest}"
        )


def _target_dir(data_dir: Path, spec: ArchiveSpec) -> Path:
    data_root = data_dir.resolve()
    target = data_dir.joinpath(*PurePosixPath(spec.target_subdir).parts)
    resolved_target = target.resolve(strict=False)
    try:
        resolved_target.relative_to(data_root)
    except ValueError as error:
        raise CacheInstallError(
            f"cache target escapes the data directory: {target}"
        ) from error
    return target


def installed_problems(data_dir: Path, spec: ArchiveSpec) -> list[str]:
    """Return missing or mismatched installed members for one dataset."""
    target = _target_dir(data_dir, spec)
    problems: list[str] = []
    for name, member in spec.members.items():
        path = target / name
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        size = path.stat().st_size
        if size != member.bytes:
            problems.append(
                f"size mismatch {path}: expected {member.bytes}, got {size}"
            )
            continue
        digest = sha256(path)
        if digest != member.sha256:
            problems.append(
                f"checksum mismatch {path}: expected {member.sha256}, got {digest}"
            )
    return problems


def _check_zip_inventory(
    archive: zipfile.ZipFile,
    spec: ArchiveSpec,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise CacheInstallError(
            f"{spec.filename} contains duplicate members: {duplicates}"
        )

    expected = set(spec.members)
    actual = set(names)
    if actual != expected:
        raise CacheInstallError(
            f"{spec.filename} member set differs from the manifest; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    by_name: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = _canonical_relative_path(
            info.filename, f"{spec.filename} member", flat=True
        )
        if info.is_dir():
            raise CacheInstallError(f"{spec.filename} contains a directory: {name}")
        if info.flag_bits & 0x1:
            raise CacheInstallError(
                f"{spec.filename} contains an encrypted member: {name}"
            )
        unix_mode = info.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise CacheInstallError(f"{spec.filename} contains a symlink: {name}")
        expected_member = spec.members[name]
        if info.file_size != expected_member.bytes:
            raise CacheInstallError(
                f"member size mismatch for {spec.filename}/{name}: "
                f"expected {expected_member.bytes}, got {info.file_size}"
            )
        by_name[name] = info
    return by_name


def _stage_archive(
    archive_path: Path,
    spec: ArchiveSpec,
    data_dir: Path,
) -> Path:
    _validate_archive_file(archive_path, spec)
    dataset_dir = data_dir / spec.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".embeddings-install-", dir=str(dataset_dir))
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            inventory = _check_zip_inventory(archive, spec)
            for name in sorted(spec.members):
                output = staging / name
                with archive.open(inventory[name], "r") as source, output.open(
                    "xb"
                ) as destination:
                    size, digest = _copy_and_hash(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                expected = spec.members[name]
                if size != expected.bytes or digest != expected.sha256:
                    raise CacheInstallError(
                        f"member checksum mismatch for {spec.filename}/{name}: "
                        f"expected {expected.sha256}, got {digest}"
                    )
    except (OSError, zipfile.BadZipFile) as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise CacheInstallError(f"invalid cache archive {archive_path}: {error}") from error
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def install_archive(
    archive_path: Path,
    spec: ArchiveSpec,
    data_dir: Path,
    *,
    force: bool = False,
) -> tuple[int, int]:
    """Validate, stage, and atomically replace individual cache members."""
    problems = installed_problems(data_dir, spec)
    existing_mismatches = [
        problem for problem in problems if not problem.startswith("missing ")
    ]
    if existing_mismatches and not force:
        raise CacheInstallError(
            "refusing to overwrite non-matching cache files without --force:\n  "
            + "\n  ".join(existing_mismatches)
        )

    staging = _stage_archive(archive_path, spec, data_dir)
    target = _target_dir(data_dir, spec)
    target.mkdir(parents=True, exist_ok=True)
    installed = 0
    unchanged = 0
    try:
        for name, member in spec.members.items():
            source = staging / name
            destination = target / name
            if destination.is_file():
                if (
                    destination.stat().st_size == member.bytes
                    and sha256(destination) == member.sha256
                ):
                    source.unlink()
                    unchanged += 1
                    continue
            os.replace(source, destination)
            installed += 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return installed, unchanged


def _download_url(base_url: str, filename: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise CacheInstallError(
            "--base-url must be an explicit HTTP(S) directory URL"
        )
    return f"{base_url.rstrip('/')}/{urllib.parse.quote(filename)}"


def fetch(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
    timeout: float = 60.0,
    retries: int = 3,
) -> None:
    """Download to ``.part`` and atomically publish a complete local archive."""
    if retries < 1:
        raise ValueError("retries must be at least one")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "modality-aware-conformal-cache-installer/1"}
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        partial.unlink(missing_ok=True)
        try:
            print(f"  download {url} (attempt {attempt}/{retries})")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                if (
                    expected_bytes is not None
                    and content_length is not None
                    and int(content_length) != expected_bytes
                ):
                    raise CacheInstallError(
                        f"server size mismatch for {destination.name}: "
                        f"expected {expected_bytes}, got {content_length}"
                    )
                with partial.open("xb") as handle:
                    total = 0
                    while chunk := response.read(CHUNK_BYTES):
                        handle.write(chunk)
                        total += len(chunk)
                        if expected_bytes is not None and total > expected_bytes:
                            raise CacheInstallError(
                                f"download exceeds expected size for "
                                f"{destination.name}"
                            )
                    handle.flush()
                    os.fsync(handle.fileno())
            if expected_bytes is not None and total != expected_bytes:
                raise CacheInstallError(
                    f"download size mismatch for {destination.name}: "
                    f"expected {expected_bytes}, got {total}"
                )
            os.replace(partial, destination)
            return
        except (
            CacheInstallError,
            OSError,
            ValueError,
            urllib.error.URLError,
        ) as error:
            partial.unlink(missing_ok=True)
            last_error = error
            if attempt < retries:
                time.sleep(min(attempt, 2))
    raise CacheInstallError(f"download failed for {url}: {last_error}") from last_error


def _source_archive(
    spec: ArchiveSpec,
    *,
    archive_dir: Path | None,
    base_url: str | None,
    download_dir: Path,
) -> tuple[Path, bool]:
    if archive_dir is not None:
        source = archive_dir / spec.filename
        if not source.is_file():
            raise CacheInstallError(f"cache archive not found: {source}")
        return source, False
    if base_url is None:
        raise CacheInstallError(
            "no cache source is configured; pass --archive-dir or --base-url"
        )

    destination = download_dir / spec.filename
    if destination.exists():
        try:
            _validate_archive_file(destination, spec)
            print(f"  reuse verified download {destination}")
            return destination, True
        except CacheInstallError:
            destination.unlink()
    fetch(
        _download_url(base_url, spec.filename),
        destination,
        expected_bytes=spec.bytes,
    )
    return destination, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install or verify exact frozen-encoder caches using a tracked "
            "versioned manifest. No public download URL is configured."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=DATASETS,
        help="datasets to install (default: every archive in the manifest)",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--archive-dir",
        type=Path,
        help="local directory containing the manifest-named cache ZIP files",
    )
    sources.add_argument(
        "--base-url",
        help="explicit HTTP(S) directory URL containing the cache ZIP files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "tracked JSON manifest or legacy SHA256SUMS.txt "
            f"(default: {DEFAULT_MANIFEST.name})"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify installed members without acquiring or changing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing cache members whose hashes do not match",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="retain URL-downloaded ZIP files under data/_cache_downloads",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        datasets = args.datasets or list(manifest.archives)
        missing_specs = sorted(set(datasets) - set(manifest.archives))
        if missing_specs:
            raise CacheInstallError(
                f"datasets absent from cache manifest: {missing_specs}"
            )

        if args.verify_only:
            failed = False
            for dataset in datasets:
                problems = installed_problems(
                    args.data_dir, manifest.archives[dataset]
                )
                if problems:
                    failed = True
                    print(f"{dataset}: INVALID", file=sys.stderr)
                    for problem in problems:
                        print(f"  {problem}", file=sys.stderr)
                else:
                    print(f"{dataset}: verified")
            return 1 if failed else 0

        base_url = args.base_url or manifest.base_url
        download_dir = args.data_dir / "_cache_downloads"
        for dataset in datasets:
            spec = manifest.archives[dataset]
            problems = installed_problems(args.data_dir, spec)
            if not problems:
                print(f"{dataset}: already installed and verified")
                continue
            source, downloaded = _source_archive(
                spec,
                archive_dir=args.archive_dir,
                base_url=base_url,
                download_dir=download_dir,
            )
            try:
                installed, unchanged = install_archive(
                    source, spec, args.data_dir, force=args.force
                )
            finally:
                if downloaded and not args.keep_downloads:
                    source.unlink(missing_ok=True)
            print(
                f"{dataset}: installed {installed}, retained {unchanged} "
                f"verified arrays -> {_target_dir(args.data_dir, spec)}"
            )

        if download_dir.exists() and not any(download_dir.iterdir()):
            download_dir.rmdir()
        print("all requested caches installed and verified")
        return 0
    except CacheInstallError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
