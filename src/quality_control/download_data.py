"""Download and extract PharmaVision from its official Zenodo record."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import DATASET_ARCHIVE, DATASET_DIR, RAW_DATA_DIR, ZENODO_API_URL


def _request(url: str, timeout: int = 90):
    request = Request(url, headers={"User-Agent": "PharmaGuard-AI/1.0"})
    return urlopen(request, timeout=timeout)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path = DATASET_ARCHIVE) -> Path:
    """Fetch the archive named by Zenodo metadata and verify its checksum."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with _request(ZENODO_API_URL) as response:
            metadata = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        raise RuntimeError(
            "Could not read the official PharmaVision Zenodo record. Check your "
            f"internet connection and retry. Details: {error}"
        ) from error

    files = metadata.get("files", [])
    archive_info = next(
        (item for item in files if item.get("key", "").lower().endswith(".rar")),
        None,
    )
    if archive_info is None:
        raise RuntimeError("The Zenodo record did not list a RAR dataset archive.")

    expected_size = int(archive_info.get("size", 0))
    checksum = str(archive_info.get("checksum", ""))
    checksum_algorithm, separator, expected_digest = checksum.partition(":")
    if not separator or checksum_algorithm.lower() not in {"md5", "sha256"}:
        raise RuntimeError(f"Zenodo provided an unsupported archive checksum: {checksum!r}.")

    if destination.exists():
        digest = hashlib.new(checksum_algorithm.lower())
        with destination.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if destination.stat().st_size == expected_size and digest.hexdigest() == expected_digest:
            return destination
        raise RuntimeError(
            f"An existing archive at {destination} does not match Zenodo's size/checksum. "
            "Move it aside and retry; it was not overwritten."
        )

    download_url = archive_info.get("links", {}).get("self")
    if not download_url:
        raise RuntimeError("Zenodo metadata did not provide a download link for the archive.")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="pharmavision-", suffix=".part", dir=destination.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with _request(download_url, timeout=180) as response:
                shutil.copyfileobj(response, temporary_file)

        if temporary_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Downloaded archive size mismatch: expected {expected_size} bytes, "
                f"received {temporary_path.stat().st_size} bytes."
            )
        digest = hashlib.new(checksum_algorithm.lower())
        with temporary_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise RuntimeError("Downloaded PharmaVision archive failed Zenodo checksum validation.")

        os.replace(temporary_path, destination)
        temporary_path = None
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "Could not download PharmaVision from Zenodo. Check your internet "
            f"connection and retry. Details: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


def extract_archive(archive_path: Path = DATASET_ARCHIVE, destination: Path = DATASET_DIR) -> Path:
    """Extract the RAR using bsdtar and reject unsafe archive paths."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    extractor = shutil.which("bsdtar") or shutil.which("7z") or shutil.which("7za")
    if extractor is None:
        raise RuntimeError(
            "PharmaVision is distributed as a RAR archive, but no supported extractor "
            "was found. Install bsdtar (included with many Anaconda installations) or "
            "7-Zip, then rerun the dataset setup."
        )

    listing = subprocess.run(
        [extractor, "-tf" if Path(extractor).name.lower().startswith("bsdtar") else "l", str(archive_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError(
            "The RAR archive could not be listed by the installed extractor. "
            f"Details: {listing.stderr.strip()}"
        )

    for entry in listing.stdout.splitlines():
        candidate = PurePosixPath(entry.replace("\\", "/"))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(f"Unsafe path in PharmaVision archive: {entry!r}.")

    destination.mkdir(parents=True, exist_ok=True)
    if Path(extractor).name.lower().startswith("bsdtar"):
        command = [extractor, "-xf", str(archive_path), "-C", str(destination)]
    else:
        command = [extractor, "x", str(archive_path), f"-o{destination}", "-y"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Could not extract PharmaVision. Ensure the extractor supports RAR5. "
            f"Details: {result.stderr.strip()}"
        )
    return destination


def ensure_dataset() -> Path:
    """Return the dataset root, downloading/extracting it if not present."""
    from .preprocessing import find_dataset_root, validate_image_dataset

    try:
        root = find_dataset_root(DATASET_DIR)
        validate_image_dataset(root)
        return root
    except FileNotFoundError:
        pass

    archive = download_archive()
    extract_archive(archive)
    root = find_dataset_root(DATASET_DIR)
    validate_image_dataset(root)
    return root
