"""Download and validate the official AI4I dataset from UCI."""

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

from .config import DATASET_PATH, UCI_DATASET_URL
from .preprocessing import validate_dataset


def download_dataset(destination: Path = DATASET_PATH, timeout: int = 45) -> Path:
    """Download the UCI archive and save its validated CSV to ``destination``.

    The CSV is written only after the response, ZIP archive, expected filename,
    and dataset schema have all been checked.
    """
    destination = Path(destination)
    request = Request(
        UCI_DATASET_URL,
        headers={"User-Agent": "PharmaGuard-AI/1.0 (UCI dataset download)"},
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            archive_bytes = BytesIO(response.read())
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "Could not download the AI4I dataset from UCI. Check your internet "
            f"connection and try again. Details: {error}"
        ) from error

    try:
        with ZipFile(archive_bytes) as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if Path(name).name.lower() == "ai4i2020.csv"
            ]
            if len(csv_names) != 1:
                raise RuntimeError(
                    "The UCI archive did not contain exactly one ai4i2020.csv file."
                )

            with archive.open(csv_names[0]) as csv_file:
                dataset = pd.read_csv(csv_file)
    except BadZipFile as error:
        raise RuntimeError("UCI returned a file that is not a valid ZIP archive.") from error

    validate_dataset(dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(destination, index=False)
    return destination


def ensure_dataset(destination: Path = DATASET_PATH) -> Path:
    """Use a validated local CSV or download it from UCI when it is absent."""
    destination = Path(destination)
    if destination.exists():
        try:
            dataset = pd.read_csv(destination)
            validate_dataset(dataset)
        except (OSError, pd.errors.ParserError, ValueError) as error:
            raise RuntimeError(
                f"The local dataset at {destination} could not be validated: {error}"
            ) from error
        return destination
    return download_dataset(destination)
