"""PharmaVision split discovery, image checks, and PyTorch transforms."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import (
    BATCH_SIZE,
    CLASS_TO_INDEX,
    DATASET_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}
EXPECTED_CLASS_COUNTS = {"train": 190, "validation": 30, "test": 100}


def _split_directories(root: Path) -> dict:
    """Find a directory containing the official train/val/test folders."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"PharmaVision directory not found: {root}")

    candidates = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        found = {}
        for child in candidate.iterdir():
            if child.is_dir() and child.name.casefold() in SPLIT_ALIASES:
                split_name = SPLIT_ALIASES[child.name.casefold()]
                if split_name not in found:
                    found[split_name] = child
        if set(found) == {"train", "validation", "test"}:
            return found

    raise FileNotFoundError(
        f"Could not find original Train, Val/Validation, and test folders under {root}."
    )


def _class_directories(split_path: Path) -> dict:
    directories = {}
    for child in split_path.iterdir():
        if child.is_dir() and child.name.casefold() in CLASS_TO_INDEX:
            directories[child.name.casefold()] = child
    if set(directories) != set(CLASS_TO_INDEX):
        raise ValueError(
            f"Expected class folders named 'normal' and 'defective' in {split_path}; "
            f"found {sorted(directories)}."
        )
    return directories


def validate_image_dataset(data_path: Path = DATASET_DIR) -> dict:
    """Validate original split names, class labels, counts, and image readability."""
    root = find_dataset_root(data_path)
    splits = _split_directories(root)
    summary = {}

    for split_name, split_path in splits.items():
        classes = _class_directories(split_path)
        counts = {}
        for class_name, class_path in classes.items():
            image_paths = sorted(
                path
                for path in class_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in image_paths:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        if image.width < 1 or image.height < 1:
                            raise ValueError("image has an empty dimension")
                        image.convert("RGB")
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    raise ValueError(f"Invalid image file {image_path}: {error}") from error
            counts[class_name] = len(image_paths)
            if counts[class_name] != EXPECTED_CLASS_COUNTS[split_name]:
                raise ValueError(
                    f"Unexpected {class_name} count in {split_name}: "
                    f"expected {EXPECTED_CLASS_COUNTS[split_name]}, found {counts[class_name]}. "
                    "This check protects the dataset's original fixed splits."
                )
        summary[split_name] = counts

    return {"root": str(root), "splits": summary}


def find_dataset_root(data_path: Path = DATASET_DIR) -> Path:
    """Resolve the nested archive folder that contains all original splits."""
    root = Path(data_path)
    splits = _split_directories(root)
    # Split discovery returns paths directly inside the actual dataset root.
    return next(iter(splits.values())).parent


def build_image_transform(training: bool = False):
    """Create 224x224 ImageNet-normalized transforms; augment training only."""
    operations = []
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.07, contrast=0.07),
            ]
        )
    operations.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class TabletImageDataset(Dataset):
    """Load images from the official split folders without mixing their rows."""

    def __init__(self, split_path: Path, transform=None):
        self.transform = transform
        classes = _class_directories(Path(split_path))
        self.samples = []
        for class_name, class_path in classes.items():
            for image_path in sorted(class_path.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, CLASS_TO_INDEX[class_name]))
        if not self.samples:
            raise ValueError(f"No supported tablet images found in {split_path}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def prepare_image_dataset(
    data_path: Path = DATASET_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> dict:
    """Return loaders for the official splits, augmenting only training images."""
    root = find_dataset_root(data_path)
    validate_image_dataset(root)
    split_paths = _split_directories(root)
    loaders = {}
    for split_name, split_path in split_paths.items():
        dataset = TabletImageDataset(
            split_path,
            transform=build_image_transform(training=(split_name == "train")),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders
