"""Small CNN and pretrained MobileNetV2 training/loading helpers."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .config import CLASS_NAMES, EARLY_STOPPING_PATIENCE, IMAGE_SIZE, MAX_EPOCHS, RANDOM_STATE


class CustomTabletCNN(nn.Module):
    """A compact convolutional baseline for 224x224 RGB tablet images."""

    def __init__(self, number_of_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(192, number_of_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(self.pool(features))


def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Construct one of the two supported binary classifiers."""
    if model_name == "custom_cnn":
        return CustomTabletCNN(number_of_classes=len(CLASS_NAMES))
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        network = models.mobilenet_v2(weights=weights)
        for parameter in network.features.parameters():
            parameter.requires_grad = False
        network.classifier[1] = nn.Linear(
            network.classifier[1].in_features,
            len(CLASS_NAMES),
        )
        return network
    raise ValueError(f"Unknown tablet model {model_name!r}.")


def load_model(model_name: str, checkpoint_path: Path, device=None) -> nn.Module:
    """Load a saved model without downloading pretrained weights again."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, pretrained=False)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def train_model(
    model_name: str,
    loaders: dict,
    checkpoint_path: Path,
    device=None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> dict:
    """Train one model and save the checkpoint with the lowest validation loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    use_pretrained = model_name == "mobilenet_v2"
    try:
        model = build_model(model_name, pretrained=use_pretrained).to(device)
    except Exception as error:
        if use_pretrained:
            raise RuntimeError(
                "Could not load ImageNet pretrained MobileNetV2 weights. Check the "
                "internet connection and retry; training a random-weight substitute "
                "would not meet the transfer-learning requirement."
            ) from error
        raise

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=1e-3)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        if use_pretrained:
            # Keep frozen MobileNet BatchNorm layers in inference mode.
            model.features.eval()

        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for images, labels in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(labels)
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_count += len(labels)

        validation_loss_sum = 0.0
        validation_correct = 0
        validation_count = 0
        model.eval()
        with torch.no_grad():
            for images, labels in loaders["validation"]:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                validation_loss_sum += float(loss.item()) * len(labels)
                validation_correct += int((logits.argmax(dim=1) == labels).sum().item())
                validation_count += len(labels)

        metrics = {
            "train_loss": train_loss_sum / train_count,
            "validation_loss": validation_loss_sum / validation_count,
            "train_accuracy": train_correct / train_count,
            "validation_accuracy": validation_correct / validation_count,
        }
        for metric_name, metric_value in metrics.items():
            history[metric_name].append(float(metric_value))

        if metrics["validation_loss"] < best_validation_loss:
            best_validation_loss = metrics["validation_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "image_size": IMAGE_SIZE,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"{model_name} epoch {epoch:02d}/{max_epochs}: "
            f"train loss={metrics['train_loss']:.4f}, "
            f"validation loss={metrics['validation_loss']:.4f}, "
            f"train accuracy={metrics['train_accuracy']:.3f}, "
            f"validation accuracy={metrics['validation_accuracy']:.3f}"
        )
        if epochs_without_improvement >= patience:
            print(f"{model_name}: early stopping after epoch {epoch}.")
            break

    if best_epoch == 0:
        raise RuntimeError(f"{model_name} training did not produce a best checkpoint.")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "checkpoint_path": str(checkpoint_path),
    }
