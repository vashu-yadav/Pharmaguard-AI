"""Grad-CAM explanations for tablet image classifiers."""

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from matplotlib import colormaps


def _last_convolution(model):
    if hasattr(model, "features") and model.__class__.__name__ == "CustomTabletCNN":
        for layer in reversed(list(model.features)):
            if isinstance(layer, torch.nn.Conv2d):
                return layer
    if hasattr(model, "features") and len(model.features) > 0:
        final_block = model.features[-1]
        if isinstance(final_block, torch.nn.Sequential):
            for layer in reversed(list(final_block)):
                if isinstance(layer, torch.nn.Conv2d):
                    return layer
        if isinstance(final_block, torch.nn.Conv2d):
            return final_block
    raise ValueError("Could not find a convolution layer for Grad-CAM.")


def explain_prediction(model, image, original_image: Image.Image | None = None, target_class=None):
    """Create and return a Grad-CAM overlay for one preprocessed image tensor.

    The overlay highlights image regions that influenced the selected class
    score. It is an explanation of the model, not a pharmaceutical safety test.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Grad-CAM expects the preprocessed image as a PyTorch tensor.")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    input_batch = image.unsqueeze(0) if image.ndim == 3 else image
    input_batch = input_batch.to(device).detach().requires_grad_(True)

    activations = []
    gradients = []

    def capture_activation(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    target_layer = _last_convolution(model)
    hook = target_layer.register_forward_hook(capture_activation)
    was_training = model.training
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_batch)
            if target_class is None:
                target_class = int(logits.argmax(dim=1)[0].item())
            logits[0, int(target_class)].backward()
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM did not receive feature activations and gradients.")

    activation = activations[0].detach()
    gradient = gradients[0].detach()
    channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
    heatmap = functional.interpolate(
        heatmap,
        size=(input_batch.shape[-2], input_batch.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    heatmap = heatmap.cpu().numpy()
    maximum = float(heatmap.max())
    if maximum > 0:
        heatmap = heatmap / maximum

    if original_image is None:
        # Convert the normalized tensor back to a viewable RGB image.
        from .config import IMAGENET_MEAN, IMAGENET_STD

        display_tensor = image.detach().cpu().clone()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        display_tensor = (display_tensor * std + mean).clamp(0, 1)
        original_image = Image.fromarray(
            (display_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

    base_image = original_image.convert("RGB").resize(
        (input_batch.shape[-1], input_batch.shape[-2]),
        Image.Resampling.BILINEAR,
    )
    heatmap_rgb = (colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    color_overlay = Image.fromarray(heatmap_rgb, mode="RGB")
    return Image.blend(base_image, color_overlay, alpha=0.40)
