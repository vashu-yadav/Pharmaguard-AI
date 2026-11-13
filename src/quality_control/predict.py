"""Reusable pharmaceutical tablet image prediction interface."""


def predict_tablet(image):
    """Predict tablet image quality from an input image.

    Args:
        image: Tablet image in a format accepted by a future image pipeline.

    Raises:
        NotImplementedError: Until preprocessing and a trained model are available.
    """
    raise NotImplementedError(
        "Tablet prediction is not implemented yet. No image model is loaded or trained."
    )
