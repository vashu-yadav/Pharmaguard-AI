"""Reusable smart manufacturing prediction interface."""


def predict_machine(input_data):
    """Predict machine failure from prepared input data.

    Args:
        input_data: Machine features in the format expected by a future trained model.

    Raises:
        NotImplementedError: Until preprocessing and a trained model are available.
    """
    raise NotImplementedError(
        "Machine prediction is not implemented yet. No model is loaded or trained."
    )
