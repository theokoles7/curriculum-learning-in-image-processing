"""Initialize model based on selection."""

from models.cnn import CNN

def get_model(
    model:          str,
    channels_in:    int,
    channels_out:   int,
    dim:            int
) -> CNN:
    """Initialize and return model based on selection.

    Args:
        model (str): Model selection.
        chanells_in (int): Input channels.
        channels_out (int): Output channels.
        dim (int): Input imnage dimension.

    Returns:
        CNN: Selected model.
    """
    # Match model
    match model:

        case "cnn": return CNN(channels_in = channels_in, channels_out = channels_out, dim = dim)

        case _:     raise ValueError(f"{ARGS.model} is not a valid model selection.")