import os


def get_model_weights_path(model_dir: str) -> str:
    """Get the path to the model weights file within a model directory.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Path to model.safetensors or model.safetensors.index.json
    """
    model_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(model_path):
        return model_path

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return index_path

    # If neither exists, return the directory itself (let import_weights handle it)
    return model_dir
