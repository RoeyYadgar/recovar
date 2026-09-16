"""Dataset index mappings shared by dense-EM host orchestration."""

import numpy as np


def original_indices_for_local(experiment_dataset, local_indices) -> np.ndarray:
    """Map local image indices to their original dataset image ids."""

    local_indices = np.asarray(local_indices, dtype=np.int64)
    mapper = getattr(experiment_dataset, "original_image_indices_from_local", None)
    if mapper is not None:
        return np.asarray(mapper(local_indices), dtype=np.int64)
    original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
    if original_indices_all is None:
        return local_indices
    return np.asarray(original_indices_all, dtype=np.int64)[local_indices]
