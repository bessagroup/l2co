"""CIFAR-10 dataset downloading, processing, and task creation."""

import pickle
import tarfile
import urllib.request
from pathlib import Path

# Third-party
import jax.numpy as jnp
import numpy as np

# =============================================================================

# URL for CIFAR-10
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
file_name = "cifar-10-python.tar.gz"
extract_path = Path("./cifar10")


def download_cifar10():
    """Download, extract, and merge CIFAR-10 training batches.

    Returns
    -------
    dict[str, np.ndarray]
        ``'x'`` images ``(N, 32, 32, 3)`` and ``'y'`` labels
        ``(N,)``.
    """
    urllib.request.urlretrieve(url, file_name)

    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=extract_path)

    # Path to CIFAR-10 dataset

    cifar_path = extract_path / "cifar-10-batches-py"

    # Initialize dictionary to store merged data
    data_dict = {b"data": [], b"labels": []}

    # Loop over batch files and load them
    # CIFAR-10 has 5 training batches
    for i in range(1, 6):
        batch_file = cifar_path / f"data_batch_{i}"
        # batch_file = os.path.join(cifar_path, f"data_batch_{i}")
        with open(batch_file, "rb") as fo:
            batch = pickle.load(fo, encoding="bytes")

        # Append data and labels
        data_dict[b"data"].append(batch[b"data"])
        # Labels are lists
        data_dict[b"labels"].extend(batch[b"labels"])
        batch_file.unlink()

    (cifar_path / "batches.meta").unlink()

    # Stack all batches together
    data_dict[b"data"] = np.vstack(data_dict[b"data"])
    # Convert labels to NumPy array
    data_dict[b"labels"] = np.array(data_dict[b"labels"])

    data_dict[b"data"] = (
        data_dict[b"data"].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    )

    # # Remove cifar_path folder
    # cifar_path.rmdir()

    return {"x": data_dict[b"data"], "y": data_dict[b"labels"]}


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path):
    """Save a dataset dictionary to an ``.npz`` file.

    Parameters
    ----------
    dataset : dict[str, jnp.ndarray]
        Dictionary of arrays to persist.
    path : str or Path
        Destination file path.
    """
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


# def create_cifar10_task(
#     seed: int,
#     dataset_path: str,
# ) -> Task:
#     tag = {}

#     if not Path(dataset_path).exists():
#         dataset = download_cifar10()
#         save_dataset(dataset=dataset, path=dataset_path)
