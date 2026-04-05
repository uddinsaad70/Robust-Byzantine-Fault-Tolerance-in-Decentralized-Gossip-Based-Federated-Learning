"""
data_loader.py
--------------
Downloads MNIST dataset and distributes it across all devices.

Supports two distribution modes:
  - IID   : Each device gets a random sample (uniform class distribution)
  - Non-IID: Each device gets data from only 2-3 classes (realistic scenario)
"""

import torch
import numpy as np
from torchvision import datasets, transforms


# ──────────────────────────────────────────
# Load MNIST from disk (downloads if needed)
# ──────────────────────────────────────────
def load_mnist(data_dir: str = "./data"):
    """
    Downloads and loads MNIST.
    Returns flattened tensors ready for the SimpleCNN model.

    Returns
    -------
    X_train : Tensor [60000, 1, 28, 28]  (float32, normalized 0-1)
    y_train : Tensor [60000]             (int64)
    X_test  : Tensor [10000, 1, 28, 28]
    y_test  : Tensor [10000]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),           # converts to [0,1] float
    ])

    train_set = datasets.MNIST(root=data_dir, train=True,
                                download=True, transform=transform)
    test_set  = datasets.MNIST(root=data_dir, train=False,
                                download=True, transform=transform)

    # Stack into tensors
    X_train = train_set.data.unsqueeze(1).float() / 255.0   # [60000,1,28,28]
    y_train = train_set.targets                              # [60000]
    X_test  = test_set.data.unsqueeze(1).float() / 255.0    # [10000,1,28,28]
    y_test  = test_set.targets                               # [10000]

    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────
# IID Distribution
# Each device gets an equal random slice
# ──────────────────────────────────────────
def distribute_iid(X_train, y_train, devices: list, seed: int = 42):
    """
    Splits training data randomly across devices (IID).
    Each device gets approximately 60000 / num_devices samples.
    Class distribution is approximately uniform for all devices.
    """
    n          = len(devices)
    num_total  = len(X_train)
    per_device = num_total // n

    # Shuffle indices
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(num_total)

    for i, device in enumerate(devices):
        start = i * per_device
        end   = start + per_device if i < n - 1 else num_total
        idx   = torch.tensor(indices[start:end], dtype=torch.long)
        device.local_data = (X_train[idx], y_train[idx])


# ──────────────────────────────────────────
# Non-IID Distribution
# Each device gets data from only 2 classes
# (simulates realistic federated scenario)
# ──────────────────────────────────────────
def distribute_non_iid(X_train, y_train, devices: list,
                        classes_per_device: int = 2, seed: int = 42):
    """
    Splits training data in a Non-IID manner.
    Each device is assigned `classes_per_device` classes.
    Data is drawn only from those classes for that device.

    This simulates real-world FL where each user's data
    reflects their own behavior (e.g., a user who only
    writes digits 3 and 7).
    """
    rng        = np.random.default_rng(seed)
    num_classes = 10
    n           = len(devices)

    # Build a list of class assignments for each device
    # Cycle through classes so all classes are covered
    all_classes   = list(range(num_classes))
    class_pool    = (all_classes * ((n * classes_per_device) // num_classes + 1))
    rng.shuffle(class_pool)

    # Group indices by class
    class_indices = {}
    for c in range(num_classes):
        class_indices[c] = np.where(y_train.numpy() == c)[0].tolist()
        rng.shuffle(class_indices[c])

    # Track how far into each class's index list we are
    class_ptr = {c: 0 for c in range(num_classes)}

    for i, device in enumerate(devices):
        # Assign classes_per_device classes to this device
        assigned_classes = class_pool[i * classes_per_device:
                                      (i + 1) * classes_per_device]

        device_indices = []
        samples_per_class = 3000 // classes_per_device  # ~3000 samples per device

        for c in assigned_classes:
            ptr   = class_ptr[c]
            avail = class_indices[c][ptr: ptr + samples_per_class]
            device_indices.extend(avail)
            class_ptr[c] += len(avail)

        rng.shuffle(device_indices)
        idx = torch.tensor(device_indices, dtype=torch.long)
        device.local_data = (X_train[idx], y_train[idx])


# ──────────────────────────────────────────
# Print Distribution Summary
# ──────────────────────────────────────────
def print_distribution_summary(devices: list):
    print("=" * 65)
    print(f"{'DATA DISTRIBUTION SUMMARY':^65}")
    print("=" * 65)
    print(f"  {'Device':<30} {'Samples':>8}  {'Class distribution'}")
    print("-" * 65)

    for d in devices:
        if d.local_data is None:
            print(f"  Device {d.id:2d}: NO DATA")
            continue

        X, y        = d.local_data
        total       = len(y)
        class_counts = {}
        for c in range(10):
            count = (y == c).sum().item()
            if count > 0:
                class_counts[c] = count

        class_str = "  ".join(f"{c}:{v}" for c, v in sorted(class_counts.items()))
        byz       = " [BYZ]" if d.is_byzantine else ""
        print(f"  Device {d.id:2d} ({d.device_type:12s}){byz:7s} "
              f"{total:>6} samples  |  {class_str}")

    print("=" * 65)
