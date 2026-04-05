"""
gossip.py — Phase 4: Gossip Exchange
Source: Gossip FL PDF Section 3.4
"""

import numpy as np


def prepare_message(device, noisy_gradient: dict,
                    compressed_gradient: dict, round_num: int) -> dict:
    """
    Build gossip message.
    'gradient'   = noisy (Phase 3 output) → Phase 6 aggregation
    'compressed' = pre-noise (Phase 2 output) → Phase 5 detection
    """
    gradient_data   = {}
    compressed_data = {}
    metadata        = {}

    for name, pkg in noisy_gradient.items():
        gradient_data[name] = pkg["data"]
        metadata[name] = {
            "Cr":             pkg["Cr"],
            "mask":           pkg["mask"],
            "original_shape": pkg["original_shape"],
        }

    for name, pkg in compressed_gradient.items():
        compressed_data[name] = pkg["data"]   # pre-noise DCT tensor

    # Sanity check — compressed must NOT be empty
    assert len(compressed_data) > 0, \
        f"Device {device.id}: compressed_data is empty in gossip message!"

    return {
        "sender":     device.id,
        "round":      round_num,
        "gradient":   gradient_data,    # noisy  → Phase 6
        "compressed": compressed_data,  # clean  → Phase 5
        "metadata":   metadata,
    }


def gossip_exchange(devices: list,
                    noisy_gradients: dict,
                    round_num: int,
                    compressed_gradients: dict) -> dict:
    """
    Bidirectional gossip: every device sends to and receives from all neighbors.

    Parameters
    ----------
    noisy_gradients      : {id: noisy_gradient}       Phase 3 output → aggregation
    compressed_gradients : {id: compressed_gradient}  Phase 2 output → detection
    """
    # Build outbox
    outbox = {}
    for d in devices:
        if d.id in noisy_gradients and d.id in compressed_gradients:
            outbox[d.id] = prepare_message(
                d,
                noisy_gradients[d.id],
                compressed_gradients[d.id],
                round_num,
            )

    # Deliver messages bidirectionally
    received = {d.id: [] for d in devices}
    seen     = {d.id: set() for d in devices}

    for d in devices:
        if d.id not in outbox:
            continue
        for nid in d.neighbors:
            if d.id not in seen[nid] and d.id in outbox:
                received[nid].append(outbox[d.id])
                seen[nid].add(d.id)
            if nid not in seen[d.id] and nid in outbox:
                received[d.id].append(outbox[nid])
                seen[d.id].add(nid)

    return received


def compute_traffic(devices: list, noisy_gradients: dict) -> dict:
    traffic = {}
    for d in devices:
        if d.id not in noisy_gradients:
            continue
        ng            = noisy_gradients[d.id]
        nonzero       = sum((pkg["data"] != 0).sum().item() for pkg in ng.values())
        bytes_per_msg = nonzero * 4
        n_neighbors   = len(d.neighbors)
        traffic[d.id] = {
            "bytes_per_message": bytes_per_msg,
            "num_neighbors":     n_neighbors,
            "bytes_sent":        bytes_per_msg * n_neighbors,
            "bytes_received":    bytes_per_msg * n_neighbors,
            "total_bytes":       bytes_per_msg * n_neighbors * 2,
        }
    return traffic