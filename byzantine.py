"""
byzantine.py — Phase 5: Quality Assessment & Byzantine Detection
Source: Gossip FL PDF Section 3.5

Comparison: both sides in pre-noise DCT domain (Phase 2 output)
    own_flat  = flatten(own_compressed)      pre-noise, L2 ~ 4.4  (honest)
    recv_flat = flatten(msg['compressed'])   pre-noise, L2 ~ 1582 (Byzantine)
    mag_ratio = 4.4/1582 = 0.003 < 0.1 → BYZANTINE detected

Three Tests (PDF 3.5.1):
    cos_sim   < -0.5   → Byzantine
    mag_ratio < 0.1    → Byzantine
    outlier%  > 30%    → Byzantine

Quality: 0.6×cos_sim + 0.4×mag_ratio  (or 0.0 if Byzantine)
Reputation: rep_new = 0.8×rep_old + 0.2×quality
"""

import numpy as np
import torch


def _flatten_compressed_dict(d: dict) -> np.ndarray:
    """Flatten {layer: {'data': tensor, ...}} to 1D."""
    arrays = [pkg["data"].detach().cpu().numpy().flatten() for pkg in d.values()]
    if not arrays:
        raise ValueError("compressed dict is empty — gossip message missing 'compressed' field")
    return np.concatenate(arrays)


def _flatten_tensor_dict(d: dict) -> np.ndarray:
    """Flatten {layer: tensor} to 1D."""
    arrays = [t.detach().cpu().numpy().flatten() for t in d.values()]
    if not arrays:
        raise ValueError("tensor dict is empty")
    return np.concatenate(arrays)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def magnitude_ratio(a: np.ndarray, b: np.ndarray) -> float:
    na    = np.linalg.norm(a)
    nb    = np.linalg.norm(b)
    denom = max(na, nb)
    if denom == 0:
        return 1.0
    return float(min(na, nb) / denom)


def outlier_pct_iqr(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(b - a)
    if len(diff) < 4:
        return 0.0
    Q3        = np.percentile(diff, 75)
    Q1        = np.percentile(diff, 25)
    threshold = Q3 + 1.5 * (Q3 - Q1)
    return float(np.sum(diff > threshold) / len(diff))


def compute_quality(own: np.ndarray, recv: np.ndarray) -> tuple:
    cos_sim = cosine_similarity(own, recv)
    mag_rat = magnitude_ratio(own, recv)
    out_pct = outlier_pct_iqr(own, recv)

    # কন্ডিশন একদম ঠিক আছে
    if out_pct > 0.40 or cos_sim < -0.9 or mag_rat < 0.1:
        quality = 0.0
    else:
        # cos_sim কে [-1, 1] থেকে [0, 1] স্কেলে কনভার্ট করা হলো
        normalized_cos = (cos_sim + 1.0) / 2.0
        
        # এখন quality কখনোই নেগেটিভ হবে না
        quality = 0.6 * normalized_cos + 0.4 * mag_rat
        
        # অতিরিক্ত সেফটির জন্য: কোয়ালিটি যেন কোনোভাবেই 0 বা নেগেটিভ না হয়
        quality = max(0.01, quality)

    return quality, cos_sim, mag_rat, out_pct


def assess_received_gradients(device, received_messages: list,
                               own_compressed: dict) -> dict:
    """
    Compare own Phase 2 compressed gradient vs each received Phase 2
    compressed gradient. Both pre-noise, same DCT domain.
    """
    if own_compressed is None or not received_messages:
        return {}

    own_flat       = _flatten_compressed_dict(own_compressed)
    quality_scores = {}

    for msg in received_messages:
        sender_id = msg["sender"]

        recv_compressed = msg.get("compressed", {})
        if not recv_compressed:
            # gossip.py is old version without 'compressed' field
            raise RuntimeError(
                f"Message from Device {sender_id} has no 'compressed' field. "
                "Replace gossip.py with the latest version."
            )

        recv_flat = _flatten_tensor_dict(recv_compressed)

        n        = min(len(own_flat), len(recv_flat))
        own_cmp  = own_flat[:n]
        recv_cmp = recv_flat[:n]

        quality, cos_sim, mag_rat, out_pct = compute_quality(own_cmp, recv_cmp)

        rep_old = device.reputation.get(sender_id, 1.0)
        rep_new = round(0.8 * rep_old + 0.2 * quality, 4)
        device.reputation[sender_id] = rep_new

        # # DEBUG — Device 17 এর detection metrics track করতে
        # if sender_id == 17 and quality > 0:  # শুধু escape হলে দেখাবে
        #     print(f"      [D17 ESCAPED] "
        #         f"cos={cos_sim:.3f} mag={mag_rat:.3f} "
        #         f"out={out_pct:.3f} quality={quality:.3f} "
        #         f"rep={rep_new:.4f}")

        # DEBUG — Device 17 ধরা পড়লে (Detected) metrics track করতে
        if sender_id == 17 and quality == 0:  # শুধু ধরা পড়লে (Quality 0 হলে) দেখাবে
            print(f"      [D17 DETECTED] "
                f"cos={cos_sim:.3f} mag={mag_rat:.3f} "
                f"out={out_pct:.3f} quality={quality:.3f} "
                f"rep={rep_new:.4f}")
            
        quality_scores[sender_id] = {
            "quality":      round(quality, 4),
            "cos_sim":      round(cos_sim, 4),
            "mag_ratio":    round(mag_rat, 4),
            "outlier_pct":  round(out_pct, 4),
            "rep_old":      round(rep_old, 4),
            "rep_new":      rep_new,
            "is_byzantine": quality == 0.0,
        }

    return quality_scores


def run_phase5(devices: list, received: dict,
               compressed_gradients: dict) -> dict:
    all_quality = {}
    for d in devices:
        msgs     = received.get(d.id, [])
        own_comp = compressed_gradients.get(d.id)
        all_quality[d.id] = assess_received_gradients(d, msgs, own_comp)
    return all_quality