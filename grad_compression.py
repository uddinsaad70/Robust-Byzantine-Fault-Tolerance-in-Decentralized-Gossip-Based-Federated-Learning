"""
grad_compression.py
-------------------
Phase 2: Adaptive Gradient Compression
Source: Hidayat et al. (2024) Algorithm 2, Section IV-E, IV-F

Core principle:
    Cr বড় (Pi)      → threshold lt বড় → বেশি prune → বেশি compression
    Cr ছোট (Desktop) → threshold lt ছোট → কম prune  → কম compression

    dt (DCT threshold) = Cr — same direction:
    Cr বড় → dt বড় → DCT-ও বেশি কাটে → আরো বেশি sparsity
    Cr ছোট → dt ছোট → DCT কম কাটে → কম sparsity

Expected sparsity:
    Pi      Cr=0.150 → HIGH sparsity (~85-95%)
    Laptop  Cr=0.037 → MEDIUM sparsity (~60-75%)
    Desktop Cr=0.019 → LOW sparsity (~40-55%)
"""

import numpy as np
import torch
from scipy.fft import dct, idct


def compute_compression_ratio(device, model_size: int) -> float:
    """
    Cr = Mu / (Cc / Cu)   [Paper Eq. 13-14]

    Pi (2c, 2GB):    Cr ≈ 0.150  → বড় → বেশি compression
    Desktop (16c, 32GB): Cr ≈ 0.019 → ছোট → কম compression
    """
    r  = device.resources
    Ma = r["ram_gb"] * 1024
    Ms = Ma * 0.50
    Mt = (model_size * 4) / (1024 * 1024)
    Mu = (Ms + Mt) / Ma
    Cc = r["cpu_cores"]
    Cu = 0.60
    Cr = Mu / (Cc / Cu)
    return float(np.clip(Cr, 0.01, 0.95))


def compress_tensor(tensor: torch.Tensor, Cr: float):
    """
    Algorithm 2 with dt = Cr (both thresholds scale together).

    Pruning threshold: lt = Cr × max(|weights|)
        Pi Cr=0.150 → lt বড় → বেশি weight → 0
        Desktop Cr=0.019 → lt ছোট → কম weight → 0

    DCT threshold: dt = Cr (same value)
        Pi Cr=0.150 → dt বড় → বেশি DCT coeff → 0
        Desktop Cr=0.019 → dt ছোট → কম DCT coeff → 0

    Combined effect:
        Pi      → high sparsity (~85-95%)   ✓ বেশি compress
        Desktop → low sparsity  (~40-55%)   ✓ কম compress
    """
    original_shape = tensor.shape
    fw = tensor.detach().cpu().numpy().flatten().astype(np.float64)

    # Weight pruning threshold (Algorithm 2, line 10)
    sw = np.sort(np.abs(fw))
    lt = Cr * np.max(sw) if len(sw) > 0 else 0.0
    clw = np.where(np.abs(fw) >= lt, fw, 0.0)

    # DCT (Algorithm 2, line 19)
    dw = dct(clw, norm="ortho")

    # DCT mask with dt = Cr (Algorithm 2, lines 20-28)
    dt     = Cr   # ← scales with device capability
    max_dw = np.max(np.abs(dw)) if np.any(dw != 0) else 1.0
    mask   = (np.abs(dw) >= dt * max_dw).astype(np.float32)
    tw     = dw * mask

    compressed = torch.tensor(tw.reshape(original_shape), dtype=torch.float32)
    return compressed, mask, original_shape


def compress_gradient(gradient: dict, device, model_size: int) -> dict:
    Cr     = compute_compression_ratio(device, model_size)
    result = {}
    for name, tensor in gradient.items():
        compressed, mask, shape = compress_tensor(tensor, Cr)
        result[name] = {
            "data":           compressed,
            "mask":           mask,
            "original_shape": shape,
            "Cr":             Cr,
        }
    return result


def decompress_gradient(compressed_gradient: dict) -> dict:
    gradient = {}
    for name, pkg in compressed_gradient.items():
        flat    = pkg["data"].detach().cpu().numpy().flatten().astype(np.float64)
        spatial = idct(flat, norm="ortho")
        gradient[name] = torch.tensor(
            spatial.reshape(pkg["original_shape"]), dtype=torch.float32
        )
    return gradient


def compression_stats(original: dict, compressed: dict) -> dict:
    orig_nonzero = sum((t != 0).sum().item() for t in original.values())
    comp_nonzero = sum((pkg["data"] != 0).sum().item()
                       for pkg in compressed.values())
    total        = sum(t.numel() for t in original.values())
    Cr           = list(compressed.values())[0]["Cr"]
    return {
        "Cr":               round(Cr, 4),
        "total_params":     total,
        "nonzero_before":   orig_nonzero,
        "nonzero_after":    comp_nonzero,
        "sparsity_%":       round(100 * (1 - comp_nonzero / total), 2),
        "size_reduction_%": round(100 * (1 - comp_nonzero / max(orig_nonzero, 1)), 2),
    }