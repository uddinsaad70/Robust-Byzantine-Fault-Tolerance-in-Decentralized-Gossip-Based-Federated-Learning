"""
aggregation.py  [Thesis version — Task 2 + Task 3]
===================================================
Task 2: তিনটি aggregation rule implement করা হয়েছে
    "fedavg"      — weighted mean (baseline)
    "multi_krum"  — Blanchard et al. (NeurIPS 2017)
    "median"      — Coordinate-wise Median, Yin et al. (ICML 2018)

Task 3: Model update fixed
    OLD (wrong): param.data += lr * aggregated   ← gradient ascent
    NEW (fixed): param.data -= lr * aggregated   ← gradient descent

AGGREGATION_METHOD variable main.py থেকে pass করা হয়।
"""

import torch
import numpy as np
from grad_compression import decompress_gradient


# ═══════════════════════════════════════════════════════════
# Helper: received messages থেকে gradient list বানানো
# ═══════════════════════════════════════════════════════════
def _collect_gradients(device, received_messages, quality_scores, method):
    """
    Returns list of (weight, gradient_dict) tuples.
    weight depends on method:
        fedavg     → reputation × quality
        multi_krum → 1.0 for selected, 0.0 for excluded
        median     → 1.0 for all (median handles outliers internally)
    """
    msg_by_sender = {m["sender"]: m for m in received_messages}
    candidates = []  # list of (sender_id_or_"own", gradient_dict)

    # Own gradient always included
    if device.local_gradient:
        candidates.append(("own", device.local_gradient))

    for sender_id, qinfo in quality_scores.items():
        msg = msg_by_sender.get(sender_id)
        if msg is None:
            continue
        # Decompress
        compressed_pkg = {
            name: {
                "data":           tensor,
                "mask":           msg["metadata"][name]["mask"],
                "original_shape": msg["metadata"][name]["original_shape"],
                "Cr":             msg["metadata"][name]["Cr"],
            }
            for name, tensor in msg["gradient"].items()
        }
        grad = decompress_gradient(compressed_pkg)
        candidates.append((sender_id, grad))

    return candidates, msg_by_sender, quality_scores


# ═══════════════════════════════════════════════════════════
# GAR 1: FedAvg (baseline)
# ═══════════════════════════════════════════════════════════
def _fedavg(device, received_messages, quality_scores):
    """
    Reputation-weighted mean.
    weight_j = reputation_j × quality_j
    Byzantine: quality=0 → excluded automatically.
    """
    if device.local_gradient is None:
        return None, {}

        # Use velocity (momentum-smoothed) for local contribution if available
    local_g = getattr(device, 'local_velocity', device.local_gradient)
    
    weighted_grads = [(1.0, local_g)]
    total_w = 1.0
    w_used = {"own": 1.0}
    n_excl = 0

    msg_by_sender = {m["sender"]: m for m in received_messages}
    for sid, qinfo in quality_scores.items():
        q   = qinfo["quality"]
        rep = device.reputation.get(sid, 1.0)
        w   = rep * q
        if w <= 0:
            n_excl += 1; w_used[sid] = 0.0; continue
        msg = msg_by_sender.get(sid)
        if msg is None: continue
        pkg = {n: {"data":t,"mask":msg["metadata"][n]["mask"],
                   "original_shape":msg["metadata"][n]["original_shape"],
                   "Cr":msg["metadata"][n]["Cr"]}
               for n,t in msg["gradient"].items()}
        weighted_grads.append((w, decompress_gradient(pkg)))
        total_w += w; w_used[sid] = round(w,4)

    if total_w == 0: return None, {}

    aggregated = {}
    for name in device.local_gradient:
        agg = torch.zeros_like(device.local_gradient[name])
        for w, g in weighted_grads:
            if name in g:
                agg += w * g[name].to(agg.device)
        aggregated[name] = agg / total_w

    return aggregated, {"method":"fedavg","total_weight":round(total_w,4),
                        "n_contributors":len(weighted_grads),"n_excluded":n_excl,
                        "weights":w_used}


# ═══════════════════════════════════════════════════════════
# GAR 2: Multi-Krum
# ═══════════════════════════════════════════════════════════
def _multi_krum(device, received_messages, quality_scores, f=1, m=None):

    if device.local_gradient is None:
        return None, {}

    def _flatten(grad_dict):
        return torch.cat([v.flatten().to("cpu") for v in grad_dict.values()])

    all_grads = []
    all_keys  = []

    # Use velocity (momentum-smoothed) for local contribution
    local_g = getattr(device, 'local_velocity', device.local_gradient)

    # Own
    all_grads.append(_flatten(local_g))
    all_keys.append("own")

    msg_by_sender = {m["sender"]: m for m in received_messages}
    raw_grads = {"own": local_g}  # sender_id_or_"own" → gradient_dict

    for sid, qinfo in quality_scores.items():
        msg = msg_by_sender.get(sid)
        if msg is None: continue
        pkg = {n: {"data":t,"mask":msg["metadata"][n]["mask"],
                   "original_shape":msg["metadata"][n]["original_shape"],
                   "Cr":msg["metadata"][n]["Cr"]}
               for n,t in msg["gradient"].items()}
        g = decompress_gradient(pkg)
        all_grads.append(_flatten(g))
        all_keys.append(sid)
        raw_grads[sid] = g

    n = len(all_grads)
    if m is None:
        m = max(1, n - f)
    k = max(1, n - f - 2)   # neighbors to consider per node

    if n <= 2:
        return device.local_gradient, {"method":"multi_krum","n":n,"selected":["own"]}

    G   = torch.stack(all_grads)          # [n, D]
    dist = torch.cdist(G, G, p=2)         # [n, n]

    scores = []
    for i in range(n):
        d_i = dist[i].clone()
        d_i[i] = float("inf")             # self 제외
        knn_dist = d_i.topk(k, largest=False).values
        scores.append(knn_dist.sum().item())

    # ──────────────────────────────────────────
    # [KRUM DEBUG] Print D17 vs Honest Scores
    # ──────────────────────────────────────────
    for i in range(n):
        node_id = all_keys[i]
        score_val = scores[i]
        if node_id == 17:
            print(f"    [KRUM DEBUG] D17 (Hacker) Score: {score_val:.4f}")
        elif node_id != "own": # own বাদ দিয়ে অন্য যেকোনো একজন সৎ প্রতিবেশীর স্কোর
            print(f"    [KRUM DEBUG] Honest D{node_id} Score: {score_val:.4f}")
            break # একজন সৎ মানুষের স্কোর প্রিন্ট করেই লুপ ব্রেক, যাতে স্ক্রিন ভরে না যায়
    # ──────────────────────────────────────────

    order    = sorted(range(n), key=lambda i: scores[i])
    selected = order[:m]
    sel_keys = [all_keys[i] for i in selected]

    aggregated = {}
    for name in device.local_gradient:
        agg = torch.zeros_like(device.local_gradient[name])
        cnt = 0
        for key in sel_keys:
            g = raw_grads.get(key)
            if g and name in g:
                agg += g[name].to(agg.device); cnt += 1
        aggregated[name] = agg / max(cnt,   1)

    return aggregated, {"method":"multi_krum","n":n,"f":f,"m":m,
                        "selected":sel_keys,
                        "scores":{all_keys[i]:round(scores[i],4) for i in range(n)}}


# ═══════════════════════════════════════════════════════════
# GAR 3: Coordinate-wise Median
# ═══════════════════════════════════════════════════════════
def _coordinate_median(device, received_messages, quality_scores):

    if device.local_gradient is None:
        return None, {}

    msg_by_sender = {m["sender"]: m for m in received_messages}

    # Use velocity (momentum-smoothed) for local contribution
    local_g = getattr(device, 'local_velocity', device.local_gradient)
    all_grads = [local_g]

    for sid in quality_scores:
        msg = msg_by_sender.get(sid)
        if msg is None: continue
        pkg = {n: {"data":t,"mask":msg["metadata"][n]["mask"],
                   "original_shape":msg["metadata"][n]["original_shape"],
                   "Cr":msg["metadata"][n]["Cr"]}
               for n,t in msg["gradient"].items()}
        all_grads.append(decompress_gradient(pkg))

    n = len(all_grads)
    aggregated = {}
    for name in device.local_gradient:
        stack = torch.stack([g[name].to("cpu") for g in all_grads if name in g])
        aggregated[name] = stack.median(dim=0).values

    return aggregated, {"method":"coord_median","n":n}


# ═══════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════
def aggregate_and_update(device, received_messages, quality_scores,
                         learning_rate=0.01, method="fedavg"):
    method = method.lower()

    if method == "fedavg":
        aggregated, info = _fedavg(device, received_messages, quality_scores)
    elif method in ("multi_krum", "krum"):
        aggregated, info = _multi_krum(device, received_messages, quality_scores)
    elif method in ("median", "coord_median"):
        aggregated, info = _coordinate_median(device, received_messages, quality_scores)
    else:
        raise ValueError(f"Unknown aggregation method: '{method}'")

    if aggregated is None:
        return {}

    # Task 3: W -= lr * grad  (correct gradient descent)
    with torch.no_grad():
        for name, param in device.model.named_parameters():
            if name in aggregated:
                param.data -= learning_rate * aggregated[name].to(param.device)

    return info


def run_phase6(devices, received, all_quality, learning_rate=0.01,
               method="fedavg"):
    results = {}
    for d in devices:
        msgs    = received.get(d.id, [])
        quality = all_quality.get(d.id, {})
        results[d.id] = aggregate_and_update(
            d, msgs, quality, learning_rate, method)
    return results