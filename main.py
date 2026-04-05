"""
main.py  [Thesis version — Task 2 + Task 3]
============================================
দুটো variable বদলাও, বাকি সব auto:

    ATTACK              = "none" | "label_flip" | "alie"
    AGGREGATION_METHOD  = "fedavg" | "multi_krum" | "median"

Task 3: distribute_non_iid() ব্যবহার করা হচ্ছে (আগে iid ছিল)।
"""

import torch, copy, time, io, sys

from topology       import create_devices, build_topology
from data_loader    import load_mnist, distribute_non_iid
from grad_compression import compress_gradient
from privacy        import apply_differential_privacy
from gossip         import gossip_exchange
from byzantine      import run_phase5
from aggregation    import run_phase6
from device         import get_model_size

# ══════════════════════════════════════════════════════════
#  Thesis experiment configuration — এই দুটো লাইনই বদলাও
DATASET             = "mnist"        # "mnist" | "fmnist" | "har"
ATTACK              = "label_flip"   # "none" | "label_flip" | "alie"
AGGREGATION_METHOD  = "multi_krum"   # "fedavg" | "multi_krum" | "median"
# ══════════════════════════════════════════════════════════

NUM_ROUNDS    = 300
BATCH_SIZE    = 64
BASE_LR       = 0.01
LR_DECAY      = 0.95
LR_DECAY_STEP = 50
CLASSES_PER_DEVICE = 10   # IID: প্রতিটি device সব class দেখবে 
WEIGHT_SYNC_INTERVAL = 10   # sync weights every 10 rounds



def section(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def get_lr(r): return BASE_LR*(LR_DECAY**((r-1)//LR_DECAY_STEP))
def evaluate_all(devs,Xt,yt): return {d.id:d.evaluate(Xt,yt) for d in devs}
def avg_honest(acc,devs):
    v=[acc[d.id] for d in devs if not d.is_byzantine]; return sum(v)/len(v)


def gossip_sync_weights(devices: list, G, quality_scores_last: dict,
                        sync_fraction: float = 0.5):
    """
    Fix 3: periodic weight averaging via gossip.

    Math (D-PSGD, Lian et al. NeurIPS 2017, Eq. 4):
        W_i(t+1) = Σ_j W_ij · W_j(t)
    where W_ij = 1/(degree+1) for self and each neighbor (uniform mixing).

    Why this prevents divergence:
        In pure gradient sharing, each device integrates its neighbors'
        *gradient directions*, which are computed on different weight
        manifolds after divergence. Weight averaging collapses all devices
        back toward a common point on the loss surface every K rounds,
        resetting the divergence clock.

    Byzantine safety:
        We use the last round's quality_scores to exclude Byzantine
        neighbors from weight averaging. Only devices with quality > 0
        contribute their weights.
    """
    # Build new weight state for each device (do not mutate in-place yet)
    new_state = {}
    for d in devices:
        own_state  = d.model.state_dict()
        my_quality = quality_scores_last.get(d.id, {})

        # Collect honest neighbors' weights
        neighbor_states = []
        for nid in d.neighbors:
            qinfo = my_quality.get(nid, {})
            if qinfo.get('is_byzantine', False):
                continue   # exclude flagged Byzantine neighbors
            neighbor_dev = next((x for x in devices if x.id == nid), None)
            if neighbor_dev:
                neighbor_states.append(neighbor_dev.model.state_dict())

        if not neighbor_states:
            new_state[d.id] = own_state
            continue

        # Uniform mixing: self gets weight (1-sync_fraction), neighbors share sync_fraction
        w_self = 1.0 - sync_fraction
        w_each = sync_fraction / len(neighbor_states)

        mixed = {}
        for key in own_state:
            acc = w_self * own_state[key].float()
            for ns in neighbor_states:
                acc = acc + w_each * ns[key].float()
            mixed[key] = acc
        new_state[d.id] = mixed

    # Apply all at once (prevents order-dependent updates)
    for d in devices:
        if d.id in new_state:
            d.model.load_state_dict(new_state[d.id])



def run_experiment(run_byzantine=False):
    """
    run_byzantine=False → all devices honest (baseline)
    run_byzantine=True  → Device 17 is Byzantine with ATTACK strategy
    """
    _buf=io.StringIO()
    _tee=type('T',(),{
        'write':lambda s,x:[sys.__stdout__.write(x),_buf.write(x)],
        'flush':lambda s:[sys.__stdout__.flush(),_buf.flush()]})()
    sys.stdout=_tee

    MODEL_SIZE=get_model_size(DATASET)
    attack_tag=ATTACK if run_byzantine else "none"

    section("PHASE 0 — Initialization")
    print(f"  Dataset      : {DATASET.upper()}")
    print(f"  Attack       : {attack_tag.upper()}")
    print(f"  Aggregation  : {AGGREGATION_METHOD.upper()}")
    print(f"  Distribution : Non-IID ({CLASSES_PER_DEVICE} classes/device)")
    print(f"  Model size   : {MODEL_SIZE:,} params", flush=True)

    devices=create_devices(dataset=DATASET, attack=attack_tag)
    G,manager=build_topology(devices)

    if run_byzantine:
        for d in devices:
            if d.id==17:
                d.is_byzantine=True
                print(f"  Device 17 → BYZANTINE [{attack_tag.upper()}]")
                break

    W0=copy.deepcopy(devices[0].model.state_dict())
    for d in devices: d.model.load_state_dict(W0); d.init_reputation()

    print(f"  Loading {DATASET.upper()}...", flush=True)
    X_train,y_train,X_test,y_test=load_mnist(DATASET)
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}", flush=True)

    # Task 3: Non-IID distribution
    distribute_non_iid(X_train, y_train, devices,
                       classes_per_device=CLASSES_PER_DEVICE)
    print(f"  Non-IID: each device gets {CLASSES_PER_DEVICE} classes only")

    byz_id=next((d.id for d in devices if d.is_byzantine), None)
    acc_init=evaluate_all(devices,X_test,y_test)
    print(f"  Initial acc  : {avg_honest(acc_init,devices):.2f}%")

    history=[]
    section(f"TRAINING — {NUM_ROUNDS} rounds | {AGGREGATION_METHOD.upper()} | {attack_tag.upper()}")
    print(f"\n  {'Rnd':>5}  {'LR':>7}  {'Acc':>8}  {'Flags':>7}  "
          f"{'P1-3':>6}  {'P4':>5}  {'P5':>5}  {'P6':>5}  {'Ev':>5}")
    print(f"  {'-'*65}")

    for rnd in range(1,NUM_ROUNDS+1):
        t0=time.time(); lr=get_lr(rnd)

        # Phase 1-3
        tp13=time.time()
        noisy_g,comp_g={},{}
        for d in devices:
            g=d.local_train(BATCH_SIZE)
            c=compress_gradient(g,d,MODEL_SIZE)
            # Fix 2: DP noise before gossip, same for all recipients.
            n,_=apply_differential_privacy(c, round_num=rnd)
            comp_g[d.id]=c; noisy_g[d.id]=n
        tp13=time.time()-tp13

        # Phase 4
        tp4=time.time()
        recv=gossip_exchange(devices,noisy_g,rnd,comp_g)
        tp4=time.time()-tp4

        # Phase 5
        tp5=time.time()
        qual=run_phase5(devices,recv,comp_g)
        flags=sum(1 for sc in qual.values()
                  for i in sc.values() if i["is_byzantine"])
        tp5=time.time()-tp5

        # Phase 6 — chosen GAR
        tp6=time.time()
        run_phase6(devices,recv,qual,lr,method=AGGREGATION_METHOD)
        tp6=time.time()-tp6

        # Fix 3: periodic weight sync (prevents Non-IID weight divergence)
        if rnd % WEIGHT_SYNC_INTERVAL == 0:
            gossip_sync_weights(devices, G, qual, sync_fraction=0.4)

        # Evaluate
        tev=time.time()
        acc=evaluate_all(devices,X_test,y_test); aa=avg_honest(acc,devices)
        tev=time.time()-tev; elapsed=time.time()-t0

        rep_snap={}
        if byz_id:
            rep_snap={d.id:round(d.reputation[byz_id],4)
                      for d in devices if not d.is_byzantine
                      and byz_id in d.reputation}

        history.append({"round":rnd,"avg_acc":aa,"byz_flags":flags,
                        "lr":lr,"rep_snap":rep_snap})

        print(f"  {rnd:>5}  {lr:>7.5f}  {aa:>7.2f}%  {flags:>7}  "
              f"{tp13:>5.1f}s {tp4:>4.1f}s {tp5:>4.1f}s "
              f"{tp6:>4.1f}s {tev:>4.1f}s", flush=True)

        if rnd%10==0 and rep_snap and byz_id:
            avg_r=sum(rep_snap.values())/len(rep_snap)
            print(f"       D{byz_id}[BYZ] rep avg={avg_r:.4f} "
                  f"min={min(rep_snap.values()):.4f}", flush=True)

        if rnd%25==0:
            rem=elapsed*(NUM_ROUNDS-rnd); h,m=divmod(int(rem),3600); m=m//60
            print(f"\n  -- {rnd/NUM_ROUNDS*100:.0f}% | ~{h}h {m}m left --\n",
                  flush=True)

    section("FINAL RESULTS")
    final=evaluate_all(devices,X_test,y_test)
    honest_acc=[final[d.id] for d in devices if not d.is_byzantine]
    havg=sum(honest_acc)/len(honest_acc)

    print(f"\n  {'Device':<22} {'Accuracy':>10}")
    print(f"  {'-'*34}")
    for d in devices:
        tag=" [BYZ]" if d.is_byzantine else ""
        print(f"  Device {d.id:2d}{tag:7s}       {final[d.id]:>9.2f}%")

    print(f"\n  Honest avg : {havg:.2f}%")
    if byz_id: print(f"  Byzantine  : {final[byz_id]:.2f}%")

    honest_devs=[d for d in devices if not d.is_byzantine
                 and byz_id and byz_id in d.reputation]
    if honest_devs and byz_id:
        avg_rep=sum(d.reputation[byz_id] for d in honest_devs)/len(honest_devs)
        print(f"  D{byz_id} rep   : {avg_rep:.6f}")

    print(f"\n  Milestones [{AGGREGATION_METHOD.upper()} / {attack_tag.upper()}]:")
    for h in history:
        if h["round"] in [1,10,25,50,100,150,200]:
            print(f"  Rnd {h['round']:>3}: {h['avg_acc']:.2f}%  "
                  f"flags={h['byz_flags']}  lr={h['lr']:.5f}")

    if byz_id:
        print(f"\n  D{byz_id} reputation:")
        for h in history:
            if h["round"] in [1,5,10,20,50,100,150,200] and h["rep_snap"]:
                s=h["rep_snap"]; avg_r=sum(s.values())/len(s)
                print(f"  Rnd {h['round']:>3}: avg={avg_r:.4f} "
                      f"min={min(s.values()):.4f}")

    print(f"\n[DONE] {DATASET.upper()} | {AGGREGATION_METHOD.upper()} | "
          f"{attack_tag.upper()} | {NUM_ROUNDS} rounds")
    sys.stdout=sys.__stdout__
    return havg, history, _buf.getvalue()


# ══════════════════════════════════════════════════════════
# Run two experiments: honest baseline + Byzantine attack
# ══════════════════════════════════════════════════════════
if __name__=="__main__":
    label=f"{DATASET}_{AGGREGATION_METHOD}_{ATTACK}"

    print(f"\n{'='*60}")
    print(f"  Exp 1 — No Byzantine  [{AGGREGATION_METHOD.upper()}]")
    print(f"{'='*60}")
    acc1,hist1,out1=run_experiment(run_byzantine=False)
    with open(f"results_{label}_no_byz.txt","w") as f: f.write(out1)

    print(f"\n{'='*60}")
    print(f"  Exp 2 — Byzantine [{ATTACK.upper()}]  [{AGGREGATION_METHOD.upper()}]")
    print(f"{'='*60}")
    acc2,hist2,out2=run_experiment(run_byzantine=True)
    with open(f"results_{label}_byz.txt","w") as f: f.write(out2)

    print(f"\n{'='*60}")
    print(f"  {DATASET.upper()} | {AGGREGATION_METHOD.upper()} | {ATTACK.upper()}")
    print(f"  No Byzantine : {acc1:.2f}%")
    print(f"  Byzantine    : {acc2:.2f}%")
    print(f"  Drop         : {acc1-acc2:.2f}%")
    print(f"{'='*60}")