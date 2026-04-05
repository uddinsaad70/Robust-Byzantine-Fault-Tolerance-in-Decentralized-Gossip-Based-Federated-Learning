"""
device.py  [Thesis version — Task 1 + Task 3]
=============================================
Task 1: Two Byzantine attacks
    "label_flip" — Tolpegin et al. (ESORICS 2020)
    "alie"       — Baruch et al. (CCS 2019)

Task 3: optimizer.step() removed from honest training.
    Pure FL math: local_train() only returns gradient.
    Model update happens only in aggregation.py via W -= lr * grad.
"""

import torch
import torch.nn as nn
import numpy as np
import random

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE_RANGES = {
    "raspberry_pi": {"cpu_cores": (2,4),  "cpu_freq_ghz": (1.0,1.5), "ram_gb": (2,4),   "bandwidth_mbps": (5,20)},
    "laptop":       {"cpu_cores": (4,8),  "cpu_freq_ghz": (1.8,3.0), "ram_gb": (8,16),  "bandwidth_mbps": (30,80)},
    "desktop":      {"cpu_cores": (8,16), "cpu_freq_ghz": (3.0,4.0), "ram_gb": (16,32), "bandwidth_mbps": (80,200)},
}

def sample_resources(device_type, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    r = DEVICE_RANGES[device_type]
    step = r["cpu_cores"][1] - r["cpu_cores"][0] or 1
    return {
        "cpu_cores":      random.choice(range(r["cpu_cores"][0], r["cpu_cores"][1]+1, step)),
        "cpu_freq_ghz":   round(random.uniform(*r["cpu_freq_ghz"]), 2),
        "ram_gb":         random.choice([r["ram_gb"][0], r["ram_gb"][1]]),
        "bandwidth_mbps": round(random.uniform(*r["bandwidth_mbps"]), 1),
    }


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,3,padding=1); self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2); self.relu=nn.ReLU(); self.flatten=nn.Flatten()
        self.fc1=nn.Linear(64*7*7,128); self.fc2=nn.Linear(128,num_classes)
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x))); x=self.pool(self.relu(self.conv2(x)))
        return self.fc2(self.relu(self.fc1(self.flatten(x))))


class SimpleMLPHAR(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(561,128),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(128,64), nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(64,num_classes))
        for m in self.net:
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x): return self.net(x)


MODEL_CONFIG = {
    "mnist":  {"class":SimpleCNN,    "num_classes":10, "model_size":421_642, "max_label":9},
    "fmnist": {"class":SimpleCNN,    "num_classes":10, "model_size":421_642, "max_label":9},
    "har":    {"class":SimpleMLPHAR, "num_classes":6,  "model_size":80_582,  "max_label":5},
}
def get_model_size(dataset): return MODEL_CONFIG[dataset]["model_size"]
def build_model(dataset):
    c = MODEL_CONFIG[dataset]
    return c["class"](num_classes=c["num_classes"]).to(TORCH_DEVICE)


class EdgeDevice:
    def __init__(self, device_id, device_type, is_byzantine=False,
                 seed=None, dataset="mnist", attack="label_flip"):
        self.id=device_id; self.device_type=device_type
        self.is_byzantine=is_byzantine; self.dataset=dataset.lower()
        self.attack=attack
        self.resources=sample_resources(device_type, seed=seed if seed else device_id)
        self.resource_score=self._compute_resource_score()
        self.k=self._compute_k()
        self.neighbors=[]; self.reputation={}
        self.local_data=None; self.local_gradient=None
        self._max_label=MODEL_CONFIG[self.dataset]["max_label"]
        self.model=build_model(self.dataset)
        self.criterion=nn.CrossEntropyLoss()
        # optimizer used only for gradient computation — never stepped
        self.optimizer=torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self._velocity = {}          # persistent momentum buffer, survives rounds
        self._momentum_beta = 0.9    # Nesterov/Heavy-ball momentum coefficient

    def _compute_resource_score(self):
        r=self.resources
        return round(0.4*(r["cpu_cores"]*r["cpu_freq_ghz"]/10)+
                     0.4*(r["ram_gb"]/32)+0.2*(r["bandwidth_mbps"]/100),4)

    def _compute_k(self, k_min=3, k_max=10):
        return int(np.clip(k_min+int(self.resource_score*(k_max-k_min)),k_min,k_max))

    def init_reputation(self): self.reputation={}

    def local_train(self, batch_size=64):
        if self.local_data is None:
            raise ValueError(f"Device {self.id}: no data!")
        X,y=self.local_data
        X=X.to(TORCH_DEVICE); y=y.to(TORCH_DEVICE)
        g = self._byzantine_gradient(X,y,batch_size) if self.is_byzantine \
            else self._honest_gradient(X,y,batch_size)
        self.local_gradient=g
        return g

    # Fix 1: persistent local momentum.
    # Math:
    #     g_t  = stochastic gradient at round t  (1 batch)
    #     v_t  = β · v_{t-1} + g_t               (velocity accumulates)
    #     share g_t with neighbors (unchanged — DP/compression on raw grad)
    #     apply v_t locally:  W ← W - lr · v_t   (done in aggregation.py)

    # Why this works:
    #     v_t is an exponential moving average of past gradients.
    #     Even if g_t is noisy, v_t smooths out the noise across rounds.
    #     The effective gradient becomes:
    #         v_t = sum_{k=0}^{t} β^k · g_{t-k}
    #     This is equivalent to using ~1/(1-β) = 10 effective samples
    #     instead of 1, without any extra communication.

    # What we share vs apply:
    #     Neighbors receive g_t (raw batch gradient, as before).
    #     Local model is updated with v_t (smoothed).
    #     This is intentional: sharing v_t would leak history across devices.
    def _honest_gradient(self, X, y, batch_size):
        self.model.train()
        self.optimizer.zero_grad()
        idx = torch.randperm(len(X))[:batch_size]
        self.criterion(self.model(X[idx]), y[idx]).backward()

        raw_grad = {n: p.grad.clone()
                    for n, p in self.model.named_parameters() if p.grad is not None}

        # Update persistent velocity: v_t = β·v_{t-1} + g_t
        for name, g in raw_grad.items():
            if name not in self._velocity:
                self._velocity[name] = torch.zeros_like(g)
            self._velocity[name] = self._momentum_beta * self._velocity[name] + g

        # Store velocity on device so aggregation.py can use it instead of raw grad
        self.local_velocity = self._velocity  # aggregation reads this

        # Return raw gradient for sharing (Phase 2 → 3 → 4 pipeline unchanged)
        self.local_gradient = raw_grad
        return raw_grad

    def _byzantine_gradient(self, X, y, batch_size):
        if self.attack=="none": return self._honest_gradient(X,y,batch_size)
        if self.attack=="label_flip": return self._attack_label_flip(X,y,batch_size)
        if self.attack=="alie":       return self._attack_alie(X,y,batch_size)
        raise ValueError(f"Unknown attack: {self.attack}")

    def _attack_label_flip(self, X, y, batch_size):
        """
        Task 1-A: Label Flipping — Tolpegin et al. (ESORICS 2020)
        Flip labels: y = max_label - y  (e.g. MNIST: 0↔9, 1↔8, ...)
        Gradient looks honest but pushes model in wrong direction.
        Harder to detect in Non-IID because honest gradients also diverge.
        """
        self.model.train(); self.optimizer.zero_grad()
        idx=torch.randperm(len(X))[:batch_size]
        y_flip=self._max_label - y[idx]
        self.criterion(self.model(X[idx]), y_flip).backward()
        return {n:p.grad.clone()
                for n,p in self.model.named_parameters() if p.grad is not None}

    def _attack_alie(self, X, y, batch_size):
        """
        Task 1-B: ALIE (A Little Is Enough) — Baruch et al. (CCS 2019)
        Compute honest gradient, then add layer-wise Gaussian noise
        scaled to z_max * std(layer). z_max chosen to stay just below
        detection thresholds (Baruch et al. Eq. 3, n=20, f=1 → z≈1.5).
        Subtle enough to bypass cos_sim and mag_ratio tests while
        gradually degrading convergence.
        """
        self.model.train(); self.optimizer.zero_grad()
        idx=torch.randperm(len(X))[:batch_size]
        self.criterion(self.model(X[idx]), y[idx]).backward()
        z_max=1.5   # Baruch et al. Table 1: n=20, f=1
        result={}
        for n,p in self.model.named_parameters():
            if p.grad is None: continue
            g=p.grad.clone()
            std=max(g.std().item(), 1e-6)
            result[n]=g + torch.randn_like(g)*(z_max*std)
        return result

    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_test=X_test.to(TORCH_DEVICE); y_test=y_test.to(TORCH_DEVICE)
        with torch.no_grad():
            correct=(self.model(X_test).argmax(1)==y_test).sum().item()
        return round(100.0*correct/len(y_test),2)

    def __repr__(self):
        tag=f" [BYZ/{self.attack.upper()}]" if self.is_byzantine else ""
        return f"Device {self.id}{tag} ({self.device_type}): score={self.resource_score} k={self.k}"