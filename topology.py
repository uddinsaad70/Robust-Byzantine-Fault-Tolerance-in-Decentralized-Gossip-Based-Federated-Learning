"""topology.py — passes dataset + attack to EdgeDevice"""
import networkx as nx
from device import EdgeDevice

def create_devices(dataset="mnist", attack="label_flip"):
    config = (
        [(i,"raspberry_pi",False) for i in range(1,7)] +
        [(i,"laptop",False)       for i in range(7,17)] +
        [(17,"desktop",False)] +
        [(i,"desktop",False)      for i in range(18,21)]
    )
    return [EdgeDevice(did,dt,byz,dataset=dataset,attack=attack)
            for did,dt,byz in config]

class TopologyManager:
    HARD_CAP=12
    def __init__(self,devices):
        self.devices=list(devices); self.did={d.id:d for d in self.devices}
        self.G=nx.Graph()
        for d in self.devices: self.G.add_node(d.id,resource_score=d.resource_score,k=d.k)
    def _greedy_assign(self):
        for d in sorted(self.devices,key=lambda x:x.resource_score,reverse=True):
            if self.G.degree(d.id)>=d.k: continue
            cands=[c for c in self.devices if c.id!=d.id
                   and not self.G.has_edge(d.id,c.id) and self.G.degree(c.id)<c.k]
            cands.sort(key=lambda c:c.k-self.G.degree(c.id),reverse=True)
            for c in cands:
                if self.G.degree(d.id)>=d.k: break
                self.G.add_edge(d.id,c.id)
    def _handle_saturation(self):
        for d in self.devices:
            while self.G.degree(d.id)<d.k:
                unc=[c for c in self.devices if c.id!=d.id and not self.G.has_edge(d.id,c.id)]
                if not unc: break
                avail=[c for c in unc if self.G.degree(c.id)<c.k]
                sat  =[c for c in unc if self.G.degree(c.id)>=c.k]
                if avail:
                    self.G.add_edge(d.id,max(avail,key=lambda c:c.k-self.G.degree(c.id)).id)
                elif sat:
                    best=max(sat,key=lambda c:c.resource_score*(1-self.G.degree(c.id)/self.HARD_CAP))
                    if self.G.degree(best.id)<self.HARD_CAP:
                        best.k=min(best.k+1,self.HARD_CAP); self.G.add_edge(d.id,best.id)
                    else: break
                else: break
    def _ensure_connectivity(self):
        comps=list(nx.connected_components(self.G))
        while len(comps)>1:
            a=max(comps[0],key=lambda n:self.did[n].resource_score)
            b=max(comps[1],key=lambda n:self.did[n].resource_score)
            self.G.add_edge(a,b); comps=list(nx.connected_components(self.G))
    def _sync(self):
        for d in self.devices:
            d.neighbors=list(self.G.neighbors(d.id))
            d.reputation={n:d.reputation.get(n,1.0) for n in d.neighbors}
    def build(self):
        self._greedy_assign(); self._handle_saturation()
        self._ensure_connectivity(); self._sync(); return self.G

def build_topology(devices):
    mgr=TopologyManager(devices); return mgr.build(), mgr