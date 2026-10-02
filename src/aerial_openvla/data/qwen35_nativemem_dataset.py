"""Cached pre-merger datasets for Qwen3.5-0.8B NativeMEM."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from aerial_openvla.data.manifest import load_step_records

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

class _EpisodeIndex:
    def __init__(self, records):
        self.by_episode: dict[str,dict[int,int]]={}; self.first: dict[str,int]={}
        for i,r in enumerate(records):
            steps=self.by_episode.setdefault(r.episode_id,{})
            if r.step_id in steps: raise ValueError(f'duplicate {r.episode_id}/{r.step_id}')
            steps[r.step_id]=i
        for episode,steps in self.by_episode.items(): self.first[episode]=steps[min(steps)]

class Qwen35NativeMEMDataset(Dataset[dict[str,Any]]):
    """Stage1 uses I0+7 sampled H16+It; Stage2 uses H16+It.

    Every RGB is resolved through manifest_to_cache.npy using the canonical
    (trajectory_id, step_id) visual-cache key.  No image file is opened.
    """
    def __init__(self, manifest_path: str|Path, cache_dir: str|Path, stage: int,
                 history_length: int=16, stage1_frames: int=9):
        self.manifest_path=Path(manifest_path).resolve(); self.cache_dir=Path(cache_dir).resolve()
        self.stage=int(stage); self.history_length=int(history_length); self.stage1_frames=int(stage1_frames)
        if self.stage not in (1,2): raise ValueError('NativeMEM stage must be 1 or 2')
        if self.history_length!=16 or self.stage1_frames!=9: raise ValueError('NativeMEM requires H16 and 9-frame Stage1')
        self.records=load_step_records(self.manifest_path); self.index=_EpisodeIndex(self.records)
        cache_meta=json.loads((self.cache_dir/'cache_meta.json').read_text()); index_meta=json.loads((self.cache_dir/'index_meta.json').read_text())
        manifest_hash=_sha256(self.manifest_path)
        if not cache_meta.get('complete') or not index_meta.get('complete'): raise ValueError('NativeMEM pre-merger cache/index incomplete')
        if cache_meta.get('manifest_sha256')!=manifest_hash or index_meta.get('manifest_sha256')!=manifest_hash: raise ValueError('NativeMEM cache manifest mismatch')
        if cache_meta.get('shape')!=[index_meta.get('cache_rows'),196,768]: raise ValueError('NativeMEM cache must be [N,196,768]')
        if index_meta.get('manifest_rows')!=len(self.records): raise ValueError('NativeMEM manifest mapping length mismatch')
        self.cache_meta=cache_meta; self._features=None; self._mapping=None
    @property
    def features(self):
        if self._features is None: self._features=np.load(self.cache_dir/'features.npy',mmap_mode='r')
        return self._features
    @property
    def mapping(self):
        if self._mapping is None: self._mapping=np.load(self.cache_dir/'manifest_to_cache.npy',mmap_mode='r')
        return self._mapping
    def __len__(self): return len(self.records)
    def _gather(self, manifest_rows: list[int|None]) -> torch.Tensor:
        out=np.zeros((len(manifest_rows),196,768),dtype=np.float16)
        positions=[i for i,row in enumerate(manifest_rows) if row is not None]
        if positions:
            cache_rows=np.asarray([self.mapping[int(manifest_rows[i])] for i in positions],dtype=np.int64)
            out[np.asarray(positions)]=self.features[cache_rows]
        return torch.from_numpy(out)
    def __getitem__(self,item: int) -> dict[str,Any]:
        r=self.records[item]; steps=self.index.by_episode[r.episode_id]
        if self.stage==1:
            candidates=[steps[s] for s in range(r.step_id-16,r.step_id) if s in steps]
            selected=sorted(random.sample(candidates,min(7,len(candidates))))
            pad=7-len(selected); rows=[self.index.first[r.episode_id]]+[None]*pad+selected+[item]
            valid=[True]+[False]*pad+[True]*len(selected)+[True]
        else:
            history=[steps.get(s) for s in range(r.step_id-self.history_length,r.step_id)]
            rows=history+[item]; valid=[row is not None for row in history]
        return {'nativemem_premerger_features':self._gather(rows),'nativemem_history_valid_mask':torch.tensor(valid,dtype=torch.bool),'instruction':r.instruction,'action_id':int(r.action_id),'episode_id':r.episode_id,'trajectory_id':r.trajectory_id,'step_id':r.step_id,'current_visual_cache_row':int(self.mapping[item])}
