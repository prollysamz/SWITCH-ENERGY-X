"""Bounded reproducibility check using only supplied synthetic noise sensors.

A match must reproduce multiple observed values, not merely one near value.
No hidden target or external data is accessed; no submission is generated.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/generator_fingerprint_v1'

def main():
    if OUT.exists():raise FileExistsError('Preserve run')
    data=pd.read_csv(ROOT/'Dataset/train.csv',nrows=32)
    sensors={k:data[k].to_numpy() for k in ['feature_20','feature_25']}
    seeds=[0,1,7,42,123,2024,2025,2026,12345,20260908]
    matches=[];checked=[]
    for engine in ['default_rng','RandomState']:
        for seed in seeds:
            rng=np.random.default_rng(seed) if engine=='default_rng' else np.random.RandomState(seed)
            previous=np.empty(0)
            for block in range(20):
                chunk=rng.standard_normal(500000)
                values=np.r_[previous,chunk]
                offset=block*500000-len(previous)
                for name,truth in sensors.items():
                    finite=np.isfinite(truth);first=np.flatnonzero(finite)[0]
                    hits=np.flatnonzero(abs(values-truth[first])<1e-10)-first
                    for hit in hits:
                        if hit<0 or hit+len(truth)>len(values):continue
                        err=float(abs(values[hit:hit+len(truth)][finite]-truth[finite]).max())
                        if err<1e-9:
                            record=dict(engine=engine,seed=seed,normal_sequence_offset=int(offset+hit),sensor=name,
                                matched_observations=int(finite.sum()),maximum_error=err)
                            matches.append(record);print(json.dumps(record),flush=True)
                previous=values[-64:]
            checked.append(dict(engine=engine,seed=seed,draws=10000000))
            print(engine,seed,'checked',flush=True)
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(dict(matches=matches,checked=checked,
        limitations=['Failure only rules out these direct consecutive Gaussian sequences and tested seeds/offset ranges.',
            'Additional sensor noise, shuffling, another generator, or another seed can prevent matching.'],
        target_data_accessed=False,submission_generated=False),indent=2))

if __name__=='__main__':main()
