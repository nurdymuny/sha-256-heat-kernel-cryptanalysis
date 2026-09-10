"""Fresh graph confirmation and uniformly sampled raw-block entry control.

The graph followup was specified after the initial twenty dataset pairs:
confirm h=8 silhouette and k=10 detour, while reporting all original metrics.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from analysis import publication_study as study
from src.sha256.batch import cube_states

def main():
    out=ROOT/'publication/evidence/confirmation'; out.mkdir(parents=True,exist_ok=True)
    if (out/'graphs.json').exists(): raise RuntimeError('Confirmation receipt already exists')
    cfg={'selection_context':'Followup after initial graph results; h=8 silhouette and detour had negative descriptive intervals.',
        'primary_metrics':['h=8 silhouette','k=10 detour'],'graph_pairs':20,'graph_seed_start':261910,
        'entry_bases':256,'entry_distribution':'uniform 64-byte raw blocks, no padding interpretation',
        'entry_seed':261410,'entry_words':[0,4,8,12,15],'entry_offsets':[0,1,2],
        'entry_delta':[0,1,2,3,4,5,6,8]}
    study.dump(out/'protocol.json',cfg)
    rng=np.random.default_rng(cfg['entry_seed']); bases=rng.integers(0,256,(256,64),dtype=np.uint8)
    records=[]; arrays={'blocks':bases}
    for w in cfg['entry_words']:
        rr=[w+1+d for d in cfg['entry_delta']]
        sums=cube_states(bases,[32*w+p for p in cfg['entry_offsets']],rr)
        records.append({'message_word':w,'rounds':rr,'whole_word_zero_rates':{str(r):(sums[r]==0).mean(axis=0).tolist() for r in rr}})
        for r in rr: arrays[f'w{w}_r{r}']=sums[r]
    np.savez_compressed(out/'entry.npz',**arrays)
    study.dump(out/'entry.json',{'records':records,'base_count':256,'protocol':cfg})
    study.PROTOCOL['seed']=cfg['graph_seed_start']
    study.run_graphs(out)
    study.dump(out/'completion.json',{'completed':True,'source_hashes':study.source_hashes()})

if __name__=='__main__': main()
