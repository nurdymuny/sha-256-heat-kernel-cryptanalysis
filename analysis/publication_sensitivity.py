"""Sensitivity of the small detour contrast to null generator and feed-forward."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from analysis.publication_study import dump,detour,interval,messages
from src.sha256.batch import padded_blocks,compress_blocks,state_bits

def main():
    out=ROOT/'publication/evidence/sensitivity';out.mkdir(parents=True,exist_ok=True)
    if (out/'detour.json').exists(): raise RuntimeError('Receipt exists')
    dump(out/'protocol.json',{'context':'Followup to negative detour contrast in two independent groups; no selection on this sample.',
        'replicates':20,'points':512,'k_nonself':10,'seed_start':262910,
        'datasets':['round64 working state','SHA256 digest','PCG64 bits','MT19937 bits']})
    rows=[]
    for j in range(20):
        seed=262910+j;rng=np.random.default_rng(seed);ms=messages(rng,512)
        state=state_bits(compress_blocks(padded_blocks(ms))[64])
        dig=np.unpackbits(np.frombuffer(b''.join(hashlib.sha256(m).digest() for m in ms),dtype=np.uint8).reshape(-1,32),axis=1)
        pcg=rng.integers(0,2,(512,256),dtype=np.uint8)
        mt=np.random.Generator(np.random.MT19937(seed)).integers(0,2,(512,256),dtype=np.uint8)
        rows.append({'seed':seed,'state':detour(state),'digest':detour(dig),'PCG64':detour(pcg),'MT19937':detour(mt)})
        print(f'Detour sensitivity {j+1}/20',flush=True)
    summary={a+' minus '+b:interval([r[a]-r[b] for r in rows]) for a,b in [('state','PCG64'),('state','MT19937'),('digest','PCG64'),('digest','MT19937'),('PCG64','MT19937')]}
    dump(out/'detour.json',{'records':rows,'summary':summary})

if __name__=='__main__':main()
