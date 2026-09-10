"""Saved-protocol follow-ups: early interventions and 100 fresh detour pairs."""
from pathlib import Path
from itertools import combinations
from datetime import datetime,timezone
import argparse,hashlib,sys,time,platform
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.publication_study import dump,interval,holm,messages,detour
from src.sha256.batch import padded_blocks,compress_blocks,state_bits,flip_input_bits,cube_states

PROTOCOL={
 'version':'1.0','context':'Author follow-up after review of primary and confirmation data; not an independent preregistered study.',
 'detour':{'seed_start':263910,'pairs':100,'points':512,'k_nonself':10,'representations':['state','digest','PCG64','MT19937']},
 'search':{'seed':264110,'bases':256,'rounds':[2,3,4,5,6,7,24],'target':[1,5,7,9,10,14],'controls':20,'control_pool':'uniform six-position subsets of W0','patterns':'all 15 two-position flips; minimum Hamming distance over eight working words'},
 'cubes':{'seed':264210,'bases':512,'rounds':[2,3,4,5,6,7,8,10,12,14,16,18,20,24],'target':[1,5,7,9,10,14],'controls':20,'control_pool':'uniform six-position subsets of W0','observable':'zero XOR sum of bit 16 LSB-first of register a; full word sums saved'},
 'inference':{'bootstrap':5000,'resampling':'independent bases or whole dataset pairs; paired differences; control-set uncertainty conditional on sampled sets','intervals':'95 percent descriptive, unadjusted','cube_tests':'target-versus-one-half exact binomial; Holm across all fourteen follow-up rounds, a family distinct from primary eight tests','no_equivalence_claim':True},
 'selection':'Target inherited from prior exploratory work; all rounds, sample counts and seed ranges fixed before executing this follow-up.'}


def search_pattern_scores(blocks,positions,rounds):
    """Hamming counts, shape (bases, pairs, rounds), with no scale normalization."""
    original=compress_blocks(blocks,rounds);all_scores=[]
    for p,q in combinations(positions,2):
        modified=flip_input_bits(flip_input_bits(blocks,p),q)
        captures=compress_blocks(modified,rounds)
        all_scores.append(np.stack([state_bits(captures[r]^original[r]).sum(1) for r in rounds],axis=1))
    return np.stack(all_scores,axis=1).astype(np.uint16)


def run_early(out):
    for name in ['search','cubes']:
        cfg=PROTOCOL[name];rng=np.random.default_rng(cfg['seed'])
        controls=[sorted(rng.choice(32,6,replace=False).tolist()) for _ in range(cfg['controls'])]
        dump(out/f'frozen_{name}_sets.json',{'target':cfg['target'],'controls':controls})
        blocks=padded_blocks(messages(rng,cfg['bases']));rounds=cfg['rounds']
        if name=='search':
            patterns=np.stack([search_pattern_scores(blocks,p,rounds) for p in [cfg['target']]+controls])
            scores=patterns.min(2)
            np.savez_compressed(out/'search.npz',blocks=blocks,pattern_scores=patterns,scores=scores,rounds=rounds)
        else:
            words=[]
            for p in [cfg['target']]+controls:
                sums=cube_states(blocks,p,rounds)
                words.append(np.stack([sums[r] for r in rounds],axis=1))
            words=np.stack(words);scores=((words[:,:,:,0]>>16)&1)==0
            np.savez_compressed(out/'cubes.npz',blocks=blocks,word_sums=words,zero_indicators=scores,rounds=rounds)
        records=[]
        for j,r in enumerate(rounds):
            target=scores[0,:,j].astype(float);control=scores[1:,:,j].mean(0)
            row={'round':r,'target_mean':float(target.mean()),'control_mean':float(control.mean()),'difference':interval(target-control)}
            if name=='cubes':row['binomial_p']=float(binomtest(int(target.sum()),len(target),.5).pvalue)
            records.append(row)
        if name=='cubes':
            for row,p in zip(records,holm([r['binomial_p'] for r in records])):row['holm_p']=p
        dump(out/f'{name}.json',{'records':records,'target':cfg['target'],'controls':controls,'n_bases':cfg['bases']})
        print(f'Early {name} complete: {len(rounds)} rounds',flush=True)


def run_detour(out):
    cfg=PROTOCOL['detour'];rows=[]
    for j in range(cfg['pairs']):
        seed=cfg['seed_start']+j;rng=np.random.default_rng(seed);ms=messages(rng,cfg['points'])
        state=state_bits(compress_blocks(padded_blocks(ms))[64])
        digest=np.unpackbits(np.frombuffer(b''.join(hashlib.sha256(m).digest() for m in ms),dtype=np.uint8).reshape(-1,32),axis=1)
        pcg=rng.integers(0,2,state.shape,dtype=np.uint8)
        mt=np.random.Generator(np.random.MT19937(seed)).integers(0,2,state.shape,dtype=np.uint8)
        rows.append({'seed':seed,'state':detour(state),'digest':detour(digest),'PCG64':detour(pcg),'MT19937':detour(mt)})
        if (j+1)%10==0:print(f'Fresh detour {j+1}/{cfg["pairs"]}',flush=True)
    summary={a+' minus '+b:interval([r[a]-r[b] for r in rows]) for a,b in [('state','PCG64'),('state','MT19937'),('digest','PCG64'),('digest','MT19937'),('PCG64','MT19937')]}
    dump(out/'detour.json',{'records':rows,'summary':summary})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output
    if out.exists():raise RuntimeError('Use a fresh evidence directory; existing output is never overwritten')
    out.mkdir(parents=True)
    dump(out/'protocol.json',PROTOCOL)
    sources=[Path(__file__),ROOT/'analysis/publication_study.py',ROOT/'src/sha256/batch.py',ROOT/'src/sha256/core.py']
    dump(out/'environment.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'numpy':np.__version__,'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    start=time.monotonic();run_early(out);run_detour(out)
    dump(out/'completion.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':main()
