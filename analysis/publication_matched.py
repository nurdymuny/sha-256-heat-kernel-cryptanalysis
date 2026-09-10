"""Significance-matched control family for the early-round follow-up.

The fixed candidate occupies MSB-first offsets 1-14 of W0, the more significant
half of the word, and every candidate bit lies above the observed bit 16 of
register a. Uniform W0 controls mix both halves of the word. This runner
evaluates forty six-position subsets drawn only from offsets 0-14 on the same
saved follow-up bases and rounds, so candidate scores are identical to the
follow-up receipts and every comparison is paired on identical bases.
Existing evidence is never overwritten.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,platform,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.publication_study import dump,interval
from analysis.publication_followup import search_pattern_scores
from src.sha256.batch import cube_states

POOL=list(range(15))  # MSB-first offsets 0-14 of W0 = LSB-numbered bits 31-17
FOLLOWUP=ROOT/'publication/evidence/followup'
PROTOCOL={
 'version':'1.0',
 'context':'Second control family, added after review of the early-round follow-up. Not an independent preregistered study.',
 'control_pool':'six-position subsets of MSB-first offsets 0-14 of W0 (LSB-numbered bits 17-31); the candidate lies in this pool and every pooled bit is above the observed bit 16',
 'control_sets':40,'seed':264310,'target':[1,5,7,9,10,14],
 'bases':'the saved follow-up bases in publication/evidence/followup/search.npz and cubes.npz',
 'rounds':'the saved follow-up rounds: search [2,3,4,5,6,7,24]; cubes [2,3,4,5,6,7,8,10,12,14,16,18,20,24]',
 'statistics':'per-set means; paired candidate-minus-matched-mean percentile bootstrap over bases; count of matched sets below the candidate (search) or at or above it (cubes); whole-word zero fractions for registers a and e',
 'inference':'descriptive, unadjusted, conditional on the sampled sets; no equivalence claim',
 'selection':'protocol, seed and set count fixed before execution'}


def matched_sets(rng,count=40,pool=POOL,size=6):
    return [sorted(rng.choice(pool,size,replace=False).tolist()) for _ in range(count)]


def uniform_diagnostics():
    """Descriptive checks on the saved uniform-control follow-up receipts; no new sampling."""
    fs=np.load(FOLLOWUP/'search.npz');sj=json.loads((FOLLOWUP/'search.json').read_text())
    fq=np.load(FOLLOWUP/'cubes.npz');qj=json.loads((FOLLOWUP/'cubes.json').read_text())
    offsets=np.array([np.mean(c) for c in sj['controls']])
    search={str(int(r)):float(np.corrcoef(fs['scores'][1:,:,j].mean(1),offsets)[0,1]) for j,r in enumerate(fs['rounds'])}
    above=np.array([sum(1 for o in c if 31-o>16) for c in qj['controls']])
    rounds=[int(r) for r in fq['rounds']];j2=rounds.index(2)
    rates=fq['zero_indicators'][1:,:,j2].mean(1)
    return {'search_corr_control_mean_with_mean_offset':search,
            'cube_round2_control_zero_rates':rates.tolist(),
            'cube_round2_corr_zero_rate_with_bits_above_16':float(np.corrcoef(rates,above)[0,1]),
            'cube_round2_control_zero_rate_range':[float(rates.min()),float(rates.max())],
            'cube_round2_whole_word_a_zero':{'candidate':float((fq['word_sums'][0,:,j2,0]==0).mean()),'uniform_controls':float((fq['word_sums'][1:,:,j2,0]==0).mean())},
            'cube_round2_whole_word_e_zero':{'candidate':float((fq['word_sums'][0,:,j2,4]==0).mean()),'uniform_controls':float((fq['word_sums'][1:,:,j2,4]==0).mean())}}


def run(out):
    rng=np.random.default_rng(PROTOCOL['seed']);controls=matched_sets(rng,PROTOCOL['control_sets'])
    dump(out/'frozen_matched_sets.json',{'target':PROTOCOL['target'],'controls':controls,'pool':POOL})
    target=PROTOCOL['target']
    fs=np.load(FOLLOWUP/'search.npz');blocks=fs['blocks'];rounds=[int(r) for r in fs['rounds']]
    patterns=np.stack([search_pattern_scores(blocks,p,rounds) for p in [target]+controls]);scores=patterns.min(2)
    np.testing.assert_array_equal(scores[0],fs['scores'][0])
    np.savez_compressed(out/'search.npz',blocks=blocks,pattern_scores=patterns,scores=scores,rounds=np.array(rounds))
    records=[]
    for j,r in enumerate(rounds):
        cand=scores[0,:,j].astype(float);ctrl=scores[1:,:,j];set_means=ctrl.mean(1)
        records.append({'round':r,'candidate_mean':float(cand.mean()),'matched_mean':float(ctrl.mean()),
            'difference':interval(cand-ctrl.mean(0)),'set_means':set_means.tolist(),
            'matched_sets_below_candidate':int((set_means<cand.mean()).sum())})
    dump(out/'search.json',{'records':records,'target':target,'controls':controls,'n_bases':int(len(blocks))})
    print('Matched search complete',flush=True)
    fq=np.load(FOLLOWUP/'cubes.npz');blocks=fq['blocks'];rounds=[int(r) for r in fq['rounds']]
    words=[]
    for p in [target]+controls:
        sums=cube_states(blocks,p,rounds);words.append(np.stack([sums[r] for r in rounds],axis=1))
    words=np.stack(words);zero=((words[:,:,:,0]>>16)&1)==0
    np.testing.assert_array_equal(words[0],fq['word_sums'][0])
    np.savez_compressed(out/'cubes.npz',blocks=blocks,word_sums=words,zero_indicators=zero,rounds=np.array(rounds))
    records=[]
    for j,r in enumerate(rounds):
        cand=zero[0,:,j].astype(float);ctrl=zero[1:,:,j];rates=ctrl.mean(1)
        records.append({'round':r,'candidate_zero_rate':float(cand.mean()),'matched_zero_rate':float(ctrl.mean()),
            'difference':interval(cand-ctrl.mean(0)),'set_rates':rates.tolist(),
            'matched_range':[float(rates.min()),float(rates.max())],
            'matched_sets_at_or_above_candidate':int((rates>=cand.mean()).sum()),
            'whole_word_a_zero':{'candidate':float((words[0,:,j,0]==0).mean()),'matched':float((words[1:,:,j,0]==0).mean())},
            'whole_word_e_zero':{'candidate':float((words[0,:,j,4]==0).mean()),'matched':float((words[1:,:,j,4]==0).mean())}})
    dump(out/'cubes.json',{'records':records,'target':target,'controls':controls,'n_bases':int(len(blocks))})
    print('Matched cubes complete',flush=True)
    dump(out/'uniform_diagnostics.json',uniform_diagnostics())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output
    if out.exists():raise RuntimeError('Use a fresh evidence directory; existing output is never overwritten')
    out.mkdir(parents=True)
    dump(out/'protocol.json',PROTOCOL)
    sources=[Path(__file__),ROOT/'analysis/publication_followup.py',ROOT/'analysis/publication_study.py',ROOT/'src/sha256/batch.py',ROOT/'src/sha256/core.py']
    inputs=[FOLLOWUP/'search.npz',FOLLOWUP/'cubes.npz']
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    dump(out/'environment.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'numpy':np.__version__,
        'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sources},
        'input_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in inputs}})
    start=time.monotonic();run(out)
    dump(out/'completion.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':main()
