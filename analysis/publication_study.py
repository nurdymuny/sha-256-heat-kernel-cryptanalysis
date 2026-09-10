"""Controlled SHA-256 measurements. All publication numbers derive from this run.

No old receipts are overwritten. Graph time is not compression-round time.
Protocol is saved before computation; selection sets are frozen before holdout.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
import numpy as np
import scipy
import sklearn
from scipy import sparse, stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.sha256.batch import padded_blocks,compress_blocks,state_bits,flip_input_bits,cube_states
from src.analysis.laplacian import LaplacianConstructor
from src.embeddings.euclidean import EuclideanEmbedding
from src.analysis.anomaly import AnomalyDetector,NullDistribution
from src.analysis.heat_kernel import HeatKernelSolver

PROTOCOL={
    'version':'1.0','seed':260910,
    'primitive':'SHA-256; raw working-state measurements are explicitly distinguished from final digests',
    'coordinates':'MSB-first input bit; W index = bit//32; round is completed compression rounds',
    'graphs':{'replicates':20,'points':512,'k':30,'kernel_times':[1.0,8.0],'diffusion_times':[.01,.1,1.,10.],
              'clusters':5,'spectral_coordinates':5,'spectrum':'full','state':'working registers after 64 rounds before feed-forward'},
    'avalanche':{'input_bits':440,'discovery_bases':128,'holdout_bases':128,'rounds':list(range(1,25))+[32,48,64],
                 'selection':'bottom 22 input positions by raw mean Hamming distance at round 24 in discovery only',
                 'estimand':'holdout selected minus other positions in message word W0, and across all message bits; bases are replication units'},
    'search':{'discovery_bases':128,'holdout_bases':256,'round':24,'target_positions':[1,5,7,9,10,14],
              'control_sets':20,'metric':'minimum raw Hamming distance over the 15 two-bit patterns; equal budgets'},
    'cubes':{'dimension':6,'bases':512,'target_positions':[1,5,7,9,10,14],
             'control_sets':20,'control_pool':'six distinct bits uniformly drawn from W0, equal injection time',
             'rounds':[8,10,12,14,16,18,20,24],
             'observable':'XOR sum of bit 16 counted LSB-first in register a; zero rate',
             'entry_control':{'dimension':3,'positions':[0,1,2],'words':[0,4,8,12,15],'bases':256,'relative_rounds':[0,1,2,3,4,5,6,8]}},
    'ml':{'replicates':5,'train_per_class':2000,'test_per_class':2000,
          'models':['RF:100 trees, max_depth=10','PCA:50 train-only components + RBF SVM C=1 gamma=scale'],
          'observable':'actual hashlib SHA-256 digests of random 55-byte messages vs independent PCG64 bit strings'},
    'inference':{'bootstrap_replicates':5000,'level':.95,
                 'graph':'paired mean SHA-minus-random differences over independently seeded dataset pairs; interval descriptive, not multiplicity adjusted',
                 'cube':'paired base-level difference, target versus average of 20 equal-entry random sets; percentile bootstrap; Holm adjustment over eight round-wise exact target binomial tests',
                 'negative_results':'absence of evidence for these observables, not security or general indistinguishability'},
}

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def interval(values,seed=915):
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(5000,len(x)))].mean(axis=1)
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n_units':len(x)}

def holm(pvalues):
    p=np.asarray(pvalues); order=np.argsort(p); out=np.empty(len(p)); prev=0.
    for j,i in enumerate(order):
        prev=max(prev,min(1.,float(p[i])*(len(p)-j))); out[i]=prev
    return out.tolist()

def source_hashes():
    files=list((ROOT/'src').rglob('*.py'))+[Path(__file__)]
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def messages(rng,n):
    return [rng.bytes(55) for _ in range(n)]

def graph_geometry(bits,h):
    c=LaplacianConstructor(EuclideanEmbedding(),k_neighbors=30,diffusion_time=h)
    result=c.build_from_points(bits.astype(float))
    vals,vecs=np.linalg.eigh(result.laplacian.toarray())
    d=np.asarray(result.weight_matrix.sum(axis=1)).ravel()
    residual=np.linalg.norm(result.laplacian@np.sqrt(d))/np.linalg.norm(np.sqrt(d))
    assert abs(vals[0])<1e-10 and residual<1e-10
    coords=vecs[:,1:6]
    labels=KMeans(n_clusters=5,n_init=10,random_state=42).fit_predict(coords)
    return {'silhouette':float(silhouette_score(coords,labels)),
            'cluster_sizes':np.bincount(labels,minlength=5).tolist(),
            'spectral_gap':float(vals[1]),'eigenvalues':vals.tolist(),
            'heat_trace':[float(np.exp(-t*vals).sum()) for t in PROTOCOL['graphs']['diffusion_times']],
            'zero_mode_residual':float(residual),'labels':labels.tolist()}

def detour(bits):
    d=squareform(pdist(bits.astype(float)))
    # 10 nonself neighbors, undirected union, edge weights = Euclidean distance.
    np.fill_diagonal(d,np.inf)
    idx=np.argsort(d,axis=1)[:,:10]
    rows=np.repeat(np.arange(len(d)),10)
    w=sparse.csr_matrix((d[rows,idx.ravel()],(rows,idx.ravel())),shape=d.shape)
    w=w.maximum(w.T)
    sp=shortest_path(w,directed=False)
    mask=np.triu(np.ones(d.shape,dtype=bool),1)&np.isfinite(sp)
    return float(np.mean(sp[mask]/d[mask]))

def run_graphs(out):
    records=[]
    for seed in range(PROTOCOL['graphs']['replicates']):
        rng=np.random.default_rng(PROTOCOL['seed']+seed)
        blocks=padded_blocks(messages(rng,512))
        sha=state_bits(compress_blocks(blocks)[64])
        random=rng.integers(0,2,sha.shape,dtype=np.uint8)
        record={'seed':PROTOCOL['seed']+seed,'sha_detour':detour(sha),'random_detour':detour(random),'kernels':{}}
        for h in PROTOCOL['graphs']['kernel_times']:
            record['kernels'][str(h)]={'sha':graph_geometry(sha,h),'random':graph_geometry(random,h)}
        records.append(record)
        print(f'Graph replicate {seed+1}/20 complete',flush=True)
    summaries={}
    for h in PROTOCOL['graphs']['kernel_times']:
        key=str(h); rows=[r['kernels'][key] for r in records]
        summaries[key]={}
        for metric in ['silhouette','spectral_gap']:
            summaries[key][metric]={'sha_mean':float(np.mean([r['sha'][metric] for r in rows])),
                'random_mean':float(np.mean([r['random'][metric] for r in rows])),
                'difference':interval([r['sha'][metric]-r['random'][metric] for r in rows])}
    summaries['detour']={'sha_mean':float(np.mean([r['sha_detour'] for r in records])),
        'random_mean':float(np.mean([r['random_detour'] for r in records])),
        'difference':interval([r['sha_detour']-r['random_detour'] for r in records])}
    result={'records':records,'summary':summaries}
    dump(out/'graphs.json',result); return result

def avalanche_group(rng,n,rounds):
    # Independent bases; all 440 one-bit interventions share each base.
    bases=padded_blocks(messages(rng,n)); original=compress_blocks(bases,rounds)
    data=np.empty((n,440,len(rounds)),dtype=np.uint16)
    for start in range(0,440,40):
        pos=np.arange(start,min(start+40,440))
        expanded=np.repeat(bases,len(pos),axis=0)
        modified=flip_input_bits(expanded,np.tile(pos,n))
        states=compress_blocks(modified,rounds)
        for j,r in enumerate(rounds):
            ref=np.repeat(original[r],len(pos),axis=0)
            data[:,pos,j]=state_bits(states[r]^ref).sum(axis=1).reshape(n,len(pos))
    return bases,data

def run_avalanche(out):
    rounds=PROTOCOL['avalanche']['rounds']; target=rounds.index(24)
    rng=np.random.default_rng(PROTOCOL['seed']+100)
    bases_d,discovery=avalanche_group(rng,128,rounds)
    selected=np.argsort(discovery[:,:,target].mean(axis=0),kind='stable')[:22]
    # Persist selection before evaluating the independent holdout.
    dump(out/'frozen_avalanche_selection.json',{'selected':selected.tolist(),'round':24,'seed':PROTOCOL['seed']+100})
    bases_h,holdout=avalanche_group(rng,128,rounds)
    rest=np.setdiff1d(np.arange(440),selected)
    differences=holdout[:,selected,target].mean(axis=1)-holdout[:,rest,target].mean(axis=1)
    w0selected=selected[selected<32]; w0rest=np.setdiff1d(np.arange(32),w0selected)
    np.savez_compressed(out/'avalanche.npz',discovery=discovery,holdout=holdout,discovery_blocks=bases_d,
        holdout_blocks=bases_h,rounds=np.array(rounds),selected=selected)
    result={'selected':selected.tolist(),'selected_message_words':(selected//32).tolist(),
            'holdout_selected_minus_others':interval(differences),
            'holdout_word0_selected_minus_others':interval(holdout[:,w0selected,target].mean(axis=1)-holdout[:,w0rest,target].mean(axis=1)) if len(w0selected) and len(w0rest) else None,
            'discovery_mean_by_bit_round':discovery.mean(axis=0).tolist(),'holdout_mean_by_bit_round':holdout.mean(axis=0).tolist(),
            'round64_mean_hamming':float(holdout[:,:,-1].mean()),'rounds':rounds,
            'bases_per_partition':128,'pairs_per_partition':128*440}
    dump(out/'avalanche.json',result); print('Discovery and holdout avalanche complete',flush=True); return result

def search_scores(blocks,positions):
    from itertools import combinations
    raw=compress_blocks(blocks,[24])[24]
    scores=[]
    for p,q in combinations(positions,2):
        modified=flip_input_bits(flip_input_bits(blocks,p),q)
        score=state_bits(compress_blocks(modified,[24])[24]^raw).sum(axis=1)
        scores.append(score)
    return np.min(scores,axis=0)

def run_search(out):
    rng=np.random.default_rng(PROTOCOL['seed']+200)
    target=PROTOCOL['search']['target_positions']
    controls=[sorted(rng.choice(32,6,replace=False).tolist()) for _ in range(20)]
    dump(out/'frozen_search_sets.json',{'target':target,'controls':controls})
    bases=padded_blocks(messages(rng,256))
    target_scores=search_scores(bases,target)
    control_scores=np.array([search_scores(bases,p) for p in controls])
    np.savez_compressed(out/'search.npz',blocks=bases,target=target_scores,controls=control_scores)
    result={'target_mean_min_hamming':float(target_scores.mean()),'control_mean_min_hamming':float(control_scores.mean()),
        'target_minus_mean_control':interval(target_scores-control_scores.mean(axis=0)),
        'n_bases':256,'control_sets':controls,'target':target}
    dump(out/'search.json',result); print('Matched two-bit search complete',flush=True); return result

def run_cubes(out):
    rng=np.random.default_rng(PROTOCOL['seed']+300)
    rounds=PROTOCOL['cubes']['rounds']; target=PROTOCOL['cubes']['target_positions']
    controls=[sorted(rng.choice(32,6,replace=False).tolist()) for _ in range(20)]
    dump(out/'frozen_cube_sets.json',{'target':target,'controls':controls})
    bases=padded_blocks(messages(rng,512)); scores=[]
    for j,positions in enumerate([target]+controls):
        sums=cube_states(bases,positions,rounds)
        scores.append(np.array([((sums[r][:,0]>>16)&1)==0 for r in rounds]).T)
        print(f'Cube set {j+1}/21 complete',flush=True)
    scores=np.array(scores); records=[]
    for j,r in enumerate(rounds):
        t=scores[0,:,j].astype(float); c=scores[1:,:,j].mean(axis=0)
        records.append({'round':r,'target_zero_rate':float(t.mean()),'control_zero_rate':float(c.mean()),
            'difference':interval(t-c),'target_binomial_p':float(stats.binomtest(int(t.sum()),len(t),.5).pvalue)})
    for row,p in zip(records,holm([row['target_binomial_p'] for row in records])): row['holm_p']=p
    np.savez_compressed(out/'cubes.npz',blocks=bases,zero_indicators=scores,rounds=rounds)
    # Matched word shifts: use the same bases and offsets; define delta=0 at injection.
    cfg=PROTOCOL['cubes']['entry_control']; entry_bases=padded_blocks(messages(rng,cfg['bases']))
    entry=[]
    for w in cfg['words']:
        rr=[w+1+d for d in cfg['relative_rounds']]
        sums=cube_states(entry_bases,[w*32+p for p in cfg['positions']],rr)
        rates={str(r):np.mean(sums[r]==0,axis=0).tolist() for r in rr}
        entry.append({'message_word':w,'rounds':rr,'whole_word_zero_rates':rates})
    result={'records':records,'target':target,'controls':controls,'entry_control':entry,
            'base_count':512,'corner_evaluations_main':21*512*64}
    dump(out/'cubes.json',result); return result

def run_ml(out):
    records=[]
    for j in range(5):
        rng=np.random.default_rng(PROTOCOL['seed']+400+j)
        def dataset(n):
            digests=b''.join(hashlib.sha256(m).digest() for m in messages(rng,n))
            sha=np.unpackbits(np.frombuffer(digests,dtype=np.uint8).reshape(n,32),axis=1)
            random=rng.integers(0,2,(n,256),dtype=np.uint8)
            x=np.vstack([sha,random]); y=np.repeat([1,0],n)
            idx=rng.permutation(2*n); return x[idx],y[idx]
        xtr,ytr=dataset(2000); xte,yte=dataset(2000)
        models={'RF':RandomForestClassifier(n_estimators=100,max_depth=10,random_state=j,n_jobs=2),
                'PCA_SVM':make_pipeline(PCA(n_components=50,svd_solver='full'),SVC(C=1,kernel='rbf',gamma='scale'))}
        row={'seed':PROTOCOL['seed']+400+j,'test_n':len(yte),'models':{}}
        arrays={'x_train':xtr,'y_train':ytr,'x_test':xte,'y_test':yte}
        for name,model in models.items():
            model.fit(xtr,ytr); pred=model.predict(xte)
            scores=model.predict_proba(xte)[:,1] if name=='RF' else model.decision_function(xte)
            row['models'][name]={'correct':int(np.sum(pred==yte)),'accuracy':float(accuracy_score(yte,pred)),
                'auc':float(roc_auc_score(yte,scores))}
            arrays[name+'_prediction']=pred; arrays[name+'_score']=scores
        np.savez_compressed(out/f'ml_{j}.npz',**arrays)
        records.append(row); print(f'ML replication {j+1}/5 complete',flush=True)
    summary={name:{'mean_accuracy':float(np.mean([r['models'][name]['accuracy'] for r in records])),
        'range':[float(min(r['models'][name]['accuracy'] for r in records)),float(max(r['models'][name]['accuracy'] for r in records))],
        'mean_auc':float(np.mean([r['models'][name]['auc'] for r in records]))} for name in ['RF','PCA_SVM']}
    result={'records':records,'summary':summary}; dump(out/'ml.json',result); return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=ROOT/'publication/evidence')
    parser.add_argument('--sections',nargs='+',default=['graphs','avalanche','search','cubes','ml'])
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=PROTOCOL: raise RuntimeError('Existing protocol differs; use a new output directory')
    else: dump(out/'protocol.json',PROTOCOL)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'sklearn':sklearn.__version__,'started_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':source_hashes()})
    functions={'graphs':run_graphs,'avalanche':run_avalanche,'search':run_search,'cubes':run_cubes,'ml':run_ml}
    start=time.monotonic()
    for name in args.sections:
        if (out/f'{name}.json').exists(): raise RuntimeError(f'Receipt exists: {name}; choose a fresh output directory')
        functions[name](out)
    dump(out/'completion.json',{'sections':args.sections,'elapsed_seconds':time.monotonic()-start,'finished_utc':datetime.now(timezone.utc).isoformat()})
    print('Requested experiment sections complete.',flush=True)

if __name__=='__main__': main()
