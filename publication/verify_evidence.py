"""Cross-foot publication receipts independently of figure rendering."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.publication_study import interval,holm
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
E=ROOT/'publication/evidence'
def read(p):return json.loads((E/p).read_text())
def same(x,y):np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-10)
def check_effect_mean(values, reported):
    np.testing.assert_allclose(np.asarray(values, dtype=float).mean(), reported,
                               rtol=1e-10, atol=1e-10,
                               err_msg="reported effect mean differs from per-unit values")
checks=[]
for path in ['graphs.json','confirmation/graphs.json']:
    data=read(path)
    for r in data['records']:
        for h,row in r['kernels'].items():
            for label,v in row.items():
                vals=np.array(v['eigenvalues'])
                assert len(vals)==512 and abs(vals[0])<1e-10 and np.all(np.diff(vals)>=-1e-10)
                assert sum(v['cluster_sizes'])==512 and len(v['labels'])==512
                assert v['zero_mode_residual']<1e-10
                same(v['heat_trace'],[np.exp(-t*vals).sum() for t in [.01,.1,1.,10.]])
    for h in ['1.0','8.0']:
        for metric in ['silhouette','spectral_gap']:
            dif=[r['kernels'][h]['sha'][metric]-r['kernels'][h]['random'][metric] for r in data['records']]
            expected=interval(dif)
            same(expected['ci95'],data['summary'][h][metric]['difference']['ci95'])
    checks.append(path+': full spectra, partitions, heat traces and intervals')
a=np.load(E/'avalanche.npz');aj=read('avalanche.json');selected=a['selected'];n=128
assert a['holdout'].shape==(128,440,27) and a['discovery'].shape==(128,440,27)
np.testing.assert_array_equal(selected,np.argsort(a['discovery'][:,:,23].mean(axis=0),kind='stable')[:22])
rest=np.setdiff1d(np.arange(440),selected)
dif=a['holdout'][:,selected,23].mean(axis=1)-a['holdout'][:,rest,23].mean(axis=1)
same(interval(dif)['ci95'],aj['holdout_selected_minus_others']['ci95']);same(a['holdout'][:,:,-1].mean(),aj['round64_mean_hamming'])
checks.append('avalanche: frozen discovery selection and held-out effect')
s=np.load(E/'search.npz');sj=read('search.json');same(interval(s['target']-s['controls'].mean(axis=0))['ci95'],sj['target_minus_mean_control']['ci95'])
checks.append('search: equal-budget saved per-base results')
q=np.load(E/'cubes.npz');qj=read('cubes.json');pv=[]
for j,row in enumerate(qj['records']):
    t=q['zero_indicators'][0,:,j];c=q['zero_indicators'][1:,:,j].mean(axis=0)
    same(t.mean(),row['target_zero_rate']);same(c.mean(),row['control_zero_rate'])
    check_effect_mean(t.astype(float)-c, row['difference']['mean'])
    same(interval(t.astype(float)-c)['ci95'],row['difference']['ci95'])
    pv.append(binomtest(int(t.sum()),512,.5).pvalue)
same(holm(pv),[r['holm_p'] for r in qj['records']]);checks.append('cubes: indicators, intervals and Holm adjustment')
entry=np.load(E/'confirmation/entry.npz');ej=read('confirmation/entry.json')
for row in ej['records']:
    w=row['message_word']; rounds=row['rounds']
    for r in rounds:
        v=entry[f'w{w}_r{r}'];same((v==0).mean(axis=0),row['whole_word_zero_rates'][str(r)])
        for delay,dst,src in [(1,1,0),(2,2,0),(3,3,0),(1,5,4),(2,6,4),(3,7,4)]:
            if r-delay in rounds:np.testing.assert_array_equal(v[:,dst],entry[f'w{w}_r{r-delay}'][:,src])
checks.append('entry control: whole-word receipts and exact register-transport identities')
ml=read('ml.json')
for i,row in enumerate(ml['records']):
    arrays=np.load(E/f'ml_{i}.npz')
    for name,v in row['models'].items():
        same((arrays[name+'_prediction']==arrays['y_test']).mean(),v['accuracy'])
        same(roc_auc_score(arrays['y_test'],arrays[name+'_score']),v['auc'])
checks.append('ML: held-out predictions, labels, accuracy and AUC')
source_base=ROOT/'publication/repro_source'
if not source_base.exists():source_base=ROOT
for p,h in read('environment.json')['source_hashes'].items():
    assert hashlib.sha256((source_base/p).read_bytes()).hexdigest()==h,p
checks.append('main-run source snapshot: every recorded hash matches')
follow=read('followup/protocol.json')
for p,digest in read('followup/environment.json')['source_hashes'].items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
checks.append('follow-up protocol and executed source hashes')
fs=np.load(E/'followup/search.npz');sj=read('followup/search.json')
np.testing.assert_array_equal(fs['scores'],fs['pattern_scores'].min(2))
assert fs['pattern_scores'].shape==(21,256,15,7)
for j,row in enumerate(sj['records']):
    values=fs['scores'][:,:,j];d=values[0].astype(float)-values[1:].mean(0)
    same(values[0].mean(),row['target_mean']);same(values[1:].mean(),row['control_mean'])
    check_effect_mean(d, row['difference']['mean'])
    same(interval(d)['ci95'],row['difference']['ci95'])
checks.append('follow-up search: all 15 patterns, minima and paired intervals')
fq=np.load(E/'followup/cubes.npz');qj=read('followup/cubes.json');pv=[]
np.testing.assert_array_equal(fq['zero_indicators'],((fq['word_sums'][:,:,:,0]>>16)&1)==0)
assert fq['word_sums'].shape==(21,512,14,8)
for j,row in enumerate(qj['records']):
    values=fq['zero_indicators'][:,:,j];d=values[0].astype(float)-values[1:].mean(0)
    same(values[0].mean(),row['target_mean']);same(values[1:].mean(),row['control_mean'])
    check_effect_mean(d, row['difference']['mean'])
    same(interval(d)['ci95'],row['difference']['ci95'])
    pv.append(binomtest(int(values[0].sum()),512,.5).pvalue)
same(holm(pv),[r['holm_p'] for r in qj['records']])
# Independent scalar recomputation of representative complete six-bit cubes.
from src.sha256.core import InstrumentedSHA256
scalar=InstrumentedSHA256([2,3,7,24]);rr=list(fq['rounds'])
for base in [0,1]:
    for set_index,positions in enumerate([qj['target'],qj['controls'][0]]):
        sums={r:np.zeros(8,dtype=np.uint32) for r in [2,3,7,24]}
        for corner in range(64):
            message=bytearray(fq['blocks'][base,:55].tobytes())
            for j,bit in enumerate(positions):
                if (corner>>j)&1:message[bit//8]^=1<<(7-bit%8)
            trajectory=scalar.hash_with_trajectory(bytes(message))
            for r in sums:sums[r]^=np.array(trajectory.state_at(r).working_vars,dtype=np.uint32)
        for r in sums:np.testing.assert_array_equal(sums[r],fq['word_sums'][set_index,base,rr.index(r)])
checks.append('follow-up cubes: full words, indicators, Holm family and independent scalar corners')
z=read('followup/detour.json')
assert [r['seed'] for r in z['records']]==list(range(263910,264010))
for k,v in z['summary'].items():
    left,right=k.split(' minus ');d=[r[left]-r[right] for r in z['records']]
    check_effect_mean(d, v['mean']);same(interval(d)['ci95'],v['ci95'])
checks.append('100 fresh detour pairs: seed range and all five intervals')
assert np.all(a['discovery'][:,0,0]==2) and np.all(a['holdout'][:,0,0]==2)
checks.append('exact round-one MSB observation in both original partitions')
# Significance-matched control family: same bases and candidate, pool restricted to offsets 0-14.
menv=read('matched/environment.json')
for p,digest in list(menv['source_hashes'].items())+list(menv['input_hashes'].items()):
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
msets=read('matched/frozen_matched_sets.json')
assert msets['pool']==list(range(15)) and len(msets['controls'])==40
assert all(len(set(c))==6 and set(c)<=set(range(15)) and c==sorted(c) for c in msets['controls'])
ms=np.load(E/'matched/search.npz');msj=read('matched/search.json')
assert msj['controls']==msets['controls'] and ms['pattern_scores'].shape==(41,256,15,7)
np.testing.assert_array_equal(ms['scores'],ms['pattern_scores'].min(2))
np.testing.assert_array_equal(ms['blocks'],fs['blocks']);np.testing.assert_array_equal(ms['scores'][0],fs['scores'][0])
for j,row in enumerate(msj['records']):
    values=ms['scores'][:,:,j];cand=values[0].astype(float);ctrl=values[1:]
    same(cand.mean(),row['candidate_mean']);same(ctrl.mean(),row['matched_mean']);same(ctrl.mean(1),row['set_means'])
    check_effect_mean(cand-ctrl.mean(0), row['difference']['mean'])
    same(interval(cand-ctrl.mean(0))['ci95'],row['difference']['ci95'])
    assert row['matched_sets_below_candidate']==int((ctrl.mean(1)<cand.mean()).sum())
checks.append('matched search: identical bases and candidate scores, pool membership, per-set means, intervals and counts')
mq=np.load(E/'matched/cubes.npz');mqj=read('matched/cubes.json')
assert mqj['controls']==msets['controls'] and mq['word_sums'].shape==(41,512,14,8)
np.testing.assert_array_equal(mq['zero_indicators'],((mq['word_sums'][:,:,:,0]>>16)&1)==0)
np.testing.assert_array_equal(mq['blocks'],fq['blocks']);np.testing.assert_array_equal(mq['word_sums'][0],fq['word_sums'][0])
for j,row in enumerate(mqj['records']):
    values=mq['zero_indicators'][:,:,j];cand=values[0].astype(float);ctrl=values[1:];rates=ctrl.mean(1)
    same(cand.mean(),row['candidate_zero_rate']);same(ctrl.mean(),row['matched_zero_rate']);same(rates,row['set_rates'])
    check_effect_mean(cand-ctrl.mean(0), row['difference']['mean'])
    same(interval(cand-ctrl.mean(0))['ci95'],row['difference']['ci95']);same([rates.min(),rates.max()],row['matched_range'])
    assert row['matched_sets_at_or_above_candidate']==int((rates>=cand.mean()).sum())
    same((mq['word_sums'][0,:,j,0]==0).mean(),row['whole_word_a_zero']['candidate']);same((mq['word_sums'][1:,:,j,0]==0).mean(),row['whole_word_a_zero']['matched'])
checks.append('matched cubes: identical bases and candidate word sums, indicators, per-set rates, ranges, intervals and whole-word fractions')
ud=read('matched/uniform_diagnostics.json');offsets=np.array([np.mean(c) for c in sj['controls']])
for j,r in enumerate(fs['rounds']):same(np.corrcoef(fs['scores'][1:,:,j].mean(1),offsets)[0,1],ud['search_corr_control_mean_with_mean_offset'][str(int(r))])
above=np.array([sum(1 for o in c if 31-o>16) for c in qj['controls']]);j2=list(fq['rounds']).index(2);rates=fq['zero_indicators'][1:,:,j2].mean(1)
same(rates,ud['cube_round2_control_zero_rates']);same(np.corrcoef(rates,above)[0,1],ud['cube_round2_corr_zero_rate_with_bits_above_16'])
same([rates.min(),rates.max()],ud['cube_round2_control_zero_rate_range'])
same((fq['word_sums'][0,:,j2,0]==0).mean(),ud['cube_round2_whole_word_a_zero']['candidate']);same((fq['word_sums'][1:,:,j2,0]==0).mean(),ud['cube_round2_whole_word_a_zero']['uniform_controls'])
checks.append('uniform-control diagnostics: offset correlations, round-2 rates, ranges and whole-word fractions recomputed')
# Manuscript macros and table files regenerate identically from the receipts.
sys.path.insert(0,str(ROOT/'publication'));import derived
pm,pt=derived.primary_macros(E);assert derived.render(pm,pt)==(ROOT/'publication/numbers.tex').read_text(encoding='utf-8')
for name,filename in derived.TABLE_FILES:assert (ROOT/'publication'/filename).read_text(encoding='utf-8')==derived.table_text(pt[name]),filename
fm,ft=derived.followup_macros(E);assert derived.render(fm,ft)==(ROOT/'publication/followup_numbers.tex').read_text(encoding='utf-8')
checks.append('manuscript macros: numbers.tex, followup_numbers.tex and the four table files regenerate identically from the receipts')
# Graph summaries: means, detour intervals, and cluster sizes against saved labels.
for path in ['graphs.json','confirmation/graphs.json']:
    data=read(path)
    for h in ['1.0','8.0']:
        for metric in ['silhouette','spectral_gap']:
            v=data['summary'][h][metric];sha=[r['kernels'][h]['sha'][metric] for r in data['records']];rnd=[r['kernels'][h]['random'][metric] for r in data['records']]
            same(np.mean(sha),v['sha_mean']);same(np.mean(rnd),v['random_mean']);same(np.mean(np.array(sha)-np.array(rnd)),v['difference']['mean'])
        for r in data['records']:
            for label in ['sha','random']:
                v=r['kernels'][h][label];np.testing.assert_array_equal(np.bincount(v['labels'],minlength=5),v['cluster_sizes'])
    dif=[r['sha_detour']-r['random_detour'] for r in data['records']];v=data['summary']['detour']
    same(np.mean([r['sha_detour'] for r in data['records']]),v['sha_mean']);same(np.mean([r['random_detour'] for r in data['records']]),v['random_mean'])
    same(np.mean(dif),v['difference']['mean']);same(interval(dif)['ci95'],v['difference']['ci95'])
checks.append('graph summaries: means, detour intervals and cluster sizes against labels recomputed')
zs=read('sensitivity/detour.json')
for k,v in zs['summary'].items():
    left,right=k.split(' minus ');d=[r[left]-r[right] for r in zs['records']];same(np.mean(d),v['mean']);same(interval(d)['ci95'],v['ci95'])
checks.append('sensitivity detour: all five means and intervals recomputed from records')
aj0=read('avalanche.json');sel=a['selected'];rest=np.setdiff1d(np.arange(440),sel)
same((a['holdout'][:,sel,23].mean(axis=1)-a['holdout'][:,rest,23].mean(axis=1)).mean(),aj0['holdout_selected_minus_others']['mean'])
w0s=sel[sel<32];w0r=np.setdiff1d(np.arange(32),w0s)
w0diff=a['holdout'][:,w0s,23].mean(axis=1)-a['holdout'][:,w0r,23].mean(axis=1)
check_effect_mean(w0diff,aj0['holdout_word0_selected_minus_others']['mean'])
same(interval(w0diff)['ci95'],aj0['holdout_word0_selected_minus_others']['ci95'])
sj0=read('search.json');same(s['target'].mean(),sj0['target_mean_min_hamming']);same(s['controls'].mean(),sj0['control_mean_min_hamming'])
same((s['target']-s['controls'].mean(axis=0)).mean(),sj0['target_minus_mean_control']['mean'])
for name in ['RF','PCA_SVM']:
    accs=[r['models'][name]['accuracy'] for r in ml['records']];aucs=[r['models'][name]['auc'] for r in ml['records']]
    same(np.mean(accs),ml['summary'][name]['mean_accuracy']);same([min(accs),max(accs)],ml['summary'][name]['range']);same(np.mean(aucs),ml['summary'][name]['mean_auc'])
checks.append('avalanche, search and classifier summary means and ranges recomputed from records')
result={'passed':True,'checks':checks}
(ROOT/'publication/verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
