"""Generate supplementary manuscript tables from original and follow-up receipts."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.publication_study import interval
P=ROOT/'publication';E=P/'evidence'
def read(p):return json.loads((E/p).read_text())
def ci(v,scale=1):return f"[{scale*v['ci95'][0]:+.3f}, {scale*v['ci95'][1]:+.3f}]"
macros={};tables={};heat=[]
for group,path in [('Primary','graphs.json'),('Confirmation','confirmation/graphs.json')]:
    g=read(path)
    for h in ['1.0','8.0']:
        for j,t in enumerate([.01,.1,1.,10.]):
            v=interval([r['kernels'][h]['sha']['heat_trace'][j]-r['kernels'][h]['random']['heat_trace'][j] for r in g['records']])
            power=int(np.floor(np.log10(max(abs(x) for x in v['ci95']))));scale=10.**(-power)
            heat.append(f"{group} & {float(h):g} & {t:g} & $10^{{{power}}}$ & {v['mean']*scale:+.3f} & {ci(v,scale)} \\")
tables['HeatTraceRows']=heat
arr=np.load(E/'avalanche.npz');D=arr['discovery'];H=arr['holdout'];rounds=arr['rounds'];early=[]
for r in range(1,8):
    j=int(np.flatnonzero(rounds==r)[0]);d=D[:,:32,j].mean(0);h=H[:,:32,j].mean(0)
    early.append(f"{r} & {np.corrcoef(d,h)[0,1]:.3f} & {h[0]:.2f} & {h.mean():.2f} & {int(d.argmin())} & {int(h.argmin())} \\")
tables['EarlyCoordinateRows']=early
x=H[:,:,23].astype(float);center=x-x.mean(1,keepdims=True)
macros.update({'PositionSD':f'{x.mean(0).std(ddof=1):.3f}','NoiseFloor':f'{np.sqrt(center.var(0,ddof=1).sum()/((x.shape[1]-1)*len(x))):.3f}','RoundCorrelation':f'{np.corrcoef(D[:,:,23].mean(0),x.mean(0))[0,1]:.3f}'})
s=read('followup/search.json');q=read('followup/cubes.json');z=read('followup/detour.json')
ms=read('matched/search.json');mq=read('matched/cubes.json');ud=read('matched/uniform_diagnostics.json')
def ci2(v,scale=1,nd=2):return '['+f"{scale*v['ci95'][0]:+.{nd}f}"+', '+f"{scale*v['ci95'][1]:+.{nd}f}"+']'
rows=[]
for u,m in zip(s['records'],ms['records']):
    assert u['round']==m['round'] and abs(u['target_mean']-m['candidate_mean'])<1e-9
    rows.append(f"{u['round']} & {u['target_mean']:.2f} & {u['control_mean']:.2f} & {u['difference']['mean']:+.2f} {ci2(u['difference'])} & {m['matched_mean']:.2f} & {m['difference']['mean']:+.2f} {ci2(m['difference'])} & {m['matched_sets_below_candidate']}/40 \\")
tables['EarlySearchRows']=rows
def pformat(p):
    if p>=.001:return f'{p:.3f}'
    power=int(np.floor(np.log10(p)));return f'${p/10.**power:.2f}\\times10^{{{power}}}$'
rows=[]
for u,m in zip(q['records'],mq['records']):
    assert u['round']==m['round'] and abs(u['target_mean']-m['candidate_zero_rate'])<1e-9
    rows.append(f"{u['round']} & {100*u['target_mean']:.2f} & {100*u['control_mean']:.2f} & {100*u['difference']['mean']:+.2f} {ci2(u['difference'],100,1)} & {100*m['matched_zero_rate']:.2f} & {100*m['difference']['mean']:+.2f} {ci2(m['difference'],100,1)} & {m['matched_sets_at_or_above_candidate']}/40 & {pformat(u['holm_p'])} \\")
tables['EarlyCubeRows']=rows
tables['FreshDetourRows']=[f"{k.replace(' minus ',' -- ')} & {v['mean']:+.5f} & [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]" for k,v in z['summary'].items()]
v=z['summary']['state minus PCG64'];macros.update({'FreshDetourMean':f"{v['mean']:+.5f}",'FreshDetourCI':f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"})
v=q['records'][0];macros.update({'EarlyCubeTarget':f"{100*v['target_mean']:.2f}",'EarlyCubeControl':f"{100*v['control_mean']:.2f}",'EarlyCubeDifference':f"{100*v['difference']['mean']:.2f}",'EarlyCubeCI':ci(v['difference'],100),'EarlyCubeP':pformat(v['holm_p'])})
m2=ms['records'][0];c2=mq['records'][0]
macros.update({'MatchedSearchDiffTwo':f"{m2['difference']['mean']:+.2f}",'MatchedSearchCITwo':ci2(m2['difference']),'MatchedSearchBelowTwo':str(m2['matched_sets_below_candidate']),
 'UniformSearchOffsetCorr':f"{ud['search_corr_control_mean_with_mean_offset']['2']:.2f}",
 'UniformCubeMinTwo':f"{100*ud['cube_round2_control_zero_rate_range'][0]:.2f}",'UniformCubeMaxTwo':f"{100*ud['cube_round2_control_zero_rate_range'][1]:.2f}",
 'UniformCubeAboveCorr':f"{ud['cube_round2_corr_zero_rate_with_bits_above_16']:.2f}",
 'MatchedCubeMeanTwo':f"{100*c2['matched_zero_rate']:.2f}",'MatchedCubeRangeTwo':f"{100*c2['matched_range'][0]:.1f}--{100*c2['matched_range'][1]:.1f}",
 'MatchedCubeAtOrAboveTwo':str(c2['matched_sets_at_or_above_candidate']),'MatchedCubeDiffTwo':f"{100*c2['difference']['mean']:+.2f}",'MatchedCubeCITwo':ci2(c2['difference'],100,1),
 'WholeWordATwo':f"{100*c2['whole_word_a_zero']['candidate']:.2f}",'WholeWordAControlTwo':f"{100*ud['cube_round2_whole_word_a_zero']['uniform_controls']:.2f}"})
lines=['\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items()]
for k,rows in tables.items():
    rows=[row.rstrip(chr(92)).rstrip()+' '+chr(92)*2 for row in rows]
    lines.append('\\newcommand{\\'+k+'}{%\n'+'\n'.join(rows)+'%\n}')
(P/'followup_numbers.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(6.7,2.55),layout='constrained')
for data,label,color,style in [(D,'Discovery','#24618D','-'),(H,'Holdout','#C06B36','--')]:
    axes[0].plot(np.arange(32),data[:,:32,1].mean(0),ls=style,marker='o',ms=2.5,lw=1,color=color,label=label)
axes[0].set_xlabel('MSB-first input position within W0');axes[0].set_ylabel('Mean changed state bits');axes[0].set_title('Round 2: same entry, different response');axes[0].legend(frameon=False,fontsize=8)
for recs,label,color,marker,dx in [(s['records'],'Uniform W0 controls (20 sets)','#24618D','o',-.15),(ms['records'],'Significance-matched controls (40 sets)','#C06B36','s',.15)]:
    y=np.array([r['difference']['mean'] for r in recs]);bounds=np.array([r['difference']['ci95'] for r in recs])
    axes[1].errorbar(np.arange(len(y))+dx,y,yerr=[y-bounds[:,0],bounds[:,1]-y],fmt=marker,capsize=3,color=color,ms=4,label=label)
axes[1].legend(frameon=False,fontsize=6.5,loc='upper right')
axes[1].set_xticks(np.arange(len(y)),[r['round'] for r in s['records']]);axes[1].axhline(0,c='black',ls=':',lw=.8)
axes[1].set_xlabel('Completed rounds (discrete evaluations)');axes[1].set_ylabel('Candidate minus control (bits)');axes[1].set_title('Fresh equal-budget search')
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.savefig(P/'figures/early_structure.pdf',bbox_inches='tight');fig.savefig(P/'figures/early_structure.png',dpi=170,bbox_inches='tight');plt.close(fig)
