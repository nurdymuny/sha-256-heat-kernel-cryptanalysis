"""Build all tables, figures, and numeric macros from frozen study receipts."""
from pathlib import Path
import json,sys,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'publication'; E=P/'evidence'; F=P/'figures'; F.mkdir(exist_ok=True)
def read(name):return json.loads((E/name).read_text())
g=read('graphs.json'); c=read('confirmation/graphs.json'); a=read('avalanche.json'); s=read('search.json'); q=read('cubes.json'); m=read('ml.json'); z=read('sensitivity/detour.json'); entry=read('confirmation/entry.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'savefig.bbox':'tight'})
blue='#24618D'; orange='#C06B36'; gray='#64748B'
def save(fig,name):
    fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.png'),dpi=170);plt.close(fig)
def fmt(x):return f'{x:.4f}'
def ci(v):return f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
def tablefile(name,rows): (P/name).write_text('\n'.join(rows)+'\n',encoding='utf-8')

# Figure 1: every independent dataset, shared fitted pipeline and explicit bandwidth.
fig,axes=plt.subplots(1,2,figsize=(6.7,2.8),layout='constrained')
for ax,h in zip(axes,['1.0','8.0']):
    for j,(cohort,data) in enumerate([('Primary',g),('Confirmation',c)]):
        for offset,label,col in [(-.14,'sha',blue),(.14,'random',orange)]:
            y=[r['kernels'][h][label]['silhouette'] for r in data['records']]
            x=j+offset+np.linspace(-.05,.05,len(y));ax.scatter(x,y,s=13,color=col,alpha=.7,label=label.upper() if j==0 else None)
            ax.plot([j+offset-.08,j+offset+.08],[np.mean(y)]*2,color=col,lw=2)
    ax.set_xticks([0,1],['Primary','Confirmation']);ax.set_title(f'Kernel parameter h = {float(h):g}');ax.set_ylabel('Fitted silhouette');ax.grid(axis='y',alpha=.15)
axes[0].legend(frameon=False,loc='lower left'); save(fig,'graph_controls')

# Figure 2: actual per-input-coordinate interventions and entry-aligned curves.
arr=np.load(E/'avalanche.npz'); held=arr['holdout']; rounds=arr['rounds']; selected=arr['selected']; means=held.mean(axis=0)
byword=np.array([means[w*32:min((w+1)*32,440),:24].mean(axis=0) for w in range(14)])
fig,axes=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
im=axes[0].imshow(byword,origin='upper',aspect='auto',vmin=0,vmax=128,cmap='viridis',extent=[.5,24.5,13.5,-.5]);axes[0].set_xlabel('Completed compression rounds');axes[0].set_ylabel('Flipped message word Wj');axes[0].set_title('Holdout mean Hamming distance');axes[0].set_yticks([0,4,8,12,13]);fig.colorbar(im,ax=axes[0],shrink=.78,label='Changed state bits')
for w,col in zip([0,4,8,12],[blue,orange,'#458C70',gray]):
    x=rounds[:24]-(w+1);mask=(x>=0)&(x<=10)
    axes[1].plot(x[mask],byword[w,mask],marker='o',ms=3,label=f'W{w}',color=col)
axes[1].axhline(128,color='black',ls=':',lw=.8);axes[1].set_xlabel('Rounds after message-word entry');axes[1].set_ylabel('Changed state bits');axes[1].set_title('Entry-aligned trajectories');axes[1].legend(frameon=False);save(fig,'avalanche_entry')

# Figure 3: holdout selection and equal-budget search.
discovery=arr['discovery'].mean(axis=0)[:,23]; holdout=means[:,23]
fig,axes=plt.subplots(1,2,figsize=(6.7,2.9),layout='constrained')
axes[0].scatter(discovery,holdout,s=9,color=gray,alpha=.5);axes[0].scatter(discovery[selected],holdout[selected],s=18,color=orange,label='22 discovery-selected bits')
axes[0].axhline(128,c='black',ls=':',lw=.7);axes[0].set_xlabel('Discovery mean Hamming distance');axes[0].set_ylabel('Holdout mean Hamming distance');axes[0].set_title('Round 24; one-bit interventions');axes[0].legend(frameon=False,loc='lower right',fontsize=7)
search=np.load(E/'search.npz');dif=search['target']-search['controls'].mean(axis=0)
axes[1].hist(dif,bins=20,color=blue,alpha=.85);axes[1].axvline(0,c='black',ls=':',lw=.8);axes[1].set_xlabel('Target minus mean control (bits)');axes[1].set_ylabel('Independent base messages');axes[1].set_title('Equal-budget two-bit searches');save(fig,'holdout_search')

# Figure 4: same raw-block cube observable at two injection words.
fig,axes=plt.subplots(1,3,figsize=(6.7,2.8),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.45]})
for ax,row in zip(axes[:2],[entry['records'][0],entry['records'][-1]]):
    data=np.array([row['whole_word_zero_rates'][str(r)] for r in row['rounds']]).T
    im=ax.imshow(data,aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set_yticks(range(8),list('abcdefgh'));ax.set_xticks([0,2,4,6,7],['0','2','4','6','8']);ax.set_xlabel('Rounds after entry');ax.set_title(f"3-bit cube in W{row['message_word']}")
axes[0].set_ylabel('Whole-word cube sum is zero');fig.colorbar(im,ax=axes[:2],shrink=.7,label='Fraction of bases')
rr=[v['round'] for v in q['records']];dif=np.array([v['difference']['mean'] for v in q['records']])*100;cis=np.array([v['difference']['ci95'] for v in q['records']])*100
axes[2].errorbar(rr,dif,yerr=[dif-cis[:,0],cis[:,1]-dif],fmt='o',ms=3,capsize=2,color=orange);axes[2].axhline(0,c='black',ls=':',lw=.8);axes[2].set_xlabel('Completed rounds');axes[2].set_ylabel('Zero-rate difference (pp)');axes[2].set_title('6-bit cube: target minus control');save(fig,'cube_controls')

# Figure 5: independent model fits and actual digest holdouts.
fig,axes=plt.subplots(1,2,figsize=(6.7,2.35),layout='constrained')
for ax,key,title in zip(axes,['accuracy','auc'],['Held-out accuracy','Held-out ROC AUC']):
    for j,(name,col) in enumerate([('RF',blue),('PCA_SVM',orange)]):
        vals=[r['models'][name][key] for r in m['records']]
        ax.scatter(j+np.linspace(-.1,.1,5),vals,color=col,s=27)
        ax.plot([j-.16,j+.16],[np.mean(vals)]*2,color=col,lw=2)
    ax.axhline(.5,color='black',ls=':',lw=.8);ax.set_xticks([0,1],['Random forest','PCA + SVM']);ax.set_title(title);ax.set_ylim(.475,.525);ax.grid(axis='y',alpha=.15)
save(fig,'digest_classifiers')

rows=[]
for cohort,data in [('Primary',g),('Confirmation',c)]:
    for h in ['1.0','8.0']:
        for metric,label in [('silhouette','Silhouette'),('spectral_gap','Gap')]:
            v=data['summary'][h][metric]; rows.append(f"{cohort} & {h} & {label} & {v['sha_mean']:.4f} & {v['random_mean']:.4f} & {v['difference']['mean']:+.4f} {ci(v['difference'])} \\\\")
    v=data['summary']['detour']; rows.append(f"{cohort} & -- & Detour & {v['sha_mean']:.4f} & {v['random_mean']:.4f} & {v['difference']['mean']:+.4f} {ci(v['difference'])} \\\\")
tablefile('graph_table.tex',rows)
tablefile('sensitivity_table.tex',[f"{key.replace(' minus ',' -- ')} & {v['mean']:+.5f} & {ci(v)} \\\\" for key,v in z['summary'].items()])
tablefile('cube_table.tex',[f"{v['round']} & {100*v['target_zero_rate']:.2f} & {100*v['control_zero_rate']:.2f} & {100*v['difference']['mean']:+.2f} & [{100*v['difference']['ci95'][0]:.2f}, {100*v['difference']['ci95'][1]:.2f}] & {v['holm_p']:.3f} \\\\" for v in q['records']])
tablefile('ml_table.tex',[f"{name.replace('_',' + ')} & {100*v['mean_accuracy']:.3f} & {100*v['range'][0]:.3f}--{100*v['range'][1]:.3f} & {v['mean_auc']:.4f} \\\\" for name,v in m['summary'].items()])
macros={'HoldoutDifference':f"{a['holdout_selected_minus_others']['mean']:+.3f}",
        'HoldoutCI':ci(a['holdout_selected_minus_others']), 'SearchDifference':f"{s['target_minus_mean_control']['mean']:+.3f}",
        'SearchCI':ci(s['target_minus_mean_control']), 'FinalHamming':f"{a['round64_mean_hamming']:.3f}",
        'RFAccuracy':f"{100*m['summary']['RF']['mean_accuracy']:.3f}", 'SVMAccuracy':f"{100*m['summary']['PCA_SVM']['mean_accuracy']:.3f}"}
(P/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
with (P/'numbers.tex').open('a') as file:
    for command,filename in [('GraphRows','graph_table.tex'),('SensitivityRows','sensitivity_table.tex'),('CubeRows','cube_table.tex'),('MLRows','ml_table.tex')]:
        file.write('\\newcommand{\\'+command+'}{%\n'+(P/filename).read_text().rstrip()+'%\n}\n')
out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
for _ in range(2):
    subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(out),'sha256_measurement.tex'],cwd=P,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(out/'sha256_measurement.pdf')
