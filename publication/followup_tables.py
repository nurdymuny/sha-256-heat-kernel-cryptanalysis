"""Write followup_numbers.tex from publication/derived.py and draw Figure 5 (early structure)."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'publication';E=P/'evidence'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(P))
import derived
macros,tables=derived.followup_macros(E)
(P/'followup_numbers.tex').write_text(derived.render(macros,tables),encoding='utf-8')
arr=np.load(E/'avalanche.npz');D=arr['discovery'];H=arr['holdout']
s=derived.read(E,'followup/search.json');ms=derived.read(E,'matched/search.json')
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
