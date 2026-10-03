"""Tables, static figures and explicit hypothesis assessment from saved predictions."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hackology-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import RESULTS, REPORTS, DATA
from .evaluate import metrics, stratify_by_allele_distance, spearman

OUT = RESULTS / 'benchmark'
LABELS = {'blosum_nn':'BLOSUM MLP','blosum_ridge':'BLOSUM Ridge','onehot_ridge':'Allele-ID Ridge',
          'esm2_mean':'ESM-2 mean','esm2_complex':'ESM-2 residue'}


def main():
    REPORTS.mkdir(exist_ok=True)
    scores = pd.read_csv(OUT/'per_fold.csv')
    from .run_benchmark import ARMS
    counts=scores[scores.metric=='spearman'].groupby(['arm','split_regime']).fold.nunique()
    assert all(counts.get((a,r),0)==5 for a in ARMS for r in ['peptide','allele']), 'Run all five core arms and folds before generating the final report.'
    rows = pd.read_csv(OUT/'per_row.csv')
    summary = scores.groupby(['arm','split_regime','metric']).value.agg(['mean','std','count']).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    distance = pd.read_csv(RESULTS/'splits/allele_distance.csv')
    held = rows[rows.split_regime == 'allele']
    strat = stratify_by_allele_distance(held,distance)
    strat.to_csv(OUT/'distance_stratified.csv',index=False)
    allele_stats=[]
    for (arm,allele,fold),g in held.groupby(['arm','allele','fold']):
        allele_stats.append(dict(arm=arm,allele=allele,fold=fold,rows=len(g),**metrics(g.y_true,g.y_pred)))
    allele_stats=pd.DataFrame(allele_stats).merge(distance,on=['allele','fold'],validate='many_to_one')
    allele_stats.to_csv(OUT/'per_allele.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for arm,g in strat.groupby('arm'):
        axes[0].plot(g.identity_band,g.macro_allele_spearman,'o-',label=LABELS[arm])
    axes[0].set(xlabel='Nearest training allele: pseudosequence identity',ylabel='Mean within-allele Spearman',title='Held-out alleles: sequence distance')
    allele_stats['support_band']=pd.cut(allele_stats.nearest_train_allele_rows,[0,100,500,1000,np.inf],labels=['1–100','101–500','501–1000','>1000'])
    support=allele_stats.groupby(['arm','support_band'],observed=True).agg(spearman=('spearman','mean'),alleles=('allele','size')).reset_index()
    support.to_csv(OUT/'support_stratified.csv',index=False)
    allele_stats['identity_band']=pd.cut(allele_stats.max_identity_to_train_allele,[0,.8,.9,.95,1.0001],right=False,labels=['<80%','80–90%','90–95%','95–100%'])
    base=allele_stats[allele_stats.arm=='blosum_nn'][['allele','spearman']].rename(columns={'spearman':'baseline_spearman'})
    joint=allele_stats.merge(base,on='allele',validate='many_to_one')
    joint['gap']=joint.spearman-joint.baseline_spearman
    joint=joint[joint.arm.isin(['esm2_mean','esm2_complex'])].groupby(['arm','identity_band','support_band'],observed=True).agg(gap=('gap','mean'),alleles=('gap','count')).reset_index()
    joint.to_csv(OUT/'distance_support_gaps.csv',index=False)
    heat,heat_axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,arm in zip(heat_axes,['esm2_mean','esm2_complex']):
        g=joint[joint.arm==arm]
        grid=g.pivot(index='identity_band',columns='support_band',values='gap').reindex(index=['<80%','80–90%','90–95%','95–100%'],columns=['1–100','101–500','501–1000','>1000'])
        count=g.pivot(index='identity_band',columns='support_band',values='alleles').reindex(index=grid.index,columns=grid.columns)
        im=ax.imshow(grid.to_numpy(dtype=float),vmin=-.6,vmax=.6,cmap='RdBu',aspect='auto')
        for i in range(4):
            for j in range(4):
                if np.isfinite(grid.iloc[i,j]):
                    ax.text(j,i,f'{grid.iloc[i,j]:+.2f}\nn={int(count.iloc[i,j])}',ha='center',va='center',fontsize=8)
        ax.set_xticks(range(4),grid.columns)
        ax.set_yticks(range(4),grid.index)
        ax.set(xlabel='Measurements of nearest training allele',ylabel='Nearest allele identity',title=LABELS[arm]+' minus BLOSUM MLP')
    heat.colorbar(im,ax=heat_axes,label='Mean within-allele Spearman difference',shrink=.8)
    heat.savefig(REPORTS/'allele_joint_gap.png',dpi=180)
    plt.close(heat)

    for arm,g in support.groupby('arm'):
        axes[1].plot(g.support_band.astype(str),g.spearman,'o-',label=LABELS[arm])
    axes[1].set(xlabel='Measurements of nearest training allele',ylabel='Mean within-allele Spearman',title='Support from a neighbouring allele')
    axes[0].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(REPORTS/'allele_distance.png',dpi=180)
    plt.close(fig)
    # Full paired gaps, not a claim of significance from correlated CV folds.
    pairs=[]
    for regime in ['peptide','allele']:
        p=scores[(scores.split_regime==regime)&(scores.metric=='spearman')].pivot(index='fold',columns='arm',values='value')
        for arm in ['esm2_mean','esm2_complex']:
            if arm in p:
                for baseline in ['blosum_nn','blosum_ridge']:
                    for fold in p.index:
                        pairs.append(dict(arm=arm,baseline=baseline,split_regime=regime,fold=fold,gap=p.loc[fold,arm]-p.loc[fold,baseline]))
    pd.DataFrame(pairs,columns=['arm','baseline','split_regime','fold','gap']).to_csv(OUT/'paired_gaps.csv',index=False)
    learning=pd.read_csv(OUT/'learning_curve.csv')
    ls=learning.groupby(['arm','split_regime','fraction','metric']).value.agg(['mean','std','count']).reset_index()
    ls.to_csv(OUT/'learning_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,regime in zip(axes,['peptide','allele']):
        for arm,g in ls[(ls.metric=='spearman')&(ls.split_regime==regime)].groupby('arm'):
            ax.errorbar(g.fraction*100,g['mean'],yerr=g['std'],marker='o',capsize=3,label=LABELS[arm])
        ax.set(xlabel='Training rows retained (%)',ylabel='Spearman (mean ± fold SD)',title=f'Held-out {regime}')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(REPORTS/'learning_curve.png',dpi=180)
    plt.close(fig)
    # Gap curve assesses P3 directly.
    gap_rows=[]
    for (regime,fraction,fold),g in learning[learning.metric=='spearman'].groupby(['split_regime','fraction','fold']):
        values=g.set_index('arm').value
        for arm in ['esm2_mean','esm2_complex']:
            if arm in values and 'blosum_nn' in values:
                gap_rows.append(dict(split_regime=regime,fraction=fraction,fold=fold,arm=arm,gap=values[arm]-values['blosum_nn']))
    gap_df=pd.DataFrame(gap_rows,columns=['split_regime','fraction','fold','arm','gap'])
    gap_df.to_csv(OUT/'learning_gaps.csv',index=False)
    if len(gap_df):
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,regime in zip(axes,['peptide','allele']):
            for arm,g in gap_df[gap_df.split_regime==regime].groupby('arm'):
                stat=g.groupby('fraction').gap.agg(['mean','std'])
                ax.errorbar(stat.index*100,stat['mean'],yerr=stat['std'],marker='o',label=LABELS[arm])
            ax.axhline(0,color='black',lw=.7)
            ax.set(xlabel='Training rows retained (%)',ylabel='Spearman gap vs BLOSUM MLP',title=regime)
            ax.legend(fontsize=8)
        fig.savefig(REPORTS/'learning_gap.png',dpi=180)
        plt.close(fig)
    compute=[]
    fits=pd.read_csv(OUT/'fit_compute.csv')
    for arm in sorted(rows.arm.unique()):
        meta_path=DATA/'processed'/f'{arm}_meta.json'
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
        compute.append(dict(arm=arm,gpu_seconds=meta.get('gpu_seconds',0),
            peak_gpu_memory_bytes=meta.get('peak_gpu_memory_bytes',0),
            cpu_fit_predict_seconds=fits[(fits.arm==arm)&(fits.fraction==1)].fit_predict_cpu_wall_seconds.sum()))
    compute=pd.DataFrame(compute)
    compute.to_csv(OUT/'compute.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for regime,marker in [('peptide','o'),('allele','s')]:
        for i,row in compute.iterrows():
            val=summary[(summary.arm==row.arm)&(summary.split_regime==regime)&(summary.metric=='spearman')]['mean'].iloc[0]
            ax.scatter(row.gpu_seconds,val,marker=marker)
            ax.annotate(f'{LABELS[row.arm]} ({regime})',(row.gpu_seconds,val),xytext=((-7 if row.gpu_seconds>0 else 5),5+i*3),ha=('right' if row.gpu_seconds>0 else 'left'),textcoords='offset points',fontsize=7)
    ax.set(xlabel='One-time GPU extraction seconds (CPU fitting reported separately)',ylabel='Spearman',title='Representation compute and predictive performance')
    ax.set_xscale('symlog',linthresh=1)
    ax.grid(alpha=.2)
    fig.savefig(REPORTS/'compute_performance.png',dpi=180)
    plt.close(fig)
    table=['| Arm | Held-out peptide ρ | Held-out allele ρ |','|---|---:|---:|']
    for arm in sorted(rows.arm.unique()):
        vals=[]
        for regime in ['peptide','allele']:
            s=summary[(summary.arm==arm)&(summary.split_regime==regime)&(summary.metric=='spearman')].iloc[0]
            vals.append(f'{s["mean"]:.3f} ± {s["std"]:.3f}')
        table.append(f'| {LABELS[arm]} | {vals[0]} | {vals[1]} |')
    (REPORTS/'benchmark_table.md').write_text('\n'.join(table)+'\n')
    print('\n'.join(table))


if __name__=='__main__': main()
