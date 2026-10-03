"""v2 tables and figures: macro within-allele scores and paired confidence intervals."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hackology-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import RESULTS,REPORTS,DATA
from .evaluate import PRIMARY,allele_metrics,paired_delta_ci,stratify_by_allele_distance

OUT=RESULTS/'benchmark'
LABELS={'blosum_nn':'BLOSUM MLP (reference)','blosum_ridge':'BLOSUM Ridge',
        'onehot_ridge':'Allele-ID Ridge (floor)','esm2_mean':'ESM-2 mean','esm2_joint':'ESM-2 joint'}
COLORS=dict(zip(LABELS,['#2563eb','#6b7280','#c0a1c9','#ea580c','#059669']))
PLMS=['esm2_mean','esm2_joint']
REGIMES=['peptide','allele','locus','allele_strict']


def save_figure(fig,name):
    fig.savefig(REPORTS/name,dpi=180,bbox_inches='tight')
    plt.close(fig)


def score_ci(scores):
    rows=[]
    primary=scores[scores.metric==PRIMARY]
    for (regime,fraction),g in primary.groupby(['split_regime','fraction']):
        baseline=g[g.arm=='blosum_nn']
        for arm,h in g.groupby('arm'):
            rows.append(dict(arm=arm,baseline='blosum_nn',split_regime=regime,fraction=fraction,
                             **paired_delta_ci(h,baseline)))
    return pd.DataFrame(rows)


def per_allele(rows):
    out=[]
    for (arm,regime,fold),g in rows.groupby(['arm','split_regime','fold']):
        stat=allele_metrics(g,rows.allele.unique() if regime=='peptide' else None)
        stat['arm']=arm;stat['split_regime']=regime;stat['fold']=fold
        stat=stat.merge(rows[['allele','locus']].drop_duplicates(),on='allele',validate='one_to_one')
        out.append(stat)
    return pd.concat(out,ignore_index=True)


def allele_delta(arms,arm,baseline='blosum_nn'):
    a=arms[(arms.arm==arm)&arms.label_eligible][['allele','spearman']]
    b=arms[(arms.arm==baseline)&arms.label_eligible][['allele','spearman']]
    return paired_delta_ci(a,b,by='allele',value='spearman',method='bootstrap')


def distance_analysis(rows,alleles):
    distance=pd.read_csv(RESULTS/'splits/allele_distance.csv')
    held=rows[rows.split_regime=='allele']
    stratify_by_allele_distance(held,distance).to_csv(OUT/'distance_stratified.csv',index=False)
    meta=distance[['allele','fold','max_identity_to_train','n_train_rows','nearest_train_allele_rows']]
    a=alleles[alleles.split_regime=='allele'].merge(meta,on=['allele','fold'],validate='many_to_one')
    a['identity_band']=pd.cut(a.max_identity_to_train,[0,.8,.9,.95,1.0001],right=False,
                             labels=['<80%','80–90%','90–95%','95–100%'])
    a['support_band']=pd.cut(a.nearest_train_allele_rows,[0,100,500,1000,np.inf],
                            labels=['1–100','101–500','501–1000','>1000'])
    comparisons=[]; absolute=[]
    for field in ['identity_band','support_band']:
        for band,g in a.groupby(field,observed=True):
            for arm,h in g.groupby('arm'):
                good=h[h.include_in_macro]
                absolute.append(dict(stratifier=field,band=str(band),arm=arm,
                    macro=float(good.spearman.mean()),eligible_alleles=len(good),
                    excluded_alleles=int((~h.include_in_macro).sum())))
                comparisons.append(dict(stratifier=field,band=str(band),arm=arm,**allele_delta(g,arm)))
    # A maximum-distance stress point for each observed locus; conditional allele bootstrap.
    locus_d=pd.read_csv(RESULTS/'splits/locus_distance.csv')
    for fold,g in alleles[alleles.split_regime=='locus'].groupby('fold'):
        band=f'Holdout {g.locus.iloc[0]}'
        for arm,h in g.groupby('arm'):
            good=h[h.include_in_macro]
            absolute.append(dict(stratifier='identity_band',band=band,arm=arm,
                macro=float(good.spearman.mean()),eligible_alleles=len(good),excluded_alleles=int((~h.include_in_macro).sum())))
            comparisons.append(dict(stratifier='identity_band',band=band,arm=arm,**allele_delta(g,arm),
                median_identity=float(locus_d.loc[locus_d.fold==fold,'max_identity_to_train'].median())))
    absolute=pd.DataFrame(absolute);comparisons=pd.DataFrame(comparisons)
    absolute.to_csv(OUT/'stratified_macro.csv',index=False)
    comparisons.to_csv(OUT/'stratified_paired_ci.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),layout='constrained')
    bands=['95–100%','90–95%','80–90%','<80%','Holdout A','Holdout B']
    for arm in LABELS:
        g=absolute[(absolute.stratifier=='identity_band')&(absolute.arm==arm)].set_index('band').reindex(bands)
        axes[0].plot(range(len(bands)),g.macro,marker='o',color=COLORS[arm],label=LABELS[arm])
    axes[0].set_xticks(range(len(bands)),bands,rotation=30,ha='right')
    axes[0].set(title='Ranking within held-out alleles',ylabel='Macro within-allele Spearman',xlabel='Nearest training allele identity; locus stress tests')
    axes[0].legend(fontsize=7,loc='best')
    for ax,field,order in [(axes[1],'identity_band',bands),(axes[2],'support_band',['1–100','101–500','501–1000','>1000'])]:
        for j,arm in enumerate(PLMS):
            g=comparisons[(comparisons.stratifier==field)&(comparisons.arm==arm)].set_index('band').reindex(order)
            x=np.arange(len(order))+(j-.5)*.12
            ax.errorbar(x,g.delta,yerr=np.maximum(0,np.vstack([g.delta-g.ci_low,g.ci_high-g.delta])),
                        marker='o',capsize=3,color=COLORS[arm],label=LABELS[arm])
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(order)),order,rotation=30,ha='right')
        ax.set(ylabel='Paired macro difference vs BLOSUM MLP (95% CI)')
        ax.legend(fontsize=8)
    axes[1].set_title('Effect by sequence distance')
    axes[1].set_xlabel('Identity bands; held-out-locus anchors')
    axes[2].set_title('Support in this dataset')
    axes[2].set_xlabel('Measurements of nearest training allele')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Same-allele training rows = 0 throughout. Intervals resample paired alleles within fixed splits.',fontsize=10)
    save_figure(fig,'allele_distance.png')
    return comparisons


def learning_figures(scores,ci):
    selected=scores[(scores.metric==PRIMARY)&scores.split_regime.isin(['peptide','allele'])]
    summary=selected.groupby(['arm','split_regime','fraction']).value.agg(['mean','std','count']).reset_index()
    summary.to_csv(OUT/'learning_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,regime in zip(axes,['peptide','allele']):
        for arm,g in summary[summary.split_regime==regime].groupby('arm'):
            ax.plot(g.fraction*100,g['mean'],'o-',color=COLORS[arm],label=LABELS[arm])
        ax.set(xlabel='Training rows retained (%)',ylabel='Macro within-allele Spearman',title=f'Held-out {regime}')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save_figure(fig,'learning_curve.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,regime in zip(axes,['peptide','allele']):
        for arm in PLMS:
            g=ci[(ci.split_regime==regime)&(ci.arm==arm)].sort_values('fraction')
            ax.errorbar(g.fraction*100,g.delta,yerr=np.maximum(0,np.vstack([g.delta-g.ci_low,g.ci_high-g.delta])),
                        marker='o',capsize=3,color=COLORS[arm],label=LABELS[arm])
        ax.axhline(0,color='black',lw=.8)
        ax.set(xlabel='Training rows retained (%)',ylabel='Macro difference vs BLOSUM MLP (95% paired CI)',title=f'Held-out {regime}')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    save_figure(fig,'learning_gap.png')


def controls(alleles):
    locus=[]
    for fold,g in alleles[alleles.split_regime=='locus'].groupby('fold'):
        for arm,h in g.groupby('arm'):
            locus.append(dict(arm=arm,fold=fold,held_out_locus=h.locus.iloc[0],
                macro=float(h.loc[h.include_in_macro,'spearman'].mean()),
                eligible_alleles=int(h.include_in_macro.sum()),**allele_delta(g,arm)))
    pd.DataFrame(locus).to_csv(OUT/'locus_paired_ci.csv',index=False)
    strict=[];reference=[]
    s=alleles[alleles.split_regime=='allele_strict']
    p=alleles[(alleles.split_regime=='allele')&(alleles.fold==0)]
    for arm in LABELS:
        a=s[(s.arm==arm)&s.label_eligible][['allele','spearman']]
        b=p[(p.arm==arm)&p.label_eligible][['allele','spearman']]
        strict.append(dict(arm=arm,strict_macro=float(a.spearman.mean()),permissive_macro=float(b.spearman.mean()),
                           **paired_delta_ci(a,b,by='allele',value='spearman',method='bootstrap')))
        reference.append(dict(arm=arm,**allele_delta(s,arm)))
    pd.DataFrame(strict).to_csv(OUT/'strict_vs_permissive.csv',index=False)
    pd.DataFrame(reference).to_csv(OUT/'strict_vs_baseline_ci.csv',index=False)


def compute_plot(summary):
    fit=pd.read_csv(OUT/'fit_compute.csv'); rows=[]
    for arm in LABELS:
        path=DATA/'processed'/f'{arm}_meta.json'
        m=json.loads(path.read_text()) if path.exists() else {}
        seconds=m.get('gpu_seconds',0); n=sum(b['sequences'] for b in m.get('batches',[]))
        rows.append(dict(arm=arm,gpu_seconds=seconds,peak_gpu_memory_bytes=m.get('peak_gpu_memory_bytes',0),
            extraction_sequences=n,inference_sequences_per_second=n/m['inference_seconds'] if n else np.nan,
            gpu_function_sequences_per_second=n/seconds if n else np.nan,
            cpu_fit_predict_seconds_full_budget=fit[(fit.arm==arm)&(fit.fraction==1)].fit_predict_cpu_wall_seconds.sum(),
            new_v2_cpu_fit_predict_seconds=fit[(fit.arm==arm)&fit.split_regime.isin(['locus','allele_strict'])].fit_predict_cpu_wall_seconds.sum()))
    compute=pd.DataFrame(rows);compute.to_csv(OUT/'compute.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,regime in zip(axes,['peptide','allele']):
        for i,row in compute.iterrows():
            g=summary[(summary.arm==row.arm)&(summary.split_regime==regime)&(summary.metric==PRIMARY)].iloc[0]
            ax.scatter(row.gpu_seconds,g['mean'],color=COLORS[row.arm],s=45)
            label=LABELS[row.arm].replace(' (reference)','').replace(' (floor)','')
            offset=(5,(-12 if row.arm=='onehot_ridge' else 8)) if row.gpu_seconds==0 else (-6,8)
            ax.annotate(label,(row.gpu_seconds,g['mean']),xytext=offset,textcoords='offset points',
                        ha='left' if row.gpu_seconds==0 else 'right',fontsize=8)
        ax.set(xlabel='One-time extraction GPU seconds',ylabel='Macro within-allele Spearman',title=f'Held-out {regime}')
        ax.set_xscale('symlog',linthresh=1);ax.grid(alpha=.2)
    fig.suptitle('CPU fitting and throughput are reported separately; GPU seconds are not total cost.',fontsize=10)
    save_figure(fig,'compute_performance.png')


def main():
    scores=pd.read_csv(OUT/'per_fold.csv')
    if not set(REGIMES)<=set(scores.split_regime): raise ValueError('Complete all v2 regimes before reporting')
    full=scores[scores.fraction==1]
    expected={'peptide':5,'allele':5,'locus':2,'allele_strict':1}
    for (arm,regime),g in full[full.metric==PRIMARY].groupby(['arm','split_regime']):
        assert len(g)==expected[regime],(arm,regime,len(g))
    rows=pd.read_csv(OUT/'per_row.csv')
    summary=full.groupby(['arm','split_regime','metric']).value.agg(['mean','std','count']).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    ci=score_ci(scores);ci.to_csv(OUT/'paired_comparisons.csv',index=False)
    # Direct A/B extraction comparison is paired too (P4's tested part).
    contrasts=[]
    for (regime,fraction),g in scores[scores.metric==PRIMARY].groupby(['split_regime','fraction']):
        contrasts.append(dict(arm='esm2_joint',baseline='esm2_mean',split_regime=regime,fraction=fraction,
            **paired_delta_ci(g[g.arm=='esm2_joint'],g[g.arm=='esm2_mean'])))
    pd.DataFrame(contrasts).to_csv(OUT/'extraction_paired_ci.csv',index=False)
    a=per_allele(rows);a.to_csv(OUT/'per_allele.csv',index=False)
    a[(a.split_regime=='allele') & a.high_zero_fraction].to_csv(OUT/'high_zero_summary.csv',index=False)
    REPORTS.mkdir(exist_ok=True)
    controls(a);distance_analysis(rows,a);learning_figures(scores,ci);compute_plot(summary)
    tables=[]
    for regime in REGIMES:
        tables += [f'### {regime}', '', '| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |',
                   '|---|---:|---:|---:|---:|']
        for arm in LABELS:
            g=summary[(summary.arm==arm)&(summary.split_regime==regime)].set_index('metric')
            macro=g.loc[PRIMARY];p=g.loc['pooled_spearman','mean'];gap=g.loc['pooled_minus_macro','mean']
            c=ci[(ci.arm==arm)&(ci.split_regime==regime)&(ci.fraction==1)].iloc[0]
            if regime=='allele_strict':
                c=pd.read_csv(OUT/'strict_vs_baseline_ci.csv').set_index('arm').loc[arm]
            delta='reference' if arm=='blosum_nn' else f'{c.delta:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}]'
            score=f'{macro["mean"]:.3f}' if regime=='allele_strict' else f'{macro["mean"]:.3f} ± {macro["std"]:.3f}'
            tables.append(f'| {LABELS[arm]} | {score} | {delta} | {p:.3f} | {gap:+.3f} |')
        tables+=['']
    tables+=['Peptide/allele: paired t intervals over five folds. Locus summary: two-fold t interval, highly unstable; see per-locus paired-allele intervals. Strict: paired-allele bootstrap conditional on fold 0. Intervals do not account for shared training data or method selection.']
    (REPORTS/'benchmark_table.md').write_text('\n'.join(tables)+'\n')
    print('\n'.join(tables))


if __name__=='__main__':main()
