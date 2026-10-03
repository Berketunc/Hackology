"""Acceptance checks for the cached, full-budget matched-head follow-up."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT, RESULTS
from src.run_benchmark import MLP_EXTENSION


def main():
    out=RESULTS/'benchmark'
    jobs=out/'jobs'
    files=sorted(jobs.glob('esm2_*_nn_*.csv'))
    assert len(files)==26, f'Expected 26 new fits, found {len(files)}'
    regimes={'peptide':5,'allele':5,'locus':2,'allele_strict':1}
    pf=pd.read_csv(out/'per_fold.csv')
    audit=pf[(pf.metric=='macro_eligible_alleles')&(pf.fraction==1)]
    for arm in MLP_EXTENSION:
        for regime,n in regimes.items():
            for fold in range(n):
                stem=f'{arm}_{regime}_{fold}_1'
                m=json.loads((jobs/f'{stem}.json').read_text())
                b=json.loads((jobs/f'blosum_nn_{regime}_{fold}_1.json').read_text())
                assert m['alpha'] is None and isinstance(m['best_epoch'],int)
                assert m['training_row_ids_sha256']==b['training_row_ids_sha256']
                assert m['n_train']==b['n_train'] and m['seed']==b['seed']
                rows=pd.read_csv(jobs/f'{stem}.csv')
                baseline=pd.read_csv(jobs/f'blosum_nn_{regime}_{fold}_1.csv')
                assert rows.row_index.equals(baseline.row_index)
                np.testing.assert_array_equal(rows.y_true,baseline.y_true)
                assert np.isfinite(rows.y_pred).all()
                a=audit[(audit.split_regime==regime)&(audit.fold==fold)].set_index('arm').value
                assert a[arm]==a['blosum_nn']
    # Snapshot was requested by the runbook before any benchmark changes.
    revision='4f99482'
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'results/benchmark/jobs','results/splits','results/benchmark/design.json'],cwd=ROOT,text=True).splitlines()
    original_jobs=0
    for path in paths:
        original=subprocess.check_output(['git','show',f'{revision}:{path}'],cwd=ROOT)
        assert hashlib.sha256(original).digest()==hashlib.sha256((ROOT/path).read_bytes()).digest(),path
        original_jobs+=path.endswith('.csv') and '/jobs/' in path
    full=pd.read_csv(out/'per_row.csv')
    assert len(full)==596715 and np.isfinite(full.y_pred).all()
    assert not full.duplicated(['arm','split_regime','fold','row_index']).any()
    record=dict(passed=True,base_checkpoint=revision,new_fits=26,total_jobs=len(list(jobs.glob('*.csv'))),
        original_jobs_unchanged=original_jobs,training_hashes_match=True,test_rows_match=True,
        eligibility_matches=True,alpha_null_and_best_epoch_integer=True,split_and_v2_design_unchanged=True,
        total_full_budget_predictions=len(full),new_gpu_extraction=False,
        scope='Full-budget only; primary five-fold t intervals; strict paired-allele bootstrap; locus exploratory')
    (out/'mlp_extension_checks.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
