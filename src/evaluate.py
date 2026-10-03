import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


def spearman(y_true, y_pred):
    if len(y_true) < 3 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float('nan')
    return float(spearmanr(y_true, y_pred).statistic)


def pearson(y_true, y_pred):
    if len(y_true) < 3 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float('nan')
    return float(pearsonr(y_true, y_pred).statistic)


def rmse_log1p(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - y_pred) ** 2)))


def tier_auc(y_true_hours, y_pred, threshold):
    label = np.asarray(y_true_hours) >= threshold
    return float(roc_auc_score(label, y_pred)) if len(np.unique(label)) == 2 else float('nan')


def metrics(y_true, y_pred, alleles=None):
    # Compare in log space so exact boundary values survive exp/log rounding.
    result = dict(spearman=spearman(y_true, y_pred), pearson=pearson(y_true, y_pred),
                  rmse_log1p=rmse_log1p(y_true, y_pred),
                  auc_2h=tier_auc(y_true, y_pred, np.log1p(2)),
                  auc_6h=tier_auc(y_true, y_pred, np.log1p(6)))
    if alleles is not None:
        frame = pd.DataFrame(dict(y=y_true, p=y_pred, allele=np.asarray(alleles)))
        values = [spearman(g.y, g.p) for _, g in frame.groupby('allele')]
        result['macro_allele_spearman'] = float(np.nanmean(values))
    return result


def stratify_by_allele_distance(per_row_results, allele_distance):
    frame = per_row_results.merge(allele_distance, on=['allele', 'fold'], validate='many_to_one')
    frame['identity_band'] = pd.cut(frame.max_identity_to_train_allele,
                                    [0, .8, .9, .95, 1.0001], right=False,
                                    labels=['<80%', '80–90%', '90–95%', '95–100%'])
    rows = []
    for (arm, band), group in frame.groupby(['arm', 'identity_band'], observed=True):
        rows.append(dict(arm=arm, identity_band=band, rows=len(group), alleles=group.allele.nunique(),
                         **metrics(group.y_true, group.y_pred, group.allele)))
    return pd.DataFrame(rows)
