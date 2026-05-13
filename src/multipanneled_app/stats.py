from __future__ import annotations

import pandas as pd
from scipy import stats as sp_stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    g = df.groupby(group_col)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / out["count"].pow(0.5)
    return out


def pearson_correlation(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    clean = df[[x, y]].dropna()
    r, p = sp_stats.pearsonr(clean[x], clean[y])
    return pd.DataFrame({"metric": ["pearson_r", "p_value", "n"], "value": [r, p, len(clean)]})


def regression_stats(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    clean = df[[x, y]].dropna()
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(clean[x], clean[y])
    return pd.DataFrame(
        {
            "metric": ["slope", "intercept", "r", "r_squared", "p_value", "std_err", "n"],
            "value": [slope, intercept, r_value, r_value**2, p_value, std_err, len(clean)],
        }
    )


def anova_one_way(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    model = ols(f"{value_col} ~ C({group_col})", data=df.dropna(subset=[group_col, value_col])).fit()
    return anova_lm(model, typ=2).reset_index(names="source")


def tukey_hsd(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    clean = df.dropna(subset=[group_col, value_col])
    result = pairwise_tukeyhsd(clean[value_col], clean[group_col])
    table = pd.DataFrame(result._results_table.data[1:], columns=result._results_table.data[0])
    return table


def repeated_measures_corr(df: pd.DataFrame, subject: str, x: str, y: str) -> pd.DataFrame:
    try:
        import pingouin as pg
    except Exception:
        return pd.DataFrame({"warning": ["pingouin is not installed; repeated-measures correlation unavailable"]})
    clean = df[[subject, x, y]].dropna()
    return pg.rm_corr(data=clean, subject=subject, x=x, y=y)
