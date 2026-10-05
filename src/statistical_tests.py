import pandas as pd
import numpy as np
from scipy import stats


def run_ttest(df: pd.DataFrame) -> dict:
    """Perform 2-sample T-test on days_in_hospital: readmitted vs non-readmitted.
    
    Tests whether patients who were readmitted (readmitted=1) have significantly
    different hospital stay durations compared to those not readmitted (readmitted=0).
    
    Args:
        df: Training DataFrame with 'days_in_hospital' and 'readmitted' columns.
        
    Returns:
        Dictionary with:
        - 't_statistic': The T-test statistic
        - 'p_value': Two-tailed p-value
        - 'interpretation': Plain-English interpretation of results
        - 'mean_readmitted': Mean days for readmitted patients
        - 'mean_not_readmitted': Mean days for non-readmitted patients
    """
    readmitted = df[df['readmitted'] == 1]['days_in_hospital']
    not_readmitted = df[df['readmitted'] == 0]['days_in_hospital']
    
    t_stat, p_value = stats.ttest_ind(readmitted, not_readmitted, equal_var=False)
    
    alpha = 0.05
    if p_value < alpha:
        interpretation = (
            f"The p-value ({p_value:.4f}) is less than 0.05, indicating a statistically "
            f"significant difference in days_in_hospital between readmitted and "
            f"non-readmitted patients. Readmitted patients averaged {readmitted.mean():.2f} "
            f"days vs {not_readmitted.mean():.2f} days for non-readmitted patients."
        )
    else:
        interpretation = (
            f"The p-value ({p_value:.4f}) is greater than or equal to 0.05, indicating "
            f"no statistically significant difference in days_in_hospital between "
            f"readmitted and non-readmitted patients. Readmitted patients averaged "
            f"{readmitted.mean():.2f} days vs {not_readmitted.mean():.2f} days for "
            f"non-readmitted patients."
        )
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'interpretation': interpretation,
        'mean_readmitted': readmitted.mean(),
        'mean_not_readmitted': not_readmitted.mean()
    }


def run_anova(df: pd.DataFrame) -> dict:
    """Perform One-Way ANOVA: comorbidity_score across primary_diagnosis groups.
    
    Tests whether the mean comorbidity_score differs significantly across
    different primary diagnosis categories.
    
    Args:
        df: Training DataFrame with 'comorbidity_score' and 'primary_diagnosis' columns.
        
    Returns:
        Dictionary with:
        - 'f_statistic': The F-test statistic
        - 'p_value': P-value from ANOVA
        - 'interpretation': Plain-English interpretation
        - 'group_means': Dictionary of mean comorbidity_score per diagnosis group
    """
    groups = []
    group_means = {}
    for name, group in df.groupby('primary_diagnosis')['comorbidity_score']:
        groups.append(group.values)
        group_means[name] = group.mean()
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    alpha = 0.05
    if p_value < alpha:
        interpretation = (
            f"The ANOVA p-value ({p_value:.4f}) is less than 0.05, indicating a statistically "
            f"significant difference in comorbidity_score across primary_diagnosis groups."
        )
    else:
        interpretation = (
            f"The ANOVA p-value ({p_value:.4f}) is greater than or equal to 0.05, indicating "
            f"no statistically significant difference in comorbidity_score across "
            f"primary_diagnosis groups."
        )
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'interpretation': interpretation,
        'group_means': group_means
    }


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numerical attributes.
    
    Computes correlations between: age, num_procedures, days_in_hospital,
    comorbidity_score, and readmitted.
    
    Args:
        df: Training DataFrame with numerical columns and 'readmitted'.
        
    Returns:
        Pearson correlation matrix as a DataFrame.
    """
    numerical_cols = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score', 'readmitted']
    return df[numerical_cols].corr(method='pearson')
