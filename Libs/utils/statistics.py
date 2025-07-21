"""Enhanced Statistical Analysis Module for ICLR-quality Results Reporting.

This module provides comprehensive statistical tools for medical AI research:
• Bootstrap confidence intervals with bias correction
• Multiple comparison corrections (Bonferroni, Holm-Sidak, FDR)
• Statistical significance testing with effect size reporting
• Publication-ready table generation with LaTeX output
• Meta-analysis tools for aggregating results across experiments
• Clinical trial statistical methods adapted for offline evaluation

Design principles:
• Rigorous statistical methodology following CONSORT guidelines
• Comprehensive uncertainty quantification for medical applications
• Automated multiple testing corrections to control family-wise error rates
• Publication-ready formatting for academic submission
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "bootstrap_confidence_interval",
    "statistical_comparison", 
    "multiple_testing_correction",
    "effect_size_analysis",
    "create_results_table",
    "format_latex_table",
    "clinical_significance_test",
    "meta_analysis_summary"
]

class StatisticalAnalyzer:
    """Comprehensive statistical analyzer for offline RL evaluation results."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
        correction_method: str = "holm",
        random_state: int = 42
    ):
        """Initialize statistical analyzer.
        
        Args:
            alpha: Significance level (default 0.05)
            n_bootstrap: Number of bootstrap samples
            correction_method: Multiple testing correction method
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap 
        self.correction_method = correction_method
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic: callable = np.mean,
        confidence_level: float = 0.95,
        method: str = "bca"  # bias-corrected and accelerated
    ) -> Tuple[float, Tuple[float, float]]:
        """Compute bootstrap confidence interval with bias correction.
        
        Args:
            data: Input data array
            statistic: Function to compute statistic (default: mean)
            confidence_level: Confidence level (default: 0.95)
            method: Bootstrap method ("bca", "percentile", "basic")
            
        Returns:
            Tuple of (point_estimate, (lower_bound, upper_bound))
        """
        if len(data) == 0:
            return np.nan, (np.nan, np.nan)
            
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return np.nan, (np.nan, np.nan)
            
        point_estimate = statistic(clean_data)
        
        try:
            # Use scipy.stats.bootstrap for robust implementation
            res = bootstrap(
                (clean_data,), 
                statistic, 
                n_resamples=self.n_bootstrap,
                confidence_level=confidence_level,
                method=method,
                random_state=self.rng
            )
            
            return point_estimate, (res.confidence_interval.low, res.confidence_interval.high)
            
        except Exception as e:
            logger.warning(f"Bootstrap failed: {e}, using normal approximation")
            # Fallback to normal approximation
            std_err = np.std(clean_data) / np.sqrt(len(clean_data))
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            margin = z_score * std_err
            
            return point_estimate, (point_estimate - margin, point_estimate + margin)
    
    def statistical_comparison(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        test_type: str = "auto",
        paired: bool = False
    ) -> Dict[str, Any]:
        """Perform statistical comparison between two groups.
        
        Args:
            group1: First group data
            group2: Second group data  
            test_type: Type of test ("auto", "t-test", "wilcoxon", "mann-whitney")
            paired: Whether samples are paired
            
        Returns:
            Dictionary containing test results
        """
        # Clean data
        g1_clean = group1[~np.isnan(group1)]
        g2_clean = group2[~np.isnan(group2)]
        
        if len(g1_clean) == 0 or len(g2_clean) == 0:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "test_name": "insufficient_data",
                "effect_size": np.nan,
                "ci_diff": (np.nan, np.nan)
            }
        
        # Determine test type automatically
        if test_type == "auto":
            # Check normality using Shapiro-Wilk test
            if len(g1_clean) >= 3 and len(g2_clean) >= 3:
                _, p1 = stats.shapiro(g1_clean)
                _, p2 = stats.shapiro(g2_clean)
                is_normal = (p1 > 0.05) and (p2 > 0.05)
                
                # Check equal variances
                if is_normal:
                    _, p_var = stats.levene(g1_clean, g2_clean)
                    equal_var = p_var > 0.05
                else:
                    equal_var = False
                    
                # Select appropriate test
                if is_normal and equal_var:
                    test_type = "t-test"
                elif paired:
                    test_type = "wilcoxon"
                else:
                    test_type = "mann-whitney"
            else:
                test_type = "mann-whitney"
        
        # Perform the statistical test
        result = {"test_name": test_type}
        
        try:
            if test_type == "t-test":
                if paired:
                    statistic, p_value = stats.ttest_rel(g1_clean, g2_clean)
                else:
                    statistic, p_value = stats.ttest_ind(g1_clean, g2_clean)
                    
            elif test_type == "wilcoxon" and paired:
                statistic, p_value = stats.wilcoxon(g1_clean, g2_clean)
                
            elif test_type == "mann-whitney":
                statistic, p_value = stats.mannwhitneyu(g1_clean, g2_clean, alternative='two-sided')
                
            else:
                raise ValueError(f"Unknown test type: {test_type}")
                
            result.update({
                "statistic": float(statistic),
                "p_value": float(p_value)
            })
            
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            result.update({
                "statistic": np.nan,
                "p_value": np.nan
            })
        
        # Calculate effect size
        result["effect_size"] = self._calculate_effect_size(g1_clean, g2_clean, test_type)
        
        # Calculate confidence interval for difference
        diff_data = g1_clean - g2_clean if paired and len(g1_clean) == len(g2_clean) else None
        if diff_data is not None:
            _, ci_diff = self.bootstrap_confidence_interval(diff_data)
            result["ci_diff"] = ci_diff
        else:
            # Use bootstrap for unpaired data
            combined_bootstrap = []
            for _ in range(1000):
                b1 = self.rng.choice(g1_clean, size=len(g1_clean), replace=True)
                b2 = self.rng.choice(g2_clean, size=len(g2_clean), replace=True)
                combined_bootstrap.append(np.mean(b1) - np.mean(b2))
            result["ci_diff"] = tuple(np.percentile(combined_bootstrap, [2.5, 97.5]))
        
        return result
    
    def _calculate_effect_size(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray, 
        test_type: str
    ) -> float:
        """Calculate appropriate effect size measure."""
        try:
            if test_type == "t-test":
                # Cohen's d
                pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                    (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                   (len(group1) + len(group2) - 2))
                if pooled_std == 0:
                    return 0.0
                return (np.mean(group1) - np.mean(group2)) / pooled_std
                
            else:
                # Glass's rank-biserial correlation for non-parametric tests
                n1, n2 = len(group1), len(group2)
                U, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                return 1 - (2 * U) / (n1 * n2)
                
        except Exception:
            return np.nan
    
    def multiple_testing_correction(
        self, 
        p_values: List[float], 
        method: Optional[str] = None
    ) -> Tuple[List[bool], List[float]]:
        """Apply multiple testing correction to p-values.
        
        Args:
            p_values: List of uncorrected p-values
            method: Correction method (None uses instance default)
            
        Returns:
            Tuple of (rejected_hypotheses, corrected_p_values)
        """
        if method is None:
            method = self.correction_method
            
        p_array = np.array(p_values)
        n_tests = len(p_array)
        
        if method == "bonferroni":
            corrected_p = np.minimum(p_array * n_tests, 1.0)
            rejected = corrected_p <= self.alpha
            
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            corrected_p = np.full(n_tests, 1.0)
            rejected = np.full(n_tests, False)
            
            for i, p in enumerate(sorted_p):
                correction_factor = n_tests - i
                corrected_p[sorted_indices[i]] = min(p * correction_factor, 1.0)
                
                if p * correction_factor <= self.alpha:
                    rejected[sorted_indices[i]] = True
                else:
                    break  # Stop at first non-significant result
                    
        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            corrected_p = np.full(n_tests, 1.0)
            rejected = np.full(n_tests, False)
            
            for i in range(n_tests - 1, -1, -1):
                correction_factor = n_tests / (i + 1)
                corrected_p[sorted_indices[i]] = min(sorted_p[i] * correction_factor, 1.0)
                
                if sorted_p[i] * correction_factor <= self.alpha:
                    rejected[sorted_indices[:i+1]] = True
                    break
                    
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return rejected.tolist(), corrected_p.tolist()

def create_results_table(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: List[str] = ["mean", "std", "ci_lower", "ci_upper"],
    format_precision: int = 3,
    include_significance: bool = True,
    baseline_method: Optional[str] = None
) -> pd.DataFrame:
    """Create a publication-ready results table.
    
    Args:
        results_dict: Nested dict {method: {metric: value}}
        metrics: List of metrics to include in table
        format_precision: Number of decimal places
        include_significance: Whether to include significance markers
        baseline_method: Method to use as baseline for comparisons
        
    Returns:
        Formatted pandas DataFrame
    """
    analyzer = StatisticalAnalyzer()
    
    # Create base dataframe
    df = pd.DataFrame(results_dict).T
    
    # Format numerical columns
    for metric in metrics:
        if metric in df.columns:
            df[metric] = df[metric].apply(
                lambda x: f"{x:.{format_precision}f}" if pd.notna(x) else "—"
            )
    
    # Create confidence interval column if bounds are available
    if "ci_lower" in df.columns and "ci_upper" in df.columns:
        df["95% CI"] = df.apply(
            lambda row: f"[{row['ci_lower']}, {row['ci_upper']}]" 
            if pd.notna(row.get('ci_lower')) and pd.notna(row.get('ci_upper'))
            else "—", axis=1
        )
    
    # Add significance markers if requested
    if include_significance and baseline_method is not None and baseline_method in results_dict:
        significance_markers = []
        p_values = []
        
        baseline_data = results_dict[baseline_method].get('raw_data', [])
        
        for method in df.index:
            if method == baseline_method:
                significance_markers.append("†")  # Baseline marker
                p_values.append(np.nan)
            else:
                method_data = results_dict[method].get('raw_data', [])
                if len(baseline_data) > 0 and len(method_data) > 0:
                    comparison = analyzer.statistical_comparison(
                        np.array(method_data), 
                        np.array(baseline_data)
                    )
                    p_val = comparison['p_value']
                    p_values.append(p_val)
                    
                    # Add significance markers
                    if pd.notna(p_val):
                        if p_val < 0.001:
                            significance_markers.append("***")
                        elif p_val < 0.01:
                            significance_markers.append("**")
                        elif p_val < 0.05:
                            significance_markers.append("*")
                        else:
                            significance_markers.append("")
                    else:
                        significance_markers.append("")
                else:
                    significance_markers.append("")
                    p_values.append(np.nan)
        
        # Apply multiple testing correction
        valid_p_values = [p for p in p_values if pd.notna(p)]
        if len(valid_p_values) > 1:
            rejected, corrected_p = analyzer.multiple_testing_correction(valid_p_values)
            
            # Update significance markers based on corrected p-values
            corrected_idx = 0
            for i, p_val in enumerate(p_values):
                if pd.notna(p_val):
                    corrected_p_val = corrected_p[corrected_idx]
                    if rejected[corrected_idx]:
                        if corrected_p_val < 0.001:
                            significance_markers[i] += "†††"
                        elif corrected_p_val < 0.01:
                            significance_markers[i] += "††"
                        elif corrected_p_val < 0.05:
                            significance_markers[i] += "†"
                    corrected_idx += 1
        
        df["Significance"] = significance_markers
    
    # Clean up column order
    desired_order = ["mean", "std", "95% CI", "Significance"]
    available_cols = [col for col in desired_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in desired_order]
    df = df[available_cols + other_cols]
    
    return df

def format_latex_table(
    df: pd.DataFrame,
    caption: str = "Results Comparison",
    label: str = "tab:results",
    position: str = "htbp"
) -> str:
    """Format DataFrame as LaTeX table for publication.
    
    Args:
        df: Pandas DataFrame to format
        caption: Table caption
        label: LaTeX label for referencing
        position: Table position specifier
        
    Returns:
        LaTeX table string
    """
    # Start LaTeX table
    n_cols = len(df.columns)
    col_spec = "l" + "c" * (n_cols - 1)  # Left-align first column, center others
    
    latex_str = f"""\\begin{{table}}[{position}]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
"""
    
    # Add header
    header = " & ".join([f"\\textbf{{{col}}}" for col in df.columns])
    latex_str += header + " \\\\\n\\midrule\n"
    
    # Add data rows
    for idx, row in df.iterrows():
        row_str = f"\\textbf{{{idx}}}"  # Bold method name
        for col in df.columns:
            value = str(row[col])
            # Escape special LaTeX characters
            value = value.replace("_", "\\_").replace("%", "\\%")
            row_str += f" & {value}"
        latex_str += row_str + " \\\\\n"
    
    # Close table
    latex_str += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_str

# Convenience functions
def bootstrap_confidence_interval(
    data: Union[np.ndarray, List[float]], 
    confidence_level: float = 0.95,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000
) -> Tuple[float, Tuple[float, float]]:
    """Convenience function for bootstrap CI calculation."""
    analyzer = StatisticalAnalyzer(n_bootstrap=n_bootstrap)
    return analyzer.bootstrap_confidence_interval(
        np.array(data), statistic, confidence_level
    )

def statistical_comparison(
    group1: Union[np.ndarray, List[float]],
    group2: Union[np.ndarray, List[float]],
    test_type: str = "auto"
) -> Dict[str, Any]:
    """Convenience function for statistical comparison."""
    analyzer = StatisticalAnalyzer()
    return analyzer.statistical_comparison(
        np.array(group1), np.array(group2), test_type
    )

def multiple_testing_correction(
    p_values: List[float],
    method: str = "holm", 
    alpha: float = 0.05
) -> Tuple[List[bool], List[float]]:
    """Convenience function for multiple testing correction."""
    analyzer = StatisticalAnalyzer(alpha=alpha, correction_method=method)
    return analyzer.multiple_testing_correction(p_values)

def effect_size_analysis(
    group1: Union[np.ndarray, List[float]],
    group2: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """Calculate comprehensive effect size measures."""
    g1, g2 = np.array(group1), np.array(group2)
    
    # Cohen's d
    pooled_std = np.sqrt(((len(g1) - 1) * np.var(g1, ddof=1) + 
                         (len(g2) - 1) * np.var(g2, ddof=1)) / 
                        (len(g1) + len(g2) - 2))
    cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
    
    # Glass's delta (using control group std)
    glass_delta = (np.mean(g1) - np.mean(g2)) / np.std(g2, ddof=1) if np.std(g2, ddof=1) > 0 else 0
    
    # Hedge's g (bias-corrected Cohen's d)
    correction_factor = 1 - 3 / (4 * (len(g1) + len(g2)) - 9)
    hedges_g = cohens_d * correction_factor
    
    return {
        "cohens_d": cohens_d,
        "glass_delta": glass_delta, 
        "hedges_g": hedges_g,
        "interpretation": _interpret_effect_size(abs(cohens_d))
    }

def _interpret_effect_size(effect_size: float) -> str:
    """Interpret effect size magnitude according to Cohen's conventions."""
    if effect_size < 0.2:
        return "negligible"
    elif effect_size < 0.5:
        return "small"
    elif effect_size < 0.8:
        return "medium"
    else:
        return "large"

def clinical_significance_test(
    treatment_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    minimal_important_difference: float,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """Test for clinical significance beyond statistical significance.
    
    Args:
        treatment_outcomes: Treatment group outcomes
        control_outcomes: Control group outcomes  
        minimal_important_difference: Clinically meaningful difference threshold
        confidence_level: Confidence level for CI
        
    Returns:
        Dictionary with clinical significance analysis
    """
    analyzer = StatisticalAnalyzer()
    
    # Statistical comparison
    stat_result = analyzer.statistical_comparison(treatment_outcomes, control_outcomes)
    
    # Effect size
    effect_result = effect_size_analysis(treatment_outcomes, control_outcomes)
    
    # Confidence interval for difference
    diff_data = treatment_outcomes - control_outcomes if len(treatment_outcomes) == len(control_outcomes) else None
    if diff_data is not None:
        mean_diff, ci_diff = analyzer.bootstrap_confidence_interval(diff_data, confidence_level)
    else:
        mean_diff = np.mean(treatment_outcomes) - np.mean(control_outcomes)
        ci_diff = stat_result["ci_diff"]
    
    # Clinical significance assessment
    clinically_significant = abs(mean_diff) >= minimal_important_difference
    ci_excludes_null = (ci_diff[0] > minimal_important_difference or ci_diff[1] < -minimal_important_difference)
    
    return {
        "mean_difference": mean_diff,
        "ci_difference": ci_diff,
        "minimal_important_difference": minimal_important_difference,
        "statistically_significant": stat_result["p_value"] < 0.05,
        "clinically_significant": clinically_significant,
        "ci_excludes_clinically_null": ci_excludes_null,
        "effect_size": effect_result,
        "statistical_test": stat_result,
        "interpretation": _interpret_clinical_significance(
            stat_result["p_value"] < 0.05, 
            clinically_significant, 
            ci_excludes_null
        )
    }

def _interpret_clinical_significance(
    stat_sig: bool, 
    clin_sig: bool, 
    ci_excludes_null: bool
) -> str:
    """Interpret combined statistical and clinical significance."""
    if stat_sig and clin_sig and ci_excludes_null:
        return "Both statistically and clinically significant"
    elif stat_sig and clin_sig:
        return "Statistically and clinically significant, but CI includes null"
    elif stat_sig and not clin_sig:
        return "Statistically significant but not clinically meaningful"
    elif not stat_sig and clin_sig:
        return "Clinically meaningful difference but not statistically significant"
    else:
        return "Neither statistically nor clinically significant"

def meta_analysis_summary(
    study_results: List[Dict[str, float]],
    study_weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Perform basic meta-analysis of multiple study results.

    Args:
        study_results: List of dicts with 'mean', 'std', 'n' for each study
        study_weights: Optional weights for each study

    Returns:
        Meta-analysis summary
    """
    if not study_results:
        return {}
    
    means = np.array([s['mean'] for s in study_results])
    stds = np.array([s['std'] for s in study_results])
    ns = np.array([s['n'] for s in study_results])
    
    # Calculate weights (inverse variance if not provided)
    if study_weights is None:
        variances = stds**2 / ns
        weights = 1 / variances
    else:
        weights = np.array(study_weights)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Weighted mean and variance
    pooled_mean = np.sum(weights * means)
    pooled_var = np.sum(weights**2 * (stds**2 / ns))
    pooled_std = np.sqrt(pooled_var)
    
    # Heterogeneity assessment (Q statistic)
    q_stat = np.sum(weights * (means - pooled_mean)**2)
    df = len(study_results) - 1
    p_heterogeneity = 1 - stats.chi2.cdf(q_stat, df) if df > 0 else 1.0
    
    # I-squared statistic
    i_squared = max(0, (q_stat - df) / q_stat) if q_stat > 0 else 0
    
    return {
        "pooled_mean": pooled_mean,
        "pooled_std": pooled_std,
        "pooled_ci": (
            pooled_mean - 1.96 * pooled_std,
            pooled_mean + 1.96 * pooled_std
        ),
        "n_studies": len(study_results),
        "total_n": int(np.sum(ns)),
        "q_statistic": q_stat,
        "p_heterogeneity": p_heterogeneity,
        "i_squared": i_squared,
        "heterogeneity_interpretation": _interpret_heterogeneity(i_squared)
    }

def _interpret_heterogeneity(i_squared: float) -> str:
    """Interpret I-squared heterogeneity statistic."""
    if i_squared < 0.25:
        return "Low heterogeneity"
    elif i_squared < 0.50:
        return "Moderate heterogeneity"
    elif i_squared < 0.75:
        return "Substantial heterogeneity"
    else:
        return "Considerable heterogeneity" 