import statsmodels.formula.api as smf
import pandas as pd
from neural_analysis import utils
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pingouin as pg
from openpyxl.styles import PatternFill
import re
from itertools import product
from patsy import dmatrix
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from neural_analysis import plot_utils
import config

plt.rcParams['font.family'] = 'sans-serif'


def get_marginal_means_beta(model_idx, model_name_r, row_data_dict, model_strings_beta):
    """
    Get marginal means with standard errors for beta models fit in R.
    Uses manual predictions instead of emmeans to avoid contrast issues.
    """
    import re

    # Extract factors from model string
    formula = model_strings_beta[model_idx].split('+')[0]
    factors = re.findall(r'\b(?!value\b)\w+\b', formula.split('~')[1])
    factors = list(dict.fromkeys(factors))

    data = row_data_dict[model_idx + len(model_strings)]

    # Get unique levels for each factor
    for factor in factors:
        unique_levels = sorted(data[factor].dropna().unique())
        print(f"  Factor '{factor}' has {len(unique_levels)} levels: {unique_levels}")

    # Build emmeans specification
    emmeans_spec = ' + '.join(factors)

    # Try emmeans first
    try:
        ro.r(f'''
        library(emmeans)
        emm_{model_idx} <- emmeans({model_name_r}, ~ {emmeans_spec}, type = "response",
                                   mode = "asymptotic")
        emm_summary_{model_idx} <- summary(emm_{model_idx})
        ''')

        # Convert back to pandas
        with localconverter(ro.default_converter + pandas2ri.converter):
            emm_summary = ro.conversion.rpy2py(ro.r(f'emm_summary_{model_idx}'))

        # Create condition labels
        if len(factors) > 0 and all(f in emm_summary.columns for f in factors):
            condition_series = emm_summary[factors[0]].astype(str)
            for factor in factors[1:]:
                condition_series = condition_series + '_' + emm_summary[factor].astype(str)
            emm_summary['condition'] = condition_series

        # Rename columns
        rename_dict = {
            'response': 'marginal_mean',
            'prob': 'marginal_mean',
            'SE': 'se',
            'asymp.LCL': 'ci_lower',
            'asymp.UCL': 'ci_upper'
        }
        rename_dict = {k: v for k, v in rename_dict.items() if k in emm_summary.columns}
        emm_summary = emm_summary.rename(columns=rename_dict)

        return emm_summary

    except Exception as e:
        print(f"  emmeans failed: {e}")
        print(f"  Trying manual prediction approach...")

        # Manual approach: create newdata and predict
        try:
            # Get factor levels
            factor_levels = {factor: sorted(data[factor].unique()) for factor in factors}
            level_combinations = list(product(*[factor_levels[f] for f in factors]))

            # Create prediction dataframe in Python
            pred_data = pd.DataFrame(level_combinations, columns=factors)

            # Add a dummy session (use the first one from data)
            pred_data['session'] = data['session'].iloc[0]

            # Convert factor columns to categorical
            for factor in factors:
                pred_data[factor] = pd.Categorical(pred_data[factor])

            # Convert to R
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_newdata = ro.conversion.py2rpy(pred_data)
                ro.globalenv['newdata'] = r_newdata

            # Make predictions
            ro.r(f'''
            library(glmmTMB)
            # Predict on response scale
            preds <- predict({model_name_r}, newdata = newdata, type = "response", 
                           re.form = NA, se.fit = TRUE)

            # Create result dataframe
            result_df <- newdata
            result_df$marginal_mean <- preds$fit
            result_df$se <- preds$se.fit
            # Approximate confidence intervals (assuming normal on logit scale)
            result_df$ci_lower <- pmax(0, preds$fit - 1.96 * preds$se.fit)
            result_df$ci_upper <- pmin(1, preds$fit + 1.96 * preds$se.fit)
            ''')

            # Convert back
            with localconverter(ro.default_converter + pandas2ri.converter):
                result_df = ro.conversion.rpy2py(ro.r('result_df'))

            # Create condition labels
            condition_series = result_df[factors[0]].astype(str)
            for factor in factors[1:]:
                condition_series = condition_series + '_' + result_df[factor].astype(str)
            result_df['condition'] = condition_series

            # Drop session column
            result_df = result_df.drop('session', axis=1, errors='ignore')

            print(f"  Manual prediction successful!")
            return result_df

        except Exception as e2:
            print(f"  Manual prediction also failed: {e2}")
            return pd.DataFrame()


def pairwise_comparisons_beta(model_idx, model_name_r, row_data_dict, model_strings_beta,
                              alpha=0.05, adjustment='bonferroni'):
    """
    Perform pairwise posthoc comparisons for beta models.
    """
    import re
    from scipy import stats as scipy_stats

    # Extract factors
    formula = model_strings_beta[model_idx].split('+')[0]
    factors = re.findall(r'\b(?!value\b)\w+\b', formula.split('~')[1])
    factors = list(dict.fromkeys(factors))

    # Build emmeans specification
    emmeans_spec = ' + '.join(factors)

    # Map adjustment methods
    if adjustment == 'bonferroni':
        adjust_method = 'bonferroni'
    elif adjustment == 'holm':
        adjust_method = 'holm'
    elif adjustment == 'fdr_bh':
        adjust_method = 'fdr'
    else:
        adjust_method = 'none'

    try:
        # Try emmeans approach
        ro.r(f'''
        library(emmeans)
        emm_{model_idx} <- emmeans({model_name_r}, ~ {emmeans_spec}, type = "response",
                                   mode = "asymptotic")
        pairs_{model_idx} <- pairs(emm_{model_idx}, adjust = "{adjust_method}")
        pairs_summary_{model_idx} <- summary(pairs_{model_idx})
        ''')

        with localconverter(ro.default_converter + pandas2ri.converter):
            pairs_df = ro.conversion.rpy2py(ro.r(f'pairs_summary_{model_idx}'))

        # Parse contrast column
        if 'contrast' in pairs_df.columns:
            if ' / ' in str(pairs_df['contrast'].iloc[0]):
                contrasts = pairs_df['contrast'].str.split(' / ', expand=True)
            else:
                contrasts = pairs_df['contrast'].str.split(' - ', expand=True)

            pairs_df['condition_1'] = contrasts[0].str.strip()
            if contrasts.shape[1] > 1:
                pairs_df['condition_2'] = contrasts[1].str.strip()

        # Rename columns
        rename_dict = {}
        if 'ratio' in pairs_df.columns:
            rename_dict['ratio'] = 'ratio'
        if 'estimate' in pairs_df.columns:
            rename_dict['estimate'] = 'diff'
        if 'SE' in pairs_df.columns:
            rename_dict['SE'] = 'se'
        if 'z.ratio' in pairs_df.columns:
            rename_dict['z.ratio'] = 't_stat'
        if 't.ratio' in pairs_df.columns:
            rename_dict['t.ratio'] = 't_stat'
        if 'p.value' in pairs_df.columns:
            rename_dict['p.value'] = 'p_adjusted'

        pairs_df = pairs_df.rename(columns=rename_dict)

        if 'p_adjusted' in pairs_df.columns:
            pairs_df['significant'] = pairs_df['p_adjusted'] < alpha

        return pairs_df

    except Exception as e:
        print(f"  emmeans pairwise failed: {e}")
        print(f"  Trying manual pairwise comparisons...")

        # Get marginal means first
        mm_data = get_marginal_means_beta(model_idx, model_name_r, row_data_dict, model_strings_beta)

        if mm_data.empty:
            return pd.DataFrame()

        # Manual pairwise comparisons
        comparisons = []
        n = len(mm_data)

        for i in range(n):
            for j in range(i + 1, n):
                diff = mm_data.iloc[j]['marginal_mean'] - mm_data.iloc[i]['marginal_mean']
                # Approximate SE of difference (assuming independence)
                se_diff = np.sqrt(mm_data.iloc[i]['se'] ** 2 + mm_data.iloc[j]['se'] ** 2)

                # Z-test
                z_stat = diff / se_diff if se_diff > 0 else 0
                p_value = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_stat)))

                comparisons.append({
                    'condition_1': mm_data.iloc[i]['condition'],
                    'condition_2': mm_data.iloc[j]['condition'],
                    'diff': diff,
                    'se': se_diff,
                    't_stat': z_stat,
                    'p_value': p_value
                })

        pairs_df = pd.DataFrame(comparisons)

        # Apply adjustment
        if adjustment == 'bonferroni':
            pairs_df['p_adjusted'] = np.minimum(pairs_df['p_value'] * len(pairs_df), 1.0)
        elif adjustment == 'holm':
            from statsmodels.stats.multitest import multipletests
            _, p_adj, _, _ = multipletests(pairs_df['p_value'], method='holm')
            pairs_df['p_adjusted'] = p_adj
        elif adjustment == 'fdr_bh':
            from statsmodels.stats.multitest import multipletests
            _, p_adj, _, _ = multipletests(pairs_df['p_value'], method='fdr_bh')
            pairs_df['p_adjusted'] = p_adj
        else:
            pairs_df['p_adjusted'] = pairs_df['p_value']

        pairs_df['significant'] = pairs_df['p_adjusted'] < alpha

        print(f"  Manual pairwise comparisons successful!")
        return pairs_df
def count_differences(cond1, cond2):
    """Count how many factors differ between two conditions"""
    factors1 = str(cond1).split('_')
    factors2 = str(cond2).split('_')

    if len(factors1) != len(factors2):
        return float('inf')  # Different number of factors

    differences = sum(f1 != f2 for f1, f2 in zip(factors1, factors2))
    return differences


def get_marginal_means_with_inference(model_idx, results, row_data_dict, model_strings):
    """
    Get marginal means with standard errors for posthoc testing.
    """
    import re

    result = results[model_idx]
    data = row_data_dict[model_idx]

    # Extract factors
    factors = re.findall(r'\b(?!value\b)\w+\b', model_strings[model_idx].split('~')[1])
    factors = list(dict.fromkeys(factors))

    # Get unique levels for each factor
    factor_levels = {factor: sorted(data[factor].unique()) for factor in factors}
    level_combinations = list(product(*[factor_levels[f] for f in factors]))

    # Create prediction dataframe
    pred_data = pd.DataFrame(level_combinations, columns=factors)
    pred_data['session'] = data['session'].iloc[0]

    # Get design matrix for predictions using the MODEL's design info
    # This ensures column alignment with the fitted parameters

    formula_rhs = model_strings[model_idx].split('~')[1].strip()

    # Use the model's design_info if available, otherwise create new
    if hasattr(result.model, 'data') and hasattr(result.model.data, 'design_info'):
        X_pred = dmatrix(result.model.data.design_info, pred_data)
    else:
        X_pred = dmatrix(formula_rhs, pred_data)

    # Convert to dataframe and ensure column alignment
    if not isinstance(X_pred, pd.DataFrame):
        X_pred = pd.DataFrame(X_pred, columns=result.fe_params.index)

    # Align columns with the parameters in vcov
    vcov = result.cov_params()
    param_names = vcov.index.tolist()

    # Make sure X_pred has the same columns in the same order
    X_pred = X_pred.reindex(columns=param_names, fill_value=0)

    # Get predictions
    marginal_means = result.predict(pred_data)

    # Get standard errors
    standard_errors = []
    for i in range(len(X_pred)):
        x = X_pred.iloc[i].values
        var = x @ vcov.values @ x.T
        standard_errors.append(np.sqrt(var))

    pred_data['marginal_mean'] = marginal_means
    pred_data['se'] = standard_errors
    pred_data['ci_lower'] = marginal_means - 1.96 * np.array(standard_errors)
    pred_data['ci_upper'] = marginal_means + 1.96 * np.array(standard_errors)

    # Create condition labels
    condition_series = pred_data[factors[0]].astype(str)
    for factor in factors[1:]:
        condition_series = condition_series + '_' + pred_data[factor].astype(str)
    pred_data['condition'] = condition_series

    pred_data = pred_data.drop('session', axis=1)

    return pred_data, X_pred, vcov


def pairwise_comparisons(model_idx, results, row_data_dict, model_strings,
                         alpha=0.05, adjustment='bonferroni'):
    """
    Perform pairwise posthoc comparisons between all marginal means.
    """
    from scipy.stats import t

    # Get marginal means with design matrix
    mm_data, X_pred, vcov = get_marginal_means_with_inference(
        model_idx, results, row_data_dict, model_strings
    )

    result = results[model_idx]
    df_resid = result.df_resid

    # Compute all pairwise contrasts
    comparisons = []
    n_comparisons = len(mm_data)

    for i in range(n_comparisons):
        for j in range(i + 1, n_comparisons):
            # Contrast vector (difference between two conditions)
            contrast = X_pred.iloc[j].values - X_pred.iloc[i].values

            # Difference in means
            diff = mm_data.iloc[j]['marginal_mean'] - mm_data.iloc[i]['marginal_mean']

            # Standard error of difference
            # Make sure we're using .values for numpy arrays
            se_diff = np.sqrt(contrast @ vcov.values @ contrast.T)

            # t-statistic
            t_stat = diff / se_diff

            # p-value (two-tailed)
            p_value = 2 * (1 - t.cdf(np.abs(t_stat), df_resid))

            # Confidence interval
            t_crit = t.ppf(1 - alpha / 2, df_resid)
            ci_lower = diff - t_crit * se_diff
            ci_upper = diff + t_crit * se_diff

            comparisons.append({
                'condition_1': mm_data.iloc[i]['condition'],
                'condition_2': mm_data.iloc[j]['condition'],
                'mean_1': mm_data.iloc[i]['marginal_mean'],
                'mean_2': mm_data.iloc[j]['marginal_mean'],
                'diff': diff,
                'se_diff': se_diff,
                't_stat': t_stat,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

    results_df = pd.DataFrame(comparisons)

    # Apply multiple comparison adjustment
    if adjustment == 'bonferroni':
        results_df['p_adjusted'] = np.minimum(results_df['p_value'] * len(results_df), 1.0)
    elif adjustment == 'holm':
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(results_df['p_value'], method='holm')
        results_df['p_adjusted'] = p_adj
    elif adjustment == 'fdr_bh':
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_adjusted'] = p_adj
    else:
        results_df['p_adjusted'] = results_df['p_value']

    results_df['significant'] = results_df['p_adjusted'] < alpha

    return results_df


def plot_qq_for_model(model_idx, results, model_names, save_dir):
    """
    Create Q-Q plot for a specific model's residuals
    """
    residuals = results[model_idx].resid

    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot: {model_names[model_idx]}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add Shapiro-Wilk test result to plot
    stat, p = stats.shapiro(residuals)
    ax.text(0.05, 0.95, f'Shapiro-Wilk p = {p:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_dprime, f'qq_plot_model_{model_idx}.png'), dpi=300)
    plt.close()

    return stat, p


def get_significance_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'  # or return '' for no label


def plot_delta_left_right(delta_plot_dict, condition='unilateral',
                          figsize=(4, 6), alpha=0.5, test='paired', value='response rate'):
    """
    Plot delta values for left vs right responses with statistical tests.

    Parameters:
    -----------
    delta_plot_dict : dict
        Dictionary with structure as shown
    condition : str
        'unilateral' or 'bilateral'
    figsize : tuple
        Figure size
    alpha : float
        Transparency for individual points
    test : str
        'paired' for paired t-test or 'unpaired' for independent t-test
    """

    data = delta_plot_dict[condition]

    fig, ax = plt.subplots(figsize=figsize)

    # X-axis positions
    x_pos = [0, 1]  # left, right
    x_labels = ['Left', 'Right']

    # Colors for threshold and max
    color_thr = '#2E86AB'  # blue
    color_max = '#A23B72'  # purple/magenta

    # Plot threshold data
    thr_left = np.array(data['thr']['left'])
    thr_right = np.array(data['thr']['right'])

    # Individual points for threshold
    ax.scatter([0] * len(thr_left), thr_left, color=color_thr, alpha=alpha,
               s=50, label='Threshold', zorder=2)
    ax.scatter([1] * len(thr_right), thr_right, color=color_thr, alpha=alpha,
               s=50, zorder=2)

    if len(thr_left) == len(thr_right):
        for l, r in zip(thr_left, thr_right):
            ax.plot([0, 1], [l, r],
                    color=color_thr,
                    alpha=alpha,
                    linewidth=1,
                    zorder=1)

    # Mean line for threshold
    mean_thr_left = np.mean(thr_left)
    mean_thr_right = np.mean(thr_right)
    ax.plot([0, 1], [mean_thr_left, mean_thr_right],
            color=color_thr, linewidth=2.5, marker='o',
            markersize=8, zorder=3)

    # Statistical test for threshold (left vs right)
    if test == 'paired':
        t_stat_thr, p_val_thr = stats.ttest_rel(thr_left, thr_right)
    else:
        t_stat_thr, p_val_thr = stats.ttest_ind(thr_left, thr_right)

    # Wilcoxon tests for threshold (different from 0)
    stat_thr_left, p_thr_left = stats.wilcoxon(thr_left)
    stat_thr_right, p_thr_right = stats.wilcoxon(thr_right)

    # Plot max data
    max_left = np.array(data['max']['left'])
    max_right = np.array(data['max']['right'])

    # Individual points for max
    ax.scatter([0] * len(max_left), max_left, color=color_max, alpha=alpha,
               s=50, label='Max', zorder=2)
    ax.scatter([1] * len(max_right), max_right, color=color_max, alpha=alpha,
               s=50, zorder=2)

    # Connect paired max points (Left ↔ Right)
    if len(max_left) == len(max_right):
        for l, r in zip(max_left, max_right):
            ax.plot([0, 1], [l, r],
                    color=color_max,
                    alpha=alpha,
                    linewidth=1,
                    zorder=1)

    # Mean line for max
    mean_max_left = np.mean(max_left)
    mean_max_right = np.mean(max_right)
    ax.plot([0, 1], [mean_max_left, mean_max_right],
            color=color_max, linewidth=2.5, marker='o',
            markersize=8, zorder=3)

    # Statistical test for max (left vs right)
    if test == 'paired':
        t_stat_max, p_val_max = stats.ttest_rel(max_left, max_right)
    else:
        t_stat_max, p_val_max = stats.ttest_ind(max_left, max_right)

    # Wilcoxon tests for max (different from 0)
    stat_max_left, p_max_left = stats.wilcoxon(max_left)
    stat_max_right, p_max_right = stats.wilcoxon(max_right)

    # Format p-values
    def format_pval(p):
        if p < 0.001:
            return 'p < 0.001***'
        elif p < 0.01:
            return f'p = {p:.3f}**'
        elif p < 0.05:
            return f'p = {p:.3f}*'
        else:
            return f'p = {p:.3f} ns'

    # Get significance stars
    def get_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    # Add significance stars for left vs right comparison (on the connecting lines)
    stars_thr = get_stars(p_val_thr)
    stars_max = get_stars(p_val_max)

    if stars_thr:
        # Position stars at midpoint of threshold line, slightly above
        mid_y_thr = (mean_thr_left + mean_thr_right) / 2
        ax.text(0.5, mid_y_thr, stars_thr, ha='center', va='bottom',
                fontsize=16, color=color_thr, fontweight='bold')

    if stars_max:
        # Position stars at midpoint of max line, slightly above
        mid_y_max = (mean_max_left + mean_max_right) / 2
        ax.text(0.5, mid_y_max, stars_max, ha='center', va='bottom',
                fontsize=16, color=color_max, fontweight='bold')

    # Add significance stars for difference from 0 (next to each mean point)
    offset = 0.08  # horizontal offset from the point

    # Threshold left
    stars_thr_left = get_stars(p_thr_left)
    if stars_thr_left:
        ax.text(0 - offset, mean_thr_left, stars_thr_left, ha='right', va='center',
                fontsize=12, color=color_thr, fontweight='bold')

    # Threshold right
    stars_thr_right = get_stars(p_thr_right)
    if stars_thr_right:
        ax.text(1 + offset, mean_thr_right, stars_thr_right, ha='left', va='center',
                fontsize=12, color=color_thr, fontweight='bold')

    # Max left
    stars_max_left = get_stars(p_max_left)
    if stars_max_left:
        ax.text(0 - offset, mean_max_left, stars_max_left, ha='right', va='center',
                fontsize=12, color=color_max, fontweight='bold')

    # Max right
    stars_max_right = get_stars(p_max_right)
    if stars_max_right:
        ax.text(1 + offset, mean_max_right, stars_max_right, ha='left', va='center',
                fontsize=12, color=color_max, fontweight='bold')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylabel(f'Δ {value} (opto_on - opto_off)', fontsize=12)
    ax.set_xlabel('Side', fontsize=12)
    ax.set_title(f'{condition.capitalize()}: Opto Effect by Side', fontsize=14)
    ax.legend(fontsize=11, frameon=False)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Print statistics to console
    print(f"\n{condition.capitalize()} Condition Statistics:")
    print(f"Left vs Right comparison:")
    print(f"  Threshold: t = {t_stat_thr:.3f}, {format_pval(p_val_thr)}")
    print(f"  Max:       t = {t_stat_max:.3f}, {format_pval(p_val_max)}")
    print(f"\nDifference from 0 (Wilcoxon):")
    print(f"  Threshold Left:  {format_pval(p_thr_left)}")
    print(f"  Threshold Right: {format_pval(p_thr_right)}")
    print(f"  Max Left:        {format_pval(p_max_left)}")
    print(f"  Max Right:       {format_pval(p_max_right)}")

    return fig, ax


parse = True
bool_stats = True
bool_plot = True
skip_RT = False
figure_dir = config.FIGURE_DIR / "behavior/ctrl_S1"
output_path = figure_dir
stats_dir = config.FIGURE_DIR / "behavior/ctrl_S1"
data_pp_dir = config.DATA_DIR_PP

save_dir_dprime = config.FIGURE_DIR / "behavior/ctrl_S1/dprime"
save_dir_rr = config.FIGURE_DIR / "behavior/ctrl_S1/rr"

if not os.path.isdir(output_path):
    os.makedirs(output_path)

if not os.path.isdir(figure_dir):
    os.makedirs(figure_dir)

if not os.path.isdir(figure_dir / "png"):
    os.makedirs(figure_dir / "png")

if not os.path.isdir(figure_dir / "svg"):
    os.makedirs(figure_dir / "svg")

if not os.path.isdir(save_dir_dprime):
    os.makedirs(save_dir_dprime)

if not os.path.isdir(save_dir_rr):
    os.makedirs(save_dir_rr)


if not os.path.isdir(stats_dir):
    os.makedirs(stats_dir)

# Load d-primes from matlab struct (calculated with matlab exclusive method)
data = utils.load_matlab_struct(data_pp_dir / "ctrl_beh_pp.mat")
data_df = pd.DataFrame(data['output'])

# - Load raw data - #
trialData, sessionData, spikeData = utils.load_data(
    path_root=config.DATA_DIR_EXP,
    experiment=['ALM_3', 'ALM_2', 'ALM_1'],
    animals=None,
    data_types=['trialData', 'sessionData'])

# Convert categorical variables for proper handling
categorical_variables = ['hasphotostim', 'visualChangeSide', 'session_ID', 'mousename', 'optoNorm']
#trialData[categorical_variables] = trialData[categorical_variables].astype('category')

 #Filter sessions with hit-rate < 0.3 for either modality
(trialData_filtered,
 sessionData_filtered,
 spikeData_filtered,
 sessions_exclude) = utils.filter_sessions_hitrate(trialData,
                                                   sessionData=sessionData,
                                                   spikeData=None,
                                                   min_hitrate=0.3)

for session in trialData_filtered['session_ID'].unique():
    trialData_ses = trialData[trialData['session_ID'] == session]
    if len(np.abs(trialData_ses['visualOriChange']).unique()) < 3:
        print(f"Session {session} has fewer than 2 visual stimulus conditions")
        sessions_exclude.append(session)

    if len(np.abs(trialData_ses['audioOctChange']).unique()) < 3:
        print(f"Session {session} has fewer than 2 auditory stimuli conditions")
        sessions_exclude.append(session)

sessions_include = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]['session_ID']

sessions_unilat = data_df['session'][(data_df['opto'] == 'unilateral')].reset_index()
sessions_bilat = data_df['session'][(data_df['opto'] == 'bilateral')].reset_index()

data_incl_unilat = data_df[data_df['session'].isin(sessions_unilat.session)].reset_index()
data_incl_bilat = data_df[data_df['session'].isin(sessions_bilat.session)].reset_index()

data_incl = pd.concat([data_incl_unilat, data_incl_bilat], axis=0).reset_index()

df_long = utils.transform_df_long(data_incl)
df_long = df_long[~df_long['opto'].isna()]

# Plot response rate histograms to look at distribution
df_vis = df_long[(df_long['modality'] == 'visual') & (df_long['value_type'] == 'responserate')]
df_aud = df_long[(df_long['modality'] == 'auditory') & (df_long['value_type'] == 'responserate')]
data = [df_vis[df_vis['saliency'] == 'thr']['value'], df_vis[df_vis['saliency'] == 'max']['value'],
        df_aud[df_aud['saliency'] == 'thr']['value'], df_aud[df_aud['saliency'] == 'max']['value']]
plt.figure(figsize=(10, 6))

# Initialize parameter dfs for statistical models
set = pd.DataFrame()
df1 = pd.DataFrame(index=[0])
df2 = pd.DataFrame(index=[0])
df3 = pd.DataFrame(index=[0])
df4 = pd.DataFrame(index=[0])
df5 = pd.DataFrame(index=[0])
df6 = pd.DataFrame(index=[0])
df7 = pd.DataFrame(index=[0])
df8 = pd.DataFrame(index=[0])
df9 = pd.DataFrame(index=[0])
df10 = pd.DataFrame(index=[0])
df11 = pd.DataFrame(index=[0])
df12 = pd.DataFrame(index=[0])
df13 = pd.DataFrame(index=[0])
model_names = []

if bool_stats:
    ### --- Mixed LM models: Select appropriate data for each model --- ###
    # D-prime
    # Do animals perform differently on left or right sided visual stimuli? - No
    df1['value_type'] = 'prime'
    df1['modality'] = 'visual'
    df1['saliency'] = [['thr', 'max']]
    df1['opto_condition'] = False
    df1['direction'] = [['left', 'right']]
    df1['is_2ADC'] = True
    model_names.append("Visual d-prime left vs right")

    # Does opto affect auditory performance? - No
    df2['value_type'] = 'prime'
    df2['modality'] = 'auditory'
    df2['saliency'] = [['thr', 'max']]
    df2['is_2ADC'] = True
    model_names.append("Auditory d-prime")

    # Does opto affect visual performance? - Yes, both conditions + saliency:opto interaction
    df3['value_type'] = 'prime'
    df3['modality'] = 'visual'
    df3['saliency'] = [['thr', 'max']]
    df3['direction'] = [['left', 'right']]
    df3['is_2ADC'] = True
    model_names.append("Visual d-prime")

    # Does opto affect responses to left/right visual stimuli differentially? - No
    df4['value_type'] = 'prime'
    df4['modality'] = 'visual'
    df4['direction'] = [['left', 'right']]
    df4['saliency'] = [['thr', 'max']]
    df4['is_2ADC'] = True
    model_names.append("Visual d-prime opto effect left vs right")

    # Does opto affect responses to visual/auditory stimuli differentially? - Not sure
    df5['value_type'] = 'prime'
    df5['modality'] = [['visual', 'auditory']]
    df5['saliency'] = [['thr', 'max']]
    df5['direction'] = ['all']
    df5['is_2ADC'] = True
    model_names.append("Visual vs auditory opto effect")

    # Does saliency affect opto effect on d-prime in visual trials? - Yes, saliency:opto interaction for both conditions
    df6['value_type'] = 'prime'
    df6['modality'] = [['visual']]
    df6['saliency'] = [['thr', 'max']]
    df6['direction'] = ['all']
    df6['is_2ADC'] = True
    model_names.append("Visual saliency opto interaction")

    ####################
    ## Response rates ##
    ####################
    # Do animals respond differently to left or right sided visual stimuli?
    df7['value_type'] = 'responserate'
    df7['modality'] = [['visual']]
    df7['saliency'] = [['thr', 'max']]
    df7['opto_condition'] = False
    df7['direction'] = [['left', 'right']]
    df7['response_direction'] = 'left'
    model_names.append("Visual response left vs right")

    # Does opto affect auditory response rates?
    df8['value_type'] = 'responserate'
    df8['modality'] = 'auditory'
    df8['saliency'] = [['thr', 'max']]
    df8['response_direction'] = 'right'
    model_names.append("Auditory response")

    # Does opto affect visual response rates?
    df9['value_type'] = 'responserate'
    df9['modality'] = [['visual', 'none']]
    df9['saliency'] = [['probe', 'thr', 'max']]
    df9['direction'] = [['left', 'right', 'none']]
    df9['response_direction'] = 'left'
    model_names.append("Visual responses opto")

    # Does opto affect responses to left/right visual stimuli differentially?
    df10['value_type'] = 'responserate'
    df10['modality'] = 'visual'
    df10['saliency'] = [['thr', 'max']]
    df10['direction'] = [['left', 'right']]
    df10['response_direction'] = 'left'
    model_names.append("Visual responses opto left right")

    # Does opto affect responses to visual/auditory stimuli differentially?
    df11['value_type'] = 'responserate'
    df11['modality'] = [['visual', 'auditory']]
    df11['direction'] = ['all']
    df11['saliency'] = ['all']
    model_names.append("Visual vs auditory responses opto effect")

    # Does saliency affect opto effect on responserates in visual trials?
    df12['value_type'] = 'responserate'
    df12['modality'] = [['visual']]
    df12['direction'] = ['all']
    df12['response_direction'] = 'left'
    model_names.append("Visual responserates opto saliency interaction")

    # Does opto affect licking in probe trials?
    df13['value_type'] = 'responserate'
    df13['modality'] = 'none'
    model_names.append("False alarm rates under opto conditions")


    for df in [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13]:
        set = pd.concat([set, df], ignore_index=True)

    row_data_dict = {}
    row_data_df = []

    # Build dict of datasets for iteration
    for i, row in set.iterrows():
        row_data = df_long.copy()

        for col in row.index:
            if col in df_long.columns:  # Check if the column exists in `df_long`
                if isinstance(row[col], list):  # Handle the case where the value is a list (e.g., 'direction')
                    row_data = row_data[row_data[col].isin(row[col])]
                elif pd.notna(row[col]):
                    row_data = row_data[row_data[col] == row[col]]

        #plt.hist(row_data['value'], bins=30)
        #plt.show()

        row_data_dict[i] = row_data
        row_data_dict[i]['value'] = pd.to_numeric(row_data_dict[i]['value'], errors='coerce')
        row_data_df.append(pd.DataFrame(row_data_dict[i]))

    models = {}
    models_contrast = {}
    results = {}
    posthoc = {}

    # Models for d-primes (normally distributed so regular linear mixed model)
    model_strings = ["value ~ direction * saliency",
                     "value ~ saliency * opto",
                     "value ~ saliency * opto",
                     "value ~ direction * opto * saliency",
                     "value ~ modality * opto",
                     "value ~ saliency * opto"
                     ]

    # Models for response rates (generally not normally distributed, beta model instead of LMM)
    model_strings_beta = ["value ~ direction * saliency + (1|session)",
                          "value ~ saliency * opto + (1|session)",
                          "value ~ saliency * opto + (1|session)",
                          "value ~ direction * opto * saliency + (1|session)",
                          "value ~ modality * opto + (1|session)",
                          "value ~ saliency * opto + (1|session)",
                          "value ~ opto * response_direction + (1|session)"
                          ]

    ### --- Build and fit mixedLM models through a loop --- ###
    for idx, model_string in enumerate(model_strings):
        row_data = row_data_dict[idx]
        models[idx] = smf.mixedlm(model_string,
                         row_data, groups=row_data["session"])
        results[idx] = models[idx].fit()

    ### Beta models

    # Import R packages
    base = importr('base')
    glmmTMB = importr('glmmTMB')

    models_beta = {}
    models_contrast_beta = {}
    results_beta = {}
    posthoc_beta = {}

    ### --- Build and fit Beta models through a loop --- ###
    for idx, model_string in enumerate(model_strings_beta):
        row_data = row_data_dict[idx + len(model_strings)]
        row_data = row_data[['value', 'direction', 'saliency', 'opto', 'session', 'modality', 'response_direction']]

        # Convert any object columns that might contain arrays
        for col in row_data.columns:
            if row_data[col].dtype == 'object':  # String columns
                row_data[col] = row_data[col].astype('category')

        # Handle any 0s or 1s in your data (Beta can't handle exact boundaries)
        row_data = row_data.copy()
        epsilon = 1e-6
        row_data.loc[row_data['value'] == 0, 'value'] = epsilon
        row_data.loc[row_data['value'] == 1, 'value'] = 1 - epsilon

        clean_df = pd.DataFrame({
            'value': row_data['value'].values,
            'direction': pd.Categorical(row_data['direction'].values),
            'saliency': pd.Categorical(row_data['saliency'].values),
            'opto': pd.Categorical(row_data['opto'].values),
            'session': pd.Categorical(row_data['session'].values),
            'modality': pd.Categorical(row_data['modality'].values),
            'response_direction': pd.Categorical(row_data['response_direction'].values)
        })

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(clean_df)
            ro.globalenv['r_df'] = r_df

            # Fit model entirely in R
        print(f"\n=== Model {idx}: {model_strings_beta[idx]} ===")

        ro.r(f'''
               library(glmmTMB)
               model_{idx} <- glmmTMB({model_strings_beta[idx]}, 
                                      data=r_df, 
                                      family=beta_family(link='logit'))
               print(summary(model_{idx}))
               ''')

        # Store reference to the model in R's global environment
        models_beta[idx] = f'model_{idx}'

    ro.r('library(DHARMa)')


    # Beta model diagnostics
    for idx in models_beta.keys():
        if models_beta[idx] is not None:
            print(f"Saving diagnostics for Model {idx}...")

            try:
                # Do each plot separately with error handling

                # Combined diagnostic plot
                ro.r(f'''
                tryCatch({{
                    sim_res <- simulateResiduals(fittedModel = model_{idx}, plot = FALSE, n = 250)
                    png("{save_dir_rr}/model_{idx}_diagnostics.png", width=1200, height=600)
                    plot(sim_res)
                    dev.off()
                }}, error = function(e) {{
                    if (dev.cur() > 1) dev.off()
                    print(paste("Error in combined plot:", e$message))
                }})
                ''')

                # Q-Q plot only
                ro.r(f'''
                tryCatch({{
                    sim_res <- simulateResiduals(fittedModel = model_{idx}, plot = FALSE, n = 250)
                    png("{save_dir_rr}/model_{idx}_qq.png", width=800, height=600)
                    plotQQunif(sim_res)
                    dev.off()
                }}, error = function(e) {{
                    if (dev.cur() > 1) dev.off()
                    print(paste("Error in QQ plot:", e$message))
                }})
                ''')

                # Residuals vs predicted
                ro.r(f'''
                tryCatch({{
                    sim_res <- simulateResiduals(fittedModel = model_{idx}, plot = FALSE, n = 250)
                    png("{save_dir_rr}/model_{idx}_resid_vs_pred.png", width=800, height=600)
                    plotResiduals(sim_res)
                    dev.off()
                }}, error = function(e) {{
                    if (dev.cur() > 1) dev.off()
                    print(paste("Error in residual plot:", e$message))
                }})
                ''')

                print(f"  ✓ Model {idx} diagnostics saved")

            except Exception as e:
                print(f"  ✗ Model {idx} failed: {e}")
                # Make sure to close any open graphics devices
                ro.r('while (dev.cur() > 1) dev.off()')

    # Close any remaining graphics devices
    ro.r('while (dev.cur() > 1) dev.off()')

    print(f"\nAll diagnostic plots saved to '{save_dir_rr}'")

    print("\n" + "=" * 50 + "\n")

    # Extract unique factors from each model string (excluding 'value')
    for idx, model_string in enumerate(model_strings):
        # Extract factors from the model string
        factors = re.findall(r'\b(?!value\b)\w+\b', model_string.split('~')[1])
        factors = list(dict.fromkeys(factors))

        # Concatenate the columns element-wise with '_' separator
        condition_series = row_data_dict[idx][factors[0]].astype(str)
        for factor in factors[1:]:
            condition_series = condition_series + '_' + row_data_dict[idx][factor].astype(str)

        row_data_dict[idx]['condition'] = condition_series

    posthoc_dict = {}

    # Do posthoc analyses on marginal means of models (D-primes)
    for idx in range(len(model_strings)):
        print(f"\n{'=' * 60}")
        print(f"Model {idx}: {model_strings[idx]}")
        print(f"{'=' * 60}")

        try:
            # Get marginal means with SEs
            mm_data, _, _ = get_marginal_means_with_inference(idx, results, row_data_dict, model_strings)
            print("\nMarginal Means:")
            print(mm_data)

            # Pairwise comparisons
            posthoc = pairwise_comparisons(idx, results, row_data_dict, model_strings,
                                           adjustment='none')
            posthoc_dict[idx] = posthoc
            print("\nPairwise Comparisons:")
            print(posthoc[['condition_1', 'condition_2', 'diff', 'p_adjusted', 'significant']])
        except Exception as e:
            print(f"Error in model {idx}: {e}")
            import traceback

            traceback.print_exc()

    # LMM diagnostics
    # Create Q-Q plots for all models+
    print("\n" + "=" * 60)
    print("NORMALITY ASSESSMENT FOR ALL MODELS")
    print("=" * 60)

    normality_results = {}
    for idx in results.keys():
        stat, p = plot_qq_for_model(idx, results, model_names, stats_dir)
        normality_results[idx] = {'statistic': stat, 'p_value': p}

        print(f"\nModel {idx}: {model_names[idx]}")
        print(f"  he-Wilk: W = {stat:.4f}, p = {p:.4f}")
        if p > 0.05:
            print(f"  ✓ Residuals appear normally distributed")
        else:
            print(f"  ⚠ Residuals may violate normality assumption")

    # Save contrast analysis as readable excel file
    with pd.ExcelWriter(os.path.join(stats_dir, 'contrasts.xlsx'), engine='openpyxl', mode='w') as writer:
        for idx, (key, df) in enumerate(posthoc_dict.items()):
            df.to_excel(writer, sheet_name=model_names[idx], index=False)

            # Get the worksheet for highlighting
            worksheet = writer.sheets[model_names[idx]]

            # Define highlight color (yellow)
            highlight = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')

            # Highlight rows where only 1 factor differs
            # Adjust 'A' and 'B' to match your actual column names
            for row_idx, row in df.iterrows():
                if count_differences(row['condition_1'], row['condition_2']) == 1:
                    for col_idx in range(1, len(df.columns) + 1):
                        worksheet.cell(row=row_idx + 2, column=col_idx).fill = highlight

        #print(posthoc[idx])

    # Write detailed model results to a text file
    with open(os.path.join(stats_dir, 'model_results_detailed.txt'), 'w') as f:
        for idx, result in results.items():
            f.write(f"{'=' * 80}\n")
            f.write(f"MODEL {idx}: {model_names[idx]}\n")
            f.write(f"{'=' * 80}\n\n")

            # Trial selection criteria
            f.write("TRIAL SELECTION CRITERIA:\n")
            f.write("-" * 40 + "\n")
            trial_selection = set.iloc[idx]
            for col in trial_selection.index:
                value = trial_selection[col]
                # Check if value is not NaN (handles scalars, lists, etc.)
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    f.write(f"{col}: {value}\n")
            f.write(f"\n")

            # Data summary
            f.write("DATA SUMMARY:\n")
            f.write(f"Number of observations: {len(row_data_dict[idx])}\n")
            f.write(f"Number of sessions: {row_data_dict[idx]['session'].nunique()}\n")
            f.write(f"Number of animals: {row_data_dict[idx]['animal'].nunique()}\n")
            f.write(f"\n")

            # Full summary
            f.write("FULL SUMMARY:\n")
            f.write(str(result.summary()))
            f.write(f"\n\n")

            # Parameter estimates
            f.write("PARAMETER ESTIMATES:\n")
            f.write(str(result.params))
            f.write(f"\n\n")

            # P-values
            f.write("P-VALUES:\n")
            f.write(str(result.pvalues))
            f.write(f"\n\n")

            # Confidence intervals
            f.write("95% CONFIDENCE INTERVALS:\n")
            f.write(str(result.conf_int()))
            f.write(f"\n\n")

    # Beta posthoc
    posthoc_beta_dict = {}

    for idx in models_beta.keys():
        print(f"\n{'=' * 60}")
        print(f"Beta Model {idx}: {model_strings_beta[idx]}")
        print(f"{'=' * 60}")

        try:
            # Get marginal means
            mm_data = get_marginal_means_beta(idx, models_beta[idx], row_data_dict, model_strings_beta)
            if not mm_data.empty:
                print("\nMarginal Means:")
                print(mm_data[['condition', 'marginal_mean', 'se', 'ci_lower', 'ci_upper']])

            # Pairwise comparisons
            posthoc = pairwise_comparisons_beta(idx, models_beta[idx], row_data_dict, model_strings_beta,
                                                adjustment='none')
            if not posthoc.empty:
                posthoc_beta_dict[idx] = posthoc
                print("\nPairwise Comparisons:")
                # Print columns that exist
                cols_to_print = ['condition_1', 'condition_2']
                if 'ratio' in posthoc.columns:
                    cols_to_print.append('ratio')
                if 'diff' in posthoc.columns:
                    cols_to_print.append('diff')
                if 'p_adjusted' in posthoc.columns:
                    cols_to_print.append('p_adjusted')
                if 'significant' in posthoc.columns:
                    cols_to_print.append('significant')
                print(posthoc[cols_to_print])

        except Exception as e:
            print(f"Error in beta model {idx}: {e}")
            import traceback

            traceback.print_exc()

    # Save beta model contrasts to excel
    with pd.ExcelWriter(os.path.join(stats_dir, 'contrasts_beta.xlsx'), engine='openpyxl', mode='w') as writer:
        for idx, df in posthoc_beta_dict.items():
            sheet_name = f"Model_{idx + len(model_strings)}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Highlight rows where only 1 factor differs
            worksheet = writer.sheets[sheet_name]
            highlight = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')

            for row_idx, row in df.iterrows():
                # row_idx is already an integer from iterrows()
                if count_differences(row['condition_1'], row['condition_2']) == 1:
                    for col_idx in range(1, len(df.columns) + 1):
                        worksheet.cell(row=int(row_idx) + 2, column=col_idx).fill = highlight

if bool_plot:
    # Initialize subplot figures
    #fig_vis_resp_LR_opto_multi, ax_vis_resp_LR_opto_multi = plt.subplots(1, 2)
    #fig_vis_dprime_LR_opto_multi, ax_vis_dprime_LR_opto_multi = plt.subplots(1, 2)

    fig_opto, axes_opto = plt.subplots(2, 2, figsize=(3.5, 4))
    colors_vis_resp = {
        "probe":
            plot_utils.paper_colors("probe"),
        "thr":
            plot_utils.paper_colors("vis_thr"),
        "max":
            plot_utils.paper_colors("vis_max")
    }

    colors_aud_resp = {
        "probe":
            plot_utils.paper_colors("probe"),
        "thr":
            plot_utils.paper_colors("aud_thr"),
        "max":
            plot_utils.paper_colors("aud_max")
    }

    colors_vis_dprime = {
        "thr":
            plot_utils.paper_colors("vis_thr"),
        "max":
            plot_utils.paper_colors("vis_max")
    }

    colors_aud_dprime = {
        "thr":
            plot_utils.paper_colors("aud_thr"),
        "max":
            plot_utils.paper_colors("aud_max")
    }

    # Plot paired data
    ### --- Response rate plots --- ###
    # Visual responses for probe/thr/max
    print("Visual responses for probe/thr/max")
    df_long_plot = df_long[(df_long['response_direction']=='left')
                           & (df_long['opto']=='none')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    plt_vis_resp = utils.plot_pairwise_effects_multiple_conditions(df_long_plot,
                                                                   condition_column='saliency',
                                                                   conditions=['probe', 'thr', 'max'],
                                                                   animal_column='animal',
                                                                   selected_value_type='responserate')
    filename = "vis_resp"
    plt_vis_resp.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_vis_resp.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # Auditory responses for probe/thr/max
    print("Auditory responses for probe/thr/max")
    df_long_plot = df_long[(df_long['response_direction']=='right')
                           & (df_long['opto']=='none')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'auditory'])]
    plt_aud_resp = utils.plot_pairwise_effects_multiple_conditions(df_long_plot,
                                                                   condition_column='saliency',
                                                                   conditions=['probe', 'thr', 'max'],
                                                                   animal_column='animal',
                                                                   selected_value_type='responserate')
    filename = "aud_resp"
    plt_aud_resp.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_aud_resp.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')


    # ERRORS
    # Visual errors
    print("Visual errors")
    df_long_plot = df_long[(df_long['response_direction']=='right')
                           & (df_long['direction']).isin(['all'])
                           & (df_long['modality']).isin(['visual'])
                           & (df_long['saliency']=='all')]
    plt_vis_err = utils.plot_pairwise_effects_multiple_conditions(df_long_plot,
                                                                   condition_column='opto',
                                                                   conditions=['none', 'bilateral'],
                                                                   animal_column='animal',
                                                                   selected_value_type='responserate')
    filename = "vis_err"
    plt_vis_err.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_vis_err.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # Auditory errors
    print("Auditory errors")
    df_long_plot = df_long[(df_long['response_direction']=='left')
                           & (df_long['direction']).isin(['all'])
                           & (df_long['modality']).isin(['auditory'])
                           & (df_long['saliency']=='all')]
    plt_aud_err = utils.plot_pairwise_effects_multiple_conditions(df_long_plot,
                                                                   condition_column='opto',
                                                                   conditions=['none', 'bilateral'],
                                                                   animal_column='animal',
                                                                   selected_value_type='responserate')
    filename = "aud_err"
    plt_aud_err.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_aud_err.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')


    # Response rates for opto conditions - Visual
    print("Response rates for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                                 condition_column='saliency', conditions=['probe', 'thr', 'max'],
                                                 grouping_column='opto', groups=['none', 'bilateral'],
                                                 selected_value_type='responserate', alpha_marker=0.5, error_type='scatter')

    #ax = plt_vis_resp_multi_opto.gca()
    ax.set_xticklabels(['Probe', 'Threshold', 'Maximum'])
    ax.set_ylabel("Response rate")

    filename = "vis_resp_multi_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # MAIN FIGURE Response rates for opto conditions - Visual - alt
    print("Response rates for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='opto', conditions=['none', 'bilateral'],
                                         grouping_column='saliency', groups=['probe', 'thr', 'max'],
                                         selected_value_type='responserate', alpha_marker=0.5, error_type='scatter',
                                                 ax=axes_opto[0, 0], show_legend=False, group_colors=colors_vis_resp)

    #ax = plt_vis_resp_multi_opto_alt.gca()
    ax.set_xticklabels([])
    ax.set_ylabel("Response rate")
    ax.set_xlabel("")

    # Response rates for opto conditions - Visual - Left alt
    print("Response rates for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['left', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='opto', conditions=['none', 'bilateral'],
                                         grouping_column='saliency', groups=['probe', 'thr', 'max'],
                                         selected_value_type='responserate', alpha_marker=0.5, error_type='scatter')

    #ax = plt_vis_resp_left_multi_opto_alt.gca()
    ax.set_xticklabels(['None', 'Bilateral'])
    ax.set_ylabel("Response rate")

    filename = "vis_resp_left_multi_opto_alt"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # Response rates for opto conditions - Visual - Right alt
    print("Response rates for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['right', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='opto', conditions=['none', 'bilateral'],
                                         grouping_column='saliency', groups=['probe', 'thr', 'max'],
                                         selected_value_type='responserate', alpha_marker=0.5, error_type='scatter')

    #ax = plt_vis_resp_right_multi_opto_alt.gca()
    ax.set_xticklabels(['None', 'Bilateral'])
    ax.set_ylabel("Response rate")

    filename = "vis_resp_right_multi_opto_alt"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # Response rates for opto conditions - Auditory
    print("Response rates for opto conditions - Auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'auditory'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='saliency', conditions=['probe', 'thr', 'max'],
                                         grouping_column='opto', groups=['none', 'bilateral'],
                                         selected_value_type='responserate', alpha_marker=0.5, error_type='scatter')

    #ax = plt_aud_resp_multi_opto.gca()
    ax.set_xticklabels(['Probe', 'Threshold', 'Maximum'])
    ax.set_ylabel("Response rate")

    filename = "aud_resp_multi_opto"
    #fig = ax.get_figure()
    #fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    #fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # MAIN FIGURE Response rates for opto conditions - Auditory - alt
    print("Response rates for opto conditions - Auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'auditory'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='opto', conditions=['none', 'bilateral'],
                                         grouping_column='saliency', groups=['probe', 'thr', 'max'],
                                         selected_value_type='responserate', alpha_marker=0.5, error_type='scatter',
                                                 ax=axes_opto[0, 1], show_legend=False, group_colors=colors_aud_resp)

    ax.set_xticklabels([])
    ax.set_ylabel("")
    ax.set_xlabel("")

    # D-primes for opto conditions - Visual
    print("D-primes for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='saliency', conditions=['thr', 'max'],
                                         grouping_column='opto', groups=['none', 'bilateral'],
                                         selected_value_type='prime', alpha_marker=0.5, error_type='scatter')

    #ax = plt_vis_dprime_multi_opto.gca()
    ax.set_xticklabels(['Threshold', 'Maximum'])
    ax.set_ylabel("Performance (d')")

    filename = "vis_dprime_multi_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # MAIN FIGURE D-primes for opto conditions - Visual - opto on x-axis, saliency as different plots
    print("D-primes for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='opto', conditions=['none', 'bilateral'],
                                         grouping_column='saliency', groups=['thr', 'max'],
                                         selected_value_type='prime', alpha_marker=0.5, error_type='scatter',
                                                 ax=axes_opto[1, 0], show_legend=False, group_colors=colors_vis_dprime)

    #ax = plt_vis_dprime_multi_opto_alt.gca()
    ax.set_xticklabels(['Control', 'Bilateral'])
    ax.set_ylabel("Performance (d')")
    ax.set_xlabel("")

    #filename = "vis_dprime_multi_opto_alt"
    #fig = ax.get_figure()
    #fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    #fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # D-primes for opto conditions - Auditory
    print(" D-primes for opto conditions - Auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'auditory'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='saliency', conditions=['thr', 'max'],
                                         grouping_column='opto', groups=['none', 'bilateral'],
                                         selected_value_type='prime', alpha_marker=0.5, error_type='scatter')

    #ax = plt_aud_dprime_multi_opto.gca()
    ax.set_xticklabels(['Threshold', 'Maximum'])
    ax.set_ylabel("Performance (d')")

    filename = "aud_dprime_multi_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # MAIN FIGURE D-primes for opto conditions - Auditory - opto on x-axis, saliency as different plots
    print(" D-primes for opto conditions - Auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'auditory'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='opto', conditions=['none', 'bilateral'],
                                         grouping_column='saliency', groups=['thr', 'max'],
                                         selected_value_type='prime', alpha_marker=0.5, error_type='scatter',
                                                 ax=axes_opto[1, 1], show_legend=False, group_colors=colors_aud_dprime)

    #ax = plt_aud_dprime_multi_opto_alt.gca()
    ax.set_xticklabels(['Control', 'Bilateral'])
    ax.set_ylabel("")
    ax.set_xlabel("")

    #filename = "aud_dprime_multi_opto_alt"
    #fig = ax.get_figure()
    #fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    #fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # D-primes for left/right visual conditions
    print("D-primes for left/right visual conditions")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['left', 'right'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='direction', conditions=['left', 'right'],
                                         grouping_column='saliency', groups=['thr', 'max'],
                                         selected_value_type='prime', alpha_marker=0.5, error_type='scatter')

    #ax = plt_vis_direction_saliency.gca()
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_ylabel("Performance (d')")

    filename = "vis_direction_saliency"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')



    # D-primes for thr/max visual
    print("D-primes for thr/max visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])
                           & (df_long['opto_condition'] == False)]
    plt_vis_dprime_saliency = utils.plot_pairwise_effects_multiple_conditions(df_long_plot,
                                         condition_column='saliency', conditions=['thr', 'max'],
                                         animal_column='animal', selected_value_type ='prime',
                                                                              alpha_marker=0.2)

    ax = plt_vis_dprime_saliency.gca()
    ax.set_xticklabels(['Threshold', 'Maximum'])
    ax.set_ylabel("Performance (d')")

    filename = "vis_dprime_saliency"
    plt_vis_dprime_saliency.ylim((-1, 4))

    plt_vis_dprime_saliency.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_vis_dprime_saliency.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # D-primes for thr/max auditory
    print("D-primes for thr/max auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'auditory'])
                           & (df_long['opto_condition'] == False)]
    plt_aud_dprime_saliency = utils.plot_pairwise_effects_multiple_conditions(df_long_plot,
                                         condition_column='saliency', conditions=['thr', 'max'],
                                         animal_column= 'animal', selected_value_type = 'prime',
                                                                          alpha_marker=0.2)

    ax = plt_aud_dprime_saliency.gca()
    ax.set_xticklabels(['Threshold', 'Maximum'])
    ax.set_ylabel("Performance (d')")
    filename = "aud_dprime_saliency"
    plt_aud_dprime_saliency.ylim((-1, 4))

    plt_aud_dprime_saliency.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_aud_dprime_saliency.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')


    # D-primes for opto conditions - Visual (multi,  left right split)
    # Left
    print("D-primes for opto conditions - Visual left")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['left', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='saliency', conditions=['thr', 'max'],
                                         grouping_column='opto', groups=['none', 'bilateral'],
                                         selected_value_type='prime', alpha_marker=0.5, error_type='scatter')

    #ax = plt_vis_dprime_multi_left_opto.gca()
    ax.set_xticklabels(['Threshold', 'Maximum'])

    filename = "vis_dprime_multi_left_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # Left  alt
    # D-primes for opto conditions - Visual (multi,  left right split)
    print("D-primes for opto conditions - Visual left")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['left', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                         condition_column='opto', conditions=['none', 'bilateral'],
                                         grouping_column='saliency', groups=['thr', 'max'],
                                         selected_value_type='prime', alpha_marker=0.5, error_type='scatter')

    #ax = plt_vis_dprime_multi_left_opto_alt.gca()
    ax.set_xticklabels(['None', 'Bilateral'])

    filename = "vis_dprime_multi_left_opto_alt"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # Right

    print("D-primes for opto conditions - Visual right")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['right', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                                                          condition_column='saliency',
                                                                          conditions=['thr', 'max'],
                                                                          grouping_column='opto',
                                                                          groups=['none', 'bilateral'],
                                                                          selected_value_type='prime', alpha_marker=0.5,
                                                                           error_type='scatter')

    #ax = plt_vis_dprime_multi_right_opto.gca()
    ax.set_xticklabels(['Threshold', 'Maximum'])

    filename = "vis_dprime_multi_right_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # Right alt
    print("D-primes for opto conditions - Visual right")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['right', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                                                          condition_column='opto',
                                                                          conditions=['none', 'bilateral'],
                                                                          grouping_column='saliency',
                                                                          groups=['thr', 'max'],
                                                                          selected_value_type='prime',
                                                                          alpha_marker=0.5, error_type='scatter')

    #ax = plt_vis_dprime_multi_right_opto_alt.gca()
    ax.set_xticklabels(['None', 'Maximum'])

    filename = "vis_dprime_multi_right_opto_alt"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # Save combined figures
    fig_opto.subplots_adjust(hspace=0.1, wspace=0.2)
    axes_opto[0, 0].set_ylim(-0.05, 1.05)
    axes_opto[0, 1].set_ylim(-0.05, 1.05)
    axes_opto[1, 0].set_ylim(-1.05, 3.65)
    axes_opto[1, 1].set_ylim(-1.05, 3.65)

    for ax in axes_opto.flat:
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(labelsize=6)  # Tick labels
        ax.xaxis.label.set_size(6)  # X-axis label
        ax.yaxis.label.set_size(6)  # Y-axis label
        ax.title.set_size(8)  # Title (if you have one)
        # Method 1: Adjust tick parameters to bring labels closer
        ax.tick_params(axis='both', which='major', pad=2)  # Default is usually 4-6

        # Method 2: Shorten tick length
        ax.tick_params(axis='both', which='major', length=3)  # Default is usually 5-7

    #fig_opto.tight_layout()
    fig_opto.savefig(os.path.join(figure_dir, 'png', "resp_dprimes_vis_aud_opto.png"), format='png')
    fig_opto.savefig(os.path.join(figure_dir, 'svg', "resp_dprimes_vis_aud_opto.svg"), format='svg')

    fig_crit, axes_crit = plt.subplots(1, 2, figsize=(3.5, 2))

    fig_crit, axes_crit = plt.subplots(1, 2, figsize=(3.5, 2))

    # Visual criterion (session-level, no saliency split)
    print("Criterion for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['visual'])
                           & (df_long['saliency'] == 'all')]
    utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                            condition_column='opto', conditions=['none', 'bilateral'],
                                            grouping_column='saliency', groups=['all'],
                                            selected_value_type='c', alpha_marker=0.5, error_type='scatter',
                                            ax=axes_crit[0], show_legend=False)
    axes_crit[0].set_xticklabels(['Control', 'Bilateral'])
    axes_crit[0].set_ylabel("Criterion (c)")
    axes_crit[0].set_xlabel("")

    # Auditory criterion (session-level, no saliency split)
    print("Criterion for opto conditions - Auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'auditory'])
                           & (df_long['saliency'] == 'all')]
    utils.plot_mean_lines_with_errorbars_ax(df_long_plot,
                                            condition_column='opto', conditions=['none', 'bilateral'],
                                            grouping_column='saliency', groups=['all'],
                                            selected_value_type='c', alpha_marker=0.5, error_type='scatter',
                                            ax=axes_crit[1], show_legend=False)
    axes_crit[1].set_xticklabels(['Control', 'Bilateral'])
    axes_crit[1].set_ylabel("")
    axes_crit[1].set_xlabel("")

    for ax in axes_crit.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(6)
        ax.tick_params(axis='both', which='major', pad=2, length=3)

    fig_crit.subplots_adjust(wspace=0.2)
    fig_crit.savefig(os.path.join(figure_dir, 'png', "criterion_vis_aud_opto.png"), format='png')
    fig_crit.savefig(os.path.join(figure_dir, 'svg', "criterion_vis_aud_opto.svg"), format='svg')

    # For every session, calculate delta (D-prime_opto_off - D-prime_opto_on) for thr and max
    for value_type in ['prime', 'responserate']:
        delta_plot_dict = {
            'unilateral': {
                'thr': {
                    'left': [],
                    'right': []
                },
                'max': {
                    'left': [],
                    'right': []
                }
            },

            'bilateral': {
                'thr': {
                    'left': [],
                    'right': []
                },
                'max': {
                    'left': [],
                    'right': []
                }
            }

        }

        if value_type == 'prime':
            df_long_plot = df_long[(df_long['response_direction'] == 'left')
                                   & (df_long['direction']).isin(['left', 'right'])
                                   & (df_long['is_2ADC'] == True)
                                   & (df_long['modality'] == 'visual')]
        else:
            df_long_plot = df_long[(df_long['response_direction'] == 'left')
                                   & (df_long['direction']).isin(['left', 'right'])
                                   & (df_long['modality'] == 'visual')]

        for opto in ['unilateral', 'bilateral']:
            for saliency in ['thr', 'max']:
                for direction in ['left', 'right']:
                    for session in df_long_plot['session'].unique():
                        opto_off_value = df_long_plot[(df_long_plot['session'] == session)
                                                      & (df_long_plot['opto'] == 'none')
                                                      & (df_long_plot['saliency'] == saliency)
                                                      & (df_long_plot['direction'] == direction)
                                                      & (df_long_plot['value_type'] == value_type)]['value'].item()

                        opto_on_df = df_long_plot[(df_long_plot['session'] == session)
                                                      & (df_long_plot['opto'] == opto)
                                                      & (df_long_plot['saliency'] == saliency)
                                                      & (df_long_plot['direction'] == direction)
                                                      & (df_long_plot['value_type'] == value_type)]

                        if len(opto_on_df) == 0:
                            continue
                        else:
                            opto_on_value = opto_on_df['value'].item()

                        delta = opto_on_value - opto_off_value

                        delta_plot_dict[opto][saliency][direction].append(delta)

            fig, ax = plot_delta_left_right(delta_plot_dict, condition=opto, value=value_type)
            filename = f"vis_{value_type}_opto_delta_left_right_" + opto
            plt.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
            plt.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

