"""
Reaction time analysis using an R-based pipeline

Model: Gaussian LMM on log(RT). Handles the right-skew and positivity of RTs 
without committing to a specific bounded support

Marginal means and contrasts are computed on the log scale by emmeans,
then back-transformed to milliseconds via type = "response", which yields
geometric means on the original scale. Pairwise contrasts are reported as
RATIOS of geometric means (not differences) because differences on the
log scale exponentiate to ratios. A ratio of 0.90 means the comparison
condition is 10% faster than the baseline.
"""

import os
import re
import warnings
from pathlib import Path
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from openpyxl.styles import PatternFill, Font, Alignment

from neural_analysis import utils

# Re-use the helper functions from your behavioural script. If they're in
# a module, import them. Otherwise, paste them into this file or run this
# after the behavioural script in the same session.
#
# Required helpers (all defined in your existing pipeline):
#   _warn_fallback
#   fit_lmm_r
#   omnibus_tests_r
#   get_marginal_means_r
#   pairwise_comparisons_r
#   _enforce_opto_sign_convention
#   get_model_fit_stats
#   get_fixed_effects_r
#   diagnostics_r
#   count_differences
#   _safe_sheet_name
#   _write_section


def _warn_fallback(message):
    """Emit a prominent warning when a manual fallback is triggered."""
    banner = "!" * 78
    full_message = (
        f"\n{banner}\n"
        f"!! MANUAL FALLBACK TRIGGERED — RESULTS ARE APPROXIMATE !!\n"
        f"!! {message}\n"
        f"!! Inspect output carefully before reporting in the paper.\n"
        f"{banner}\n"
    )
    print(full_message)
    warnings.warn(full_message, UserWarning, stacklevel=2)


# =============================================================================
# UNIFIED R-BASED MODEL FITTING
# =============================================================================

def fit_lmm_r(model_idx, fixed_formula, random_formula, data, factors_to_categorize):
    """
    Fit a Gaussian linear mixed model in R using lmerTest.

    Parameters
    ----------
    model_idx : int
        Index used to name the R object (model_<idx>).
    fixed_formula : str
        Fixed-effects part, e.g. 'value ~ direction * opto * saliency'.
    random_formula : str
        Random-effects part, e.g. '(1|session)'.
    data : pd.DataFrame
        Long-format data.
    factors_to_categorize : list of str
        Columns to convert to categorical before passing to R.

    Returns
    -------
    str : the R object name (e.g. 'model_0').
    """
    clean_df = data.copy()
    for col in factors_to_categorize:
        if col in clean_df.columns:
            clean_df[col] = pd.Categorical(clean_df[col])

    # Keep only the columns we need to keep R's data frame light
    keep_cols = ['value'] + [c for c in factors_to_categorize if c in clean_df.columns]
    clean_df = clean_df[keep_cols].dropna().reset_index(drop=True)

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(clean_df)
        ro.globalenv[f'lmm_df_{model_idx}'] = r_df

    full_formula = f"{fixed_formula} + {random_formula}"
    model_name_r = f'model_{model_idx}'

    ro.r(f'''
        library(lmerTest)
        {model_name_r} <- lmer({full_formula},
                               data = lmm_df_{model_idx},
                               REML = TRUE,
                               control = lmerControl(optimizer = "bobyqa",
                                                     optCtrl = list(maxfun = 2e5)))
        cat("\\n=== Model {model_idx}: {full_formula} ===\\n")
        print(summary({model_name_r}))
    ''')

    # Check convergence
    converged = bool(ro.r(f'''
        is.null({model_name_r}@optinfo$conv$lme4$messages) ||
        length({model_name_r}@optinfo$conv$lme4$messages) == 0
    ''')[0])

    if not converged:
        _warn_fallback(f"LMM model {model_idx} did not converge cleanly.")

    return model_name_r


def fit_beta_r(model_idx, fixed_formula, random_formula, data, factors_to_categorize,
               family='ordbeta'):
    """
    Fit a beta or ordered-beta mixed model in R using glmmTMB.

    Parameters
    ----------
    family : {'ordbeta', 'beta_smithson'}
        'ordbeta' uses ordered beta regression (Kubinec 2023) which natively
        handles 0 and 1 boundaries — preferred.
        'beta_smithson' uses standard beta regression with the Smithson &
        Verkuilen (2006) transformation y' = (y*(n-1) + 0.5)/n.
    """
    clean_df = data.copy()
    for col in factors_to_categorize:
        if col in clean_df.columns:
            clean_df[col] = pd.Categorical(clean_df[col])

    keep_cols = ['value'] + [c for c in factors_to_categorize if c in clean_df.columns]
    clean_df = clean_df[keep_cols].dropna().reset_index(drop=True)

    # Apply the Smithson-Verkuilen transform if requested
    if family == 'beta_smithson':
        n = len(clean_df)
        y = clean_df['value'].values
        # Standard S&V correction: shrink slightly toward 0.5
        clean_df['value'] = (y * (n - 1) + 0.5) / n
    elif family == 'ordbeta':
        # Ordered beta handles 0/1 natively — no transformation needed.
        # Just clip slightly out-of-range values that may arise from rounding.
        clean_df['value'] = clean_df['value'].clip(lower=0.0, upper=1.0)
    else:
        raise ValueError(f"Unknown family: {family}")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(clean_df)
        ro.globalenv[f'beta_df_{model_idx}'] = r_df

    # Drop unused factor levels in R
    ro.r(f'''
        beta_df_{model_idx} <- droplevels(beta_df_{model_idx})
    ''')

    full_formula = f"{fixed_formula} + {random_formula}"
    model_name_r = f'model_beta_{model_idx}'

    if family == 'ordbeta':
        family_call = 'ordbeta(link = "logit")'
    else:
        family_call = 'beta_family(link = "logit")'

    ro.r(f'''
        library(glmmTMB)
        {model_name_r} <- glmmTMB({full_formula},
                                  data = beta_df_{model_idx},
                                  family = {family_call})
        cat("\\n=== Beta Model {model_idx}: {full_formula} ===\\n")
        print(summary({model_name_r}))
    ''')

    # Check convergence
    converged_msg = ro.r(f'''
        if (is.null({model_name_r}$fit$convergence)) {{
            "unknown"
        }} else if ({model_name_r}$fit$convergence == 0) {{
            "ok"
        }} else {{
            paste("convergence code", {model_name_r}$fit$convergence)
        }}
    ''')[0]

    if converged_msg != 'ok':
        _warn_fallback(
            f"Beta model {model_idx} convergence status: {converged_msg}"
        )

    # Check for singular fit (random effect variance collapsed to ~0)
    try:
        re_var = float(ro.r(f'''
            .vc <- VarCorr({model_name_r})$cond
            if (length(.vc) > 0) {{
                min(sapply(.vc, function(m) attr(m, "stddev")^2))
            }} else {{
                NA
            }}
        ''')[0])
        if re_var < 1e-8:
            _warn_fallback(
                f"Beta model {model_idx} has SINGULAR FIT: random-effect "
                f"variance is {re_var:.2e} (effectively zero). The "
                f"(1|session) intercept is not identifiable from the data — "
                f"either too few observations per session, or the response "
                f"is too consistent within sessions. Consider refitting "
                f"without the random intercept and reporting that choice."
            )
    except Exception:
        pass

    return model_name_r


# =============================================================================
# OMNIBUS TESTS (UNIFIED FOR R MODELS)
# =============================================================================

def omnibus_tests_r(model_name_r, model_type='lmm'):
    """
    Run Type-III omnibus tests on an R-fitted model.
      - LMM: F-tests with Kenward-Roger denominator df via lmerTest::anova
      - Beta: Wald chi-square tests via car::Anova(type=3)
    """
    if model_type == 'lmm':
        try:
            ro.r(f'''
                library(lmerTest)
                .anova_tbl <- as.data.frame(
                    anova({model_name_r}, type = 3, ddf = "Kenward-Roger")
                )
                .anova_tbl$term <- rownames(.anova_tbl)
                .anova_tbl <- .anova_tbl[, c("term",
                    setdiff(names(.anova_tbl), "term"))]
            ''')
            with localconverter(ro.default_converter + pandas2ri.converter):
                df = ro.conversion.rpy2py(ro.r('.anova_tbl'))
            return df.reset_index(drop=True)
        except Exception as e:
            _warn_fallback(
                f"Kenward-Roger F-tests failed for {model_name_r} ({e}). "
                f"Falling back to Satterthwaite df."
            )
            try:
                ro.r(f'''
                    .anova_tbl <- as.data.frame(
                        anova({model_name_r}, type = 3, ddf = "Satterthwaite")
                    )
                    .anova_tbl$term <- rownames(.anova_tbl)
                    .anova_tbl <- .anova_tbl[, c("term",
                        setdiff(names(.anova_tbl), "term"))]
                ''')
                with localconverter(ro.default_converter + pandas2ri.converter):
                    df = ro.conversion.rpy2py(ro.r('.anova_tbl'))
                return df.reset_index(drop=True)
            except Exception as e2:
                _warn_fallback(
                    f"Satterthwaite df also failed for {model_name_r} ({e2}). "
                    f"Returning empty omnibus table."
                )
                return pd.DataFrame()

    elif model_type == 'beta':
        try:
            ro.r(f'''
                library(car)
                .anova_tbl <- as.data.frame(Anova({model_name_r}, type = 3))
                .anova_tbl$term <- rownames(.anova_tbl)
                .anova_tbl <- .anova_tbl[, c("term",
                    setdiff(names(.anova_tbl), "term"))]
            ''')
            with localconverter(ro.default_converter + pandas2ri.converter):
                df = ro.conversion.rpy2py(ro.r('.anova_tbl'))
            return df.reset_index(drop=True)
        except Exception as e:
            _warn_fallback(
                f"car::Anova failed for {model_name_r} ({e}). "
                f"Trying emmeans::joint_tests as fallback."
            )
            return pd.DataFrame()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =============================================================================
# MARGINAL MEANS AND PAIRWISE COMPARISONS (UNIFIED)
# =============================================================================

def get_marginal_means_r(model_name_r, factors, model_type='lmm',
                         collapse_over=None):
    """
    Compute estimated marginal means via emmeans for any R model.

    Parameters
    ----------
    collapse_over : list of str, optional
        Factors to average over (equal weights) before reporting means.
        The model is NOT refit; only the emmeans grid changes.

    Returns columns standardised across model types:
        condition, marginal_mean, se, df, ci_lower, ci_upper, plus the
        original factor columns (excluding any collapsed factors).
    """
    if collapse_over is None:
        collapse_over = []

    keep_factors = [f for f in factors if f not in collapse_over]
    if not keep_factors:
        _warn_fallback(
            f"get_marginal_means_r: nothing left to estimate after "
            f"collapsing over {collapse_over}. Returning empty DataFrame."
        )
        return pd.DataFrame()

    emmeans_spec = ' + '.join(keep_factors)

    type_arg = '"response"' if model_type == 'beta' else '"link"'
    mode_arg = '"asymptotic"' if model_type == 'beta' else '"kenward-roger"'

    # weights = "equal" is the default but we set it explicitly so the
    # collapsing behaviour is unambiguous in the R log.
    try:
        ro.r(f'''
            library(emmeans)
            .emm <- emmeans({model_name_r}, ~ {emmeans_spec},
                            type = {type_arg}, mode = {mode_arg},
                            weights = "equal")
            .emm_summary <- as.data.frame(summary(.emm))
        ''')
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.r('.emm_summary'))

        rename_map = {
            'emmean': 'marginal_mean',
            'response': 'marginal_mean',
            'prob': 'marginal_mean',
            'SE': 'se',
            'lower.CL': 'ci_lower',
            'upper.CL': 'ci_upper',
            'asymp.LCL': 'ci_lower',
            'asymp.UCL': 'ci_upper',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items()
                                 if k in df.columns})

        if all(f in df.columns for f in keep_factors):
            condition_series = df[keep_factors[0]].astype(str)
            for f in keep_factors[1:]:
                condition_series = condition_series + '_' + df[f].astype(str)
            df['condition'] = condition_series

        return df.reset_index(drop=True)

    except Exception as e:
        _warn_fallback(
            f"emmeans failed for {model_name_r} ({e}). "
            f"Marginal means not computed — returning empty DataFrame."
        )
        return pd.DataFrame()

def pairwise_comparisons_r(model_name_r, factors, model_type='lmm',
                            adjustment='holm', alpha=0.05,
                            collapse_over=None, by_factors=None):
    """
    Pairwise contrasts via emmeans for any R model.

    Parameters
    ----------
    factors : list of str
        Factors in the emmeans grid (after collapsing).
    collapse_over : list of str, optional
        Factors to average over before computing contrasts.
    by_factors : list of str, optional
        Factors to condition on (i.e., compute pairwise within each level).
        If None and 'opto' is in factors, will default to all non-opto factors.
    adjustment : {'holm', 'bonferroni', 'fdr', 'tukey', 'none'}
    """
    if collapse_over is None:
        collapse_over = []

    keep_factors = [f for f in factors if f not in collapse_over]
    emmeans_spec = ' + '.join(keep_factors)

    # Default 'by' behaviour: if opto is present, condition on the other factors
    if by_factors is None and 'opto' in keep_factors:
        by_factors = [f for f in keep_factors if f != 'opto']

    type_arg = '"response"' if model_type == 'beta' else '"link"'
    mode_arg = '"asymptotic"' if model_type == 'beta' else '"kenward-roger"'

    # Build the contrast call
    if by_factors:
        by_spec = 'c(' + ', '.join(f'"{f}"' for f in by_factors) + ')'
        # 'simple' makes emmeans pairwise within remaining factor (opto here)
        contrast_call = (
            f'contrast(.emm, method = "pairwise", '
            f'simple = "opto", combine = TRUE, '
            f'adjust = "{adjustment}")'
        )
    else:
        contrast_call = (
            f'pairs(.emm, adjust = "{adjustment}")'
        )

    try:
        ro.r(f'''
            library(emmeans)
            .emm <- emmeans({model_name_r}, ~ {emmeans_spec},
                            type = {type_arg}, mode = {mode_arg})
            .pairs <- {contrast_call}
            .pairs_summary <- as.data.frame(summary(.pairs))
        ''')
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.r('.pairs_summary'))

        # Parse contrast string into condition_1 / condition_2
        if 'contrast' in df.columns:
            sep = ' / ' if df['contrast'].astype(str).str.contains(' / ').any() else ' - '
            split_cols = df['contrast'].astype(str).str.split(sep, expand=True)
            df['condition_1'] = split_cols[0].str.strip()
            if split_cols.shape[1] > 1:
                df['condition_2'] = split_cols[1].str.strip()

        # Standardise column names
        rename_map = {
            'odds.ratio': 'ratio',
            'estimate': 'diff',
            'p.value': 'p_value',
            't.ratio': 't_stat',
            'z.ratio': 'z_stat',
            'SE': 'se',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items()
                                 if k in df.columns})

        # Apply sign convention: higher opto - lower opto
        opto_order = ['none', 'unilateral', 'bilateral']
        if 'opto' in factors and 'condition_1' in df.columns and 'condition_2' in df.columns:
            df = _enforce_opto_sign_convention(df, opto_order, by_factors)

        if 'p_value' in df.columns:
            df['p_adjusted'] = df['p_value']  # emmeans already adjusted
            df['significant'] = df['p_adjusted'] < alpha

        return df.reset_index(drop=True)

    except Exception as e:
        _warn_fallback(
            f"emmeans pairwise contrasts failed for {model_name_r} ({e}). "
            f"No pairwise comparisons available — returning empty DataFrame. "
            f"DO NOT report p-values from this model without re-fitting."
        )
        return pd.DataFrame()


def _enforce_opto_sign_convention(df, opto_order, by_factors):
    """
    Ensure all opto contrasts are reported as higher_opto - lower_opto.
    Uses dedicated factor columns when available (preferred), otherwise
    parses condition strings (with a warning).
    """
    # Preferred path: emmeans 'by' contrasts include explicit factor columns
    # when combine=TRUE. Look for an 'opto' column in the df.
    if 'opto' in df.columns:
        # Already has explicit opto labels — but for pairwise contrasts
        # using simple="opto", emmeans collapses opto into the contrast
        # string. Try parsing it from condition_1 / condition_2 directly.
        pass

    # Parse opto level from condition strings.
    # We rely on the contrast being purely a difference in opto when
    # by_factors are used, so condition_1 / condition_2 should be opto levels.
    parsed_warning_emitted = [False]

    def parse_opto(label):
        s = str(label).strip()
        # Try direct match
        if s in opto_order:
            return s
        # Try splitting on common separators
        for token in re.split(r'[,\s_]+', s):
            token = token.strip()
            if token in opto_order:
                return token
        if not parsed_warning_emitted[0]:
            _warn_fallback(
                f"Could not unambiguously parse opto level from contrast "
                f"label '{s}'. Sign convention may not be applied "
                f"consistently. Verify manually."
            )
            parsed_warning_emitted[0] = True
        return None

    for idx_row in df.index:
        c1 = df.loc[idx_row, 'condition_1']
        c2 = df.loc[idx_row, 'condition_2']
        opto_1 = parse_opto(c1)
        opto_2 = parse_opto(c2)

        if opto_1 in opto_order and opto_2 in opto_order:
            if opto_order.index(opto_1) < opto_order.index(opto_2):
                df.loc[idx_row, 'condition_1'] = c2
                df.loc[idx_row, 'condition_2'] = c1
                for col, sign in [('diff', -1), ('t_stat', -1), ('z_stat', -1)]:
                    if col in df.columns:
                        df.loc[idx_row, col] = sign * df.loc[idx_row, col]
                if 'ratio' in df.columns and df.loc[idx_row, 'ratio'] != 0:
                    df.loc[idx_row, 'ratio'] = 1.0 / df.loc[idx_row, 'ratio']

    return df


# =============================================================================
# MODEL FIT STATISTICS
# =============================================================================

def get_model_fit_stats(model_name_r, model_type='lmm'):
    """Return AIC, BIC, logLik, and (for LMM) marginal/conditional R²."""
    stats_dict = {}

    try:
        stats_dict['AIC'] = float(ro.r(f'AIC({model_name_r})')[0])
        stats_dict['BIC'] = float(ro.r(f'BIC({model_name_r})')[0])
        stats_dict['logLik'] = float(ro.r(f'as.numeric(logLik({model_name_r}))')[0])
    except Exception as e:
        _warn_fallback(f"Could not extract AIC/BIC for {model_name_r}: {e}")

    # R² via performance package (works for both lmer and glmmTMB)
    try:
        ro.r(f'''
            suppressWarnings({{
                .r2 <- performance::r2({model_name_r})
            }})
        ''')
        try:
            stats_dict['R2_marginal'] = float(ro.r('.r2$R2_marginal')[0])
            stats_dict['R2_conditional'] = float(ro.r('.r2$R2_conditional')[0])
        except Exception:
            pass
    except Exception as e:
        _warn_fallback(
            f"performance::r2 failed for {model_name_r} ({e}). "
            f"R² not reported."
        )

    # Random-effects variance components
    try:
        if model_type == 'lmm':
            ro.r(f'''
                .vc <- as.data.frame(VarCorr({model_name_r}))
            ''')
        else:
            # glmmTMB: VarCorr(model)$cond is a list of matrices, one per
            # grouping factor. Use as.data.frame on the whole VarCorr object
            # via the broom.mixed-style approach.
            ro.r(f'''
                .vc_obj <- VarCorr({model_name_r})$cond
                .vc <- do.call(rbind, lapply(names(.vc_obj), function(g) {{
                    mat <- .vc_obj[[g]]
                    data.frame(grp = g,
                               var1 = "(Intercept)",
                               vcov = as.numeric(attr(mat, "stddev"))^2,
                               sdcor = as.numeric(attr(mat, "stddev")))
                }}))
                # Add residual variance from sigma^2 if present
                .resid_var <- tryCatch(sigma({model_name_r})^2, error = function(e) NA)
            ''')

        with localconverter(ro.default_converter + pandas2ri.converter):
            vc = ro.conversion.rpy2py(ro.r('.vc'))

        # Extract variance values regardless of column-name flavour
        if 'vcov' in vc.columns and 'grp' in vc.columns:
            for _, row in vc.iterrows():
                key = f"var_{row['grp']}"
                # For LMM, residual appears as grp='Residual' which is fine
                stats_dict[key] = float(row['vcov'])

        # Warn on singular fits (variance ≈ 0)
        for k, v in stats_dict.items():
            if k.startswith('var_') and k != 'var_Residual':
                if v is not None and v < 1e-8:
                    _warn_fallback(
                        f"{model_name_r}: random effect '{k}' has near-zero "
                        f"variance ({v:.2e}) — singular fit. The random "
                        f"intercept is not identifiable; consider dropping it "
                        f"or fitting a fixed-effects-only model."
                    )
    except Exception as e:
        _warn_fallback(
            f"Variance components extraction failed for {model_name_r}: {e}"
        )

    # Beta dispersion phi
    if model_type == 'beta':
        try:
            phi = float(ro.r(f'sigma({model_name_r})')[0])
            stats_dict['phi_dispersion'] = phi
        except Exception:
            pass

    return stats_dict


def get_fixed_effects_r(model_name_r, model_type='lmm'):
    """Tidy fixed-effects table for LMM or beta model."""
    try:
        if model_type == 'lmm':
            ro.r(f'''
                .coef_df <- as.data.frame(summary({model_name_r})$coefficients)
                .coef_df$term <- rownames(.coef_df)
                .coef_df <- .coef_df[, c("term",
                    setdiff(names(.coef_df), "term"))]
            ''')
        else:
            ro.r(f'''
                .coef_df <- as.data.frame(coef(summary({model_name_r}))$cond)
                .coef_df$term <- rownames(.coef_df)
                .coef_df <- .coef_df[, c("term",
                    setdiff(names(.coef_df), "term"))]
            ''')
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.r('.coef_df'))
        return df.reset_index(drop=True)
    except Exception as e:
        _warn_fallback(f"Fixed-effects extraction failed for {model_name_r}: {e}")
        return pd.DataFrame()


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def diagnostics_r(model_name_r, model_idx, save_dir_r, model_type='lmm', n_sim=1000):
    """Save DHARMa diagnostic plots for any R model."""
    print(f"Saving diagnostics for {model_name_r}...")
    try:
        ro.r(f'''
            library(DHARMa)
            set.seed(42)
            tryCatch({{
                sim_res <- simulateResiduals(fittedModel = {model_name_r},
                                              plot = FALSE, n = {n_sim})
                png("{save_dir_r}/{model_name_r}_diagnostics.png",
                    width = 1200, height = 600)
                plot(sim_res)
                dev.off()

                png("{save_dir_r}/{model_name_r}_qq.png",
                    width = 800, height = 600)
                plotQQunif(sim_res)
                dev.off()

                png("{save_dir_r}/{model_name_r}_resid_vs_pred.png",
                    width = 800, height = 600)
                plotResiduals(sim_res)
                dev.off()
            }}, error = function(e) {{
                if (dev.cur() > 1) dev.off()
                print(paste("DHARMa error:", e$message))
            }})
        ''')
        ro.r('while (dev.cur() > 1) dev.off()')
        print(f"  {model_name_r} diagnostics saved")
    except Exception as e:
        _warn_fallback(
            f"DHARMa diagnostics failed for {model_name_r}: {e}. "
            f"No residual plots available — model assumptions UNCHECKED."
        )


# =============================================================================
# UTILITY FUNCTIONS (UNCHANGED)
# =============================================================================

def count_differences(cond1, cond2):
    """Count how many factors differ between two conditions."""
    factors1 = str(cond1).split('_')
    factors2 = str(cond2).split('_')
    if len(factors1) != len(factors2):
        return float('inf')
    return sum(f1 != f2 for f1, f2 in zip(factors1, factors2))


def get_significance_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'


def _write_section(ws, start_row, title, df, header_fill, section_fill):
    """Write a titled section to a worksheet starting at start_row. Returns next free row."""
    cell = ws.cell(row=start_row, column=1, value=title)
    cell.font = Font(bold=True, size=12, color='FFFFFF')
    cell.fill = section_fill
    cell.alignment = Alignment(horizontal='left')
    ncols = max(1, df.shape[1] if df is not None and not df.empty else 1)
    for c in range(2, ncols + 1):
        ws.cell(row=start_row, column=c).fill = section_fill

    if df is None or df.empty:
        ws.cell(row=start_row + 1, column=1, value='(no results)').font = Font(italic=True, color='888888')
        return start_row + 3

    header_row = start_row + 1
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=header_row, column=j, value=str(col))
        c.font = Font(bold=True)
        c.fill = header_fill

    for i, (_, row) in enumerate(df.iterrows(), start=header_row + 1):
        for j, col in enumerate(df.columns, start=1):
            val = row[col]
            if isinstance(val, (np.floating, np.integer)):
                val = val.item()
            elif isinstance(val, (list, tuple, np.ndarray)):
                val = str(val)
            elif np.isscalar(val) and pd.isna(val):
                val = None
            ws.cell(row=i, column=j, value=val)

    return header_row + 1 + len(df) + 2  # blank row after section


def _safe_sheet_name(name, used):
    bad = '[]:*?/\\'
    cleaned = ''.join('_' if ch in bad else ch for ch in name)[:31]
    base = cleaned
    i = 2
    while cleaned in used:
        suffix = f"_{i}"
        cleaned = (base[:31 - len(suffix)]) + suffix
        i += 1
    used.add(cleaned)
    return cleaned


# =============================================================================
# CONFIGURATION
# =============================================================================

PAIRWISE_ADJUSTMENT = 'holm'

path_root = config.DATA_DIR_EXP
experiment = ["ALM_2", "ALM_3"]
save_dir = config.FIGURE_DIR / "behavior/rt/stats"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
save_dir_r = save_dir.as_posix()

os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

trialData_full, sessionData_full, _ = utils.load_data(
    path_root, experiment,
    data_types=['trialData', 'sessionData'],
    exclude_grace_violations=True,
)

# Only opto sessions
trialData_full = trialData_full[trialData_full['optoLat'].isin(['unilat', 'bilat'])]

# Same hit-rate filter as behavioural pipeline
(trialData_filtered,
 sessionData_filtered,
 _,
 sessions_exclude) = utils.filter_sessions_hitrate(
    trialData_full,
    sessionData=sessionData_full,
    spikeData=None,
    min_hitrate=0.3,
)

# Only correct visual / auditory hits
trialData_filtered = trialData_filtered[trialData_filtered['correctResponse'] == 1]
trialData_filtered = trialData_filtered[trialData_filtered['trialType'].isin(['X', 'Y'])]
trialData_filtered = trialData_filtered.dropna(subset=['reactionTime'])

# Convert microseconds (or whatever) to seconds, then to milliseconds.
# Your original code divided by 1000 — keep that convention.
trialData_filtered = trialData_filtered.copy()
trialData_filtered['rt_ms'] = trialData_filtered['reactionTime'] / 1000

# RTs should be positive and within the response window. Drop any that
# slipped through preprocessing.
n_before = len(trialData_filtered)
trialData_filtered = trialData_filtered[
    (trialData_filtered['rt_ms'] > 0)
    & (trialData_filtered['rt_ms'] <= 1000)
].copy()
n_dropped = n_before - len(trialData_filtered)
if n_dropped > 0:
    print(f"Dropped {n_dropped} trials with RT outside (0, 1000] ms.")

# Log-transform for modelling. We fit lmm on log(rt_ms); emmeans will
# back-transform to ms when type = "response".
trialData_filtered['log_rt'] = np.log(trialData_filtered['rt_ms'])

# Recode factors for readability and match the levels used in your
# behavioural pipeline ('none', 'unilateral', 'bilateral').
trialData_filtered['opto'] = trialData_filtered['optoNorm'].map({
    0: 'none', 1: 'unilateral', 2: 'bilateral'
})
trialData_filtered['direction'] = trialData_filtered['visualChangeSide'].map({
    1: 'left', 2: 'right'
})
trialData_filtered['visual_saliency'] = trialData_filtered['visualOriChange_norm'].map({
    0: 'none', 1: 'thr', 2: 'max'
})
trialData_filtered['auditory_saliency'] = trialData_filtered['audioOctChange_norm'].map({
    0: 'none', 1: 'thr', 2: 'max'
})

# Rename animal/session columns to match the behavioural pipeline
trialData_filtered['session'] = trialData_filtered['session_ID']
trialData_filtered['animal'] = trialData_filtered['mousename']

# Set ordered categories so emmeans reports a sensible reference
opto_levels = ['none', 'unilateral', 'bilateral']
trialData_filtered['opto'] = pd.Categorical(
    trialData_filtered['opto'], categories=opto_levels, ordered=False
)


# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================
#
# Note the model_type field — we use 'lmm' so the existing pipeline gives
# us Kenward-Roger F-tests for the omnibus and KR-adjusted t-tests for
# pairwise contrasts, just like the d-prime models. The response is
# log_rt, so emmeans contrasts are differences on the log scale; with
# type = "response" they're back-transformed to RATIOS of geometric
# means on the millisecond scale. We patch this back-transformation in
# below via a small wrapper.

model_specs = [
    {
        'name': "Visual RT opto effect left vs right",
        'family': 'lmm',
        'fixed': "value ~ opto * direction * visual_saliency",
        # Trial-level data => session nested in animal.
        # If this gives singular fits, fall back to "(1|session)".
        'random': "(1|session)",
        'filters': {
            'trialType': 'X',
            'direction': ['left', 'right'],
            'visual_saliency': ['thr', 'max'],
        },
        'collapse_over': [],
        'collapse_over_extra': ['direction'],
    },
    {
        'name': "Auditory RT",
        'family': 'lmm',
        'fixed': "value ~ opto * auditory_saliency",
        'random': "(1|session)",
        'filters': {
            'trialType': 'Y',
            'auditory_saliency': ['thr', 'max'],
        },
        'collapse_over': [],
    },
]


# =============================================================================
# DATASETS PER MODEL
# =============================================================================

row_data_dict = {}
for i, spec in enumerate(model_specs):
    rd = trialData_filtered.copy()
    for col, val in spec['filters'].items():
        if col not in rd.columns:
            continue
        if isinstance(val, list):
            rd = rd[rd[col].isin(val)]
        else:
            rd = rd[rd[col] == val]
    rd['value'] = rd['log_rt']  # the helpers expect a 'value' column
    rd = rd.dropna(subset=['value']).reset_index(drop=True)
    row_data_dict[i] = rd


# =============================================================================
# INITIALISE R
# =============================================================================

importr('base')
importr('lmerTest')
importr('emmeans')
importr('car')
try:
    importr('performance')
except Exception:
    _warn_fallback(
        "R package 'performance' not installed. R² values will be missing."
    )
importr('DHARMa')

# Raise emmeans' n-cap so KR df is computed for trial-level RT models
ro.r('emm_options(pbkrtest.limit = 1e5, lmerTest.limit = 1e5)')

# =============================================================================
# FIT MODELS
# =============================================================================

fitted_models = {}
for i, spec in enumerate(model_specs):
    factors_in_model = re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
    factors_in_model = [f for f in factors_in_model
                        if f != 'log_rt' and f in row_data_dict[i].columns]
    factors_in_model = list(dict.fromkeys(factors_in_model))
    random_cols = ['session', 'animal']
    all_cols = list(set(factors_in_model + random_cols))

    data = row_data_dict[i][['value'] + all_cols].copy()
    fitted_models[i] = fit_lmm_r(i, spec['fixed'], spec['random'],
                                  data, all_cols)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

print("\n" + "=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
for i, spec in enumerate(model_specs):
    diagnostics_r(fitted_models[i], i, save_dir_r, model_type='lmm')


# =============================================================================
# OMNIBUS TESTS
# =============================================================================

print("\n" + "=" * 60)
print("OMNIBUS TESTS (Kenward-Roger F)")
print("=" * 60)
omnibus_results = {}
for i, spec in enumerate(model_specs):
    print(f"\n--- Model {i}: {spec['name']} ---")
    omni = omnibus_tests_r(fitted_models[i], model_type='lmm')
    omnibus_results[i] = omni
    if not omni.empty:
        print(omni.to_string(index=False))


# =============================================================================
# MARGINAL MEANS AND PAIRWISE COMPARISONS
# =============================================================================
#
# We override the default 'type = "link"' for LMMs to 'type = "response"'
# because the response variable is log(RT). emmeans then exponentiates
# the marginal means back to milliseconds (geometric means), and reports
# pairwise contrasts as RATIOS of geometric means.

def get_marginal_means_rt(model_name_r, factors, collapse_over=None):
    """RT-specific wrapper: emmeans with type='response' for log link."""
    if collapse_over is None:
        collapse_over = []
    keep_factors = [f for f in factors if f not in collapse_over]
    if not keep_factors:
        return pd.DataFrame()
    emmeans_spec = ' + '.join(keep_factors)

    try:
        ro.r(f'''
            library(emmeans)
            .emm <- emmeans({model_name_r}, ~ {emmeans_spec},
                            mode = "kenward-roger",
                            tran = "log",
                            type = "response",
                            weights = "equal")
            .emm_summary <- as.data.frame(summary(.emm))
        ''')
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.r('.emm_summary'))

        # When type='response' on a log link, emmeans returns 'response'
        # rather than 'emmean'.
        rename_map = {
            'response': 'marginal_mean_ms',
            'emmean': 'marginal_mean_ms',  # safety
            'SE': 'se_ms',
            'lower.CL': 'ci_lower_ms',
            'upper.CL': 'ci_upper_ms',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items()
                                 if k in df.columns})

        if all(f in df.columns for f in keep_factors):
            condition_series = df[keep_factors[0]].astype(str)
            for f in keep_factors[1:]:
                condition_series = condition_series + '_' + df[f].astype(str)
            df['condition'] = condition_series

        return df.reset_index(drop=True)
    except Exception as e:
        _warn_fallback(f"emmeans (RT) failed for {model_name_r}: {e}")
        return pd.DataFrame()


def pairwise_contrasts_rt(model_name_r, factors, adjustment='holm',
                           alpha=0.05, collapse_over=None, by_factors=None):
    """RT-specific pairwise: contrasts are RATIOS of geometric means.

    On the log scale, a contrast 'A - B' exponentiates to A/B, which
    emmeans reports automatically when type = "response". The column is
    'ratio', not 'estimate'. t-tests on the log scale are still valid
    inference (Wald t with KR df).
    """
    if collapse_over is None:
        collapse_over = []
    keep_factors = [f for f in factors if f not in collapse_over]
    emmeans_spec = ' + '.join(keep_factors)

    if by_factors is None and 'opto' in keep_factors:
        by_factors = [f for f in keep_factors if f != 'opto']

    if by_factors:
        contrast_call = (
            f'contrast(.emm, method = "pairwise", '
            f'simple = "opto", combine = TRUE, '
            f'adjust = "{adjustment}")'
        )
    else:
        contrast_call = f'pairs(.emm, adjust = "{adjustment}")'

    try:
        ro.r(f'''
            library(emmeans)
            .emm <- emmeans({model_name_r}, ~ {emmeans_spec},
                            mode = "kenward-roger",
                            tran = "log",
                            type = "response",
                            weights = "equal")
            .pairs <- {contrast_call}
            .pairs_summary <- as.data.frame(summary(.pairs))
        ''')
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.r('.pairs_summary'))

        # Parse contrast strings
        if 'contrast' in df.columns:
            sep = ' / ' if df['contrast'].astype(str).str.contains(' / ').any() else ' - '
            split_cols = df['contrast'].astype(str).str.split(sep, expand=True)
            df['condition_1'] = split_cols[0].str.strip()
            if split_cols.shape[1] > 1:
                df['condition_2'] = split_cols[1].str.strip()

        rename_map = {
            'ratio': 'ratio',
            'estimate': 'ratio',  # safety
            'p.value': 'p_value',
            't.ratio': 't_stat',
            'SE': 'se_ratio',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items()
                                 if k in df.columns})

        # Sign convention: higher opto / lower opto. Because contrasts
        # are ratios, "higher / lower" reads as "how much slower is the
        # higher-opto condition than the lower". A ratio > 1 means
        # slower; < 1 means faster.
        opto_order = ['none', 'unilateral', 'bilateral']
        if ('opto' in factors
                and 'condition_1' in df.columns
                and 'condition_2' in df.columns):
            df = _enforce_opto_sign_convention(df, opto_order, by_factors)

        if 'p_value' in df.columns:
            df['p_adjusted'] = df['p_value']
            df['significant'] = df['p_adjusted'] < alpha

        return df.reset_index(drop=True)
    except Exception as e:
        _warn_fallback(f"emmeans pairwise (RT) failed for {model_name_r}: {e}")
        return pd.DataFrame()


print("\n" + "=" * 60)
print("MARGINAL MEANS AND PAIRWISE COMPARISONS")
print("=" * 60)

marginal_means = {}
posthoc_results = {}
for i, spec in enumerate(model_specs):
    print(f"\n--- Model {i}: {spec['name']} ---")
    factors = re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
    factors = [f for f in factors if f != 'log_rt']
    factors = list(dict.fromkeys(factors))

    mm = get_marginal_means_rt(fitted_models[i], factors)
    marginal_means[i] = mm
    if not mm.empty:
        print("\nMarginal Means (ms, geometric):")
        cols = ['condition', 'marginal_mean_ms', 'se_ms',
                'ci_lower_ms', 'ci_upper_ms']
        cols = [c for c in cols if c in mm.columns]
        print(mm[cols].to_string(index=False))

    ph = pairwise_contrasts_rt(
        fitted_models[i], factors,
        adjustment=PAIRWISE_ADJUSTMENT,
        collapse_over=spec['collapse_over'],
    )
    posthoc_results[i] = ph
    if not ph.empty:
        print("\nPairwise Comparisons (ratios of geometric means):")
        cols = ['condition_1', 'condition_2', 'ratio', 'se_ratio',
                't_stat', 'p_adjusted', 'significant']
        cols = [c for c in cols if c in ph.columns]
        print(ph[cols].to_string(index=False))


# =============================================================================
# COLLAPSED MARGINAL MEANS AND PAIRWISE (over collapse_over_extra)
# =============================================================================

marginal_means_collapsed = {}
posthoc_results_collapsed = {}
for i, spec in enumerate(model_specs):
    collapse_extra = spec.get('collapse_over_extra', [])
    if not collapse_extra:
        marginal_means_collapsed[i] = pd.DataFrame()
        posthoc_results_collapsed[i] = pd.DataFrame()
        continue

    print(f"\n--- Model {i} collapsed over {collapse_extra} ---")
    factors = re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
    factors = [f for f in factors if f != 'log_rt']
    factors = list(dict.fromkeys(factors))

    mm_c = get_marginal_means_rt(fitted_models[i], factors,
                                  collapse_over=collapse_extra)
    marginal_means_collapsed[i] = mm_c
    if not mm_c.empty:
        print(mm_c.to_string(index=False))

    ph_c = pairwise_contrasts_rt(
        fitted_models[i], factors,
        adjustment=PAIRWISE_ADJUSTMENT,
        collapse_over=collapse_extra,
    )
    posthoc_results_collapsed[i] = ph_c
    if not ph_c.empty:
        print(ph_c.to_string(index=False))


# =============================================================================
# UNIFIED EXCEL WORKBOOK
# =============================================================================

SECTION_FILL = PatternFill(start_color='305496', end_color='305496', fill_type='solid')
HEADER_FILL = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')
DIFF1_FILL = PatternFill(start_color='FFF2CC', end_color='FFF2CC', fill_type='solid')

results_workbook_path = os.path.join(save_dir, 'rt_results.xlsx')

overview_rows = []
for i, spec in enumerate(model_specs):
    rd = row_data_dict[i]
    overview_rows.append({
        'model_idx': i,
        'model_name': spec['name'],
        'family': 'LMM on log(RT) (Gaussian, lmerTest)',
        'fixed_formula': spec['fixed'],
        'random_formula': spec['random'],
        'n_obs': len(rd),
        'n_sessions': rd['session'].nunique(),
        'n_animals': rd['animal'].nunique(),
        'pairwise_adjustment': PAIRWISE_ADJUSTMENT,
    })
overview_df = pd.DataFrame(overview_rows)

readme_lines = [
    "Reaction time statistics workbook",
    "",
    "Model: Gaussian LMM on log(reaction time in ms).",
    "  - Fit with lmerTest::lmer, REML, bobyqa optimiser.",
    "  - Random effects: sessions nested in animals.",
    "",
    "Omnibus tests:",
    "  - Type-III F-tests with Kenward-Roger denominator df.",
    "",
    "Marginal means:",
    "  - Computed via emmeans with type='response' on the log link,",
    "    so means are GEOMETRIC means on the millisecond scale.",
    "  - SE and CI shown in ms but computed via the delta method from",
    "    the log-scale SE.",
    "",
    "Pairwise comparisons:",
    "  - Reported as RATIOS of geometric means (higher_opto / lower_opto).",
    "    A ratio of 1.10 means the higher-opto condition is 10% slower.",
    "  - t-statistic and df are from the log-scale Wald test with KR df.",
    f"  - Multiple-comparison adjustment: {PAIRWISE_ADJUSTMENT}.",
    "  - Yellow rows: contrasts where conditions differ in only one factor.",
    "",
    "Exclusion criteria:",
    "  - Grace-period violators excluded at load time.",
    "  - Only opto sessions (optoLat in {'unilat', 'bilat'}).",
    "  - Sessions with hit-rate < 0.3 in either modality excluded.",
    "  - Only correct hits on visual (X) or auditory (Y) trials.",
    "  - RT outside (0, 1000] ms excluded.",
]
readme_df = pd.DataFrame({'README': readme_lines})


with pd.ExcelWriter(results_workbook_path, engine='openpyxl', mode='w') as writer:
    readme_df.to_excel(writer, sheet_name='README', index=False)
    ws_rm = writer.sheets['README']
    ws_rm.column_dimensions['A'].width = 90

    overview_df.to_excel(writer, sheet_name='Overview', index=False)
    ws_ov = writer.sheets['Overview']
    for j in range(1, overview_df.shape[1] + 1):
        c = ws_ov.cell(row=1, column=j)
        c.font = Font(bold=True)
        c.fill = HEADER_FILL
    widths = {'model_idx': 10, 'model_name': 36, 'family': 36,
              'fixed_formula': 42, 'random_formula': 22,
              'n_obs': 10, 'n_sessions': 12, 'n_animals': 12,
              'pairwise_adjustment': 18}
    for j, col in enumerate(overview_df.columns, start=1):
        ws_ov.column_dimensions[chr(64 + j)].width = widths.get(col, 18)

    used_names = {'README', 'Overview'}

    for i, spec in enumerate(model_specs):
        rd = row_data_dict[i]
        sheet = _safe_sheet_name(f"M{i}_{spec['name']}", used_names)
        pd.DataFrame().to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.sheets[sheet]

        # MODEL INFO
        info_df = pd.DataFrame({
            'field': [
                'Model index', 'Model name', 'Family', 'Response',
                'Fixed formula', 'Random formula',
                'N observations (trials)', 'N sessions', 'N animals',
                'Pairwise adjustment', 'df method',
            ],
            'value': [
                i, spec['name'], 'LMM (Gaussian, lmerTest)',
                'log(RT in ms)', spec['fixed'], spec['random'],
                len(rd), rd['session'].nunique(), rd['animal'].nunique(),
                PAIRWISE_ADJUSTMENT, 'Kenward-Roger',
            ]
        })
        next_row = _write_section(ws, 1, 'MODEL INFO', info_df,
                                  HEADER_FILL, SECTION_FILL)

        # FIT STATISTICS
        fit_stats = get_model_fit_stats(fitted_models[i], model_type='lmm')
        fit_df = pd.DataFrame({'statistic': list(fit_stats.keys()),
                               'value': list(fit_stats.values())})
        next_row = _write_section(ws, next_row, 'FIT STATISTICS',
                                  fit_df, HEADER_FILL, SECTION_FILL)

        # FIXED EFFECTS (log scale)
        fe_df = get_fixed_effects_r(fitted_models[i], model_type='lmm')
        next_row = _write_section(
            ws, next_row,
            'FIXED EFFECTS (log scale — exponentiate for ms-scale ratios)',
            fe_df, HEADER_FILL, SECTION_FILL)

        # OMNIBUS
        next_row = _write_section(
            ws, next_row,
            'OMNIBUS TESTS (Type-III, Kenward-Roger F-tests, on log RT)',
            omnibus_results[i], HEADER_FILL, SECTION_FILL)

        # MARGINAL MEANS
        mm_df = marginal_means[i].copy()
        if not mm_df.empty and 'condition' in mm_df.columns:
            cols = ['condition'] + [c for c in mm_df.columns if c != 'condition']
            mm_df = mm_df[cols]
        next_row = _write_section(
            ws, next_row,
            'MARGINAL MEANS (geometric mean RT, ms)',
            mm_df, HEADER_FILL, SECTION_FILL)

        # PAIRWISE
        ph_df = posthoc_results[i]
        ph_start = next_row
        next_row = _write_section(
            ws, next_row,
            'PAIRWISE COMPARISONS (ratios of geometric mean RT)',
            ph_df, HEADER_FILL, SECTION_FILL)
        if (not ph_df.empty
                and {'condition_1', 'condition_2'}.issubset(ph_df.columns)):
            data_start = ph_start + 2
            for r_offset, (_, row) in enumerate(ph_df.iterrows()):
                if count_differences(row['condition_1'],
                                     row['condition_2']) == 1:
                    for j in range(1, ph_df.shape[1] + 1):
                        ws.cell(row=data_start + r_offset,
                                column=j).fill = DIFF1_FILL

        # COLLAPSED MARGINAL MEANS / PAIRWISE
        collapse_extra = spec.get('collapse_over_extra', [])
        if collapse_extra:
            collapsed_label = ', '.join(collapse_extra)
            mm_c_df = marginal_means_collapsed[i].copy()
            if not mm_c_df.empty and 'condition' in mm_c_df.columns:
                cols = ['condition'] + [c for c in mm_c_df.columns
                                        if c != 'condition']
                mm_c_df = mm_c_df[cols]
            next_row = _write_section(
                ws, next_row,
                f'MARGINAL MEANS (collapsed over {collapsed_label}, ms)',
                mm_c_df, HEADER_FILL, SECTION_FILL)

            ph_c_df = posthoc_results_collapsed[i]
            ph_c_start = next_row
            next_row = _write_section(
                ws, next_row,
                f'PAIRWISE COMPARISONS (collapsed over {collapsed_label}, ratios)',
                ph_c_df, HEADER_FILL, SECTION_FILL)
            if (not ph_c_df.empty
                    and {'condition_1', 'condition_2'}.issubset(ph_c_df.columns)):
                data_start = ph_c_start + 2
                for r_offset, (_, row) in enumerate(ph_c_df.iterrows()):
                    if count_differences(row['condition_1'],
                                         row['condition_2']) == 1:
                        for j in range(1, ph_c_df.shape[1] + 1):
                            ws.cell(row=data_start + r_offset,
                                    column=j).fill = DIFF1_FILL

        # CELL COUNTS
        factors_for_count = [
            f for f in re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
            if f != 'log_rt' and f in rd.columns
        ]
        factors_for_count = list(dict.fromkeys(factors_for_count))
        cell_counts = (rd.groupby(factors_for_count, observed=True)
                         .size().reset_index(name='n_trials'))
        next_row = _write_section(ws, next_row, 'CELL COUNTS',
                                  cell_counts, HEADER_FILL, SECTION_FILL)

        # ANIMAL BREAKDOWN
        animal_breakdown = (
            rd.groupby(['animal'] + factors_for_count, observed=True)
              .agg(n_trials=('value', 'size'),
                   n_sessions=('session', 'nunique'))
              .reset_index()
        )
        next_row = _write_section(ws, next_row, 'ANIMAL BREAKDOWN',
                                  animal_breakdown,
                                  HEADER_FILL, SECTION_FILL)

        for col_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            ws.column_dimensions[col_letter].width = 18
        ws.column_dimensions['A'].width = 28

print(f"\nRT results workbook saved to: {results_workbook_path}")


# =============================================================================
# PLAIN-TEXT DUMP (matches the behavioural pipeline's archival format)
# =============================================================================

with open(os.path.join(save_dir, 'rt_model_results_detailed.txt'), 'w') as f:
    for i, spec in enumerate(model_specs):
        f.write("=" * 80 + "\n")
        f.write(f"MODEL {i}: {spec['name']}\n")
        f.write(f"Family:  LMM on log(RT)\n")
        f.write(f"Fixed:   {spec['fixed']}\n")
        f.write(f"Random:  {spec['random']}\n")
        f.write("=" * 80 + "\n\n")

        rd = row_data_dict[i]
        f.write(f"N trials:    {len(rd)}\n")
        f.write(f"N sessions:  {rd['session'].nunique()}\n")
        f.write(f"N animals:   {rd['animal'].nunique()}\n\n")

        f.write("FULL R SUMMARY:\n")
        try:
            summary_text = ro.r(f'capture.output(summary({fitted_models[i]}))')
            for line in summary_text:
                f.write(line + "\n")
        except Exception as e:
            f.write(f"(summary capture failed: {e})\n")
        f.write("\n")

        if not omnibus_results[i].empty:
            f.write("OMNIBUS TESTS (KR F):\n")
            f.write(omnibus_results[i].to_string(index=False))
            f.write("\n\n")

        if not posthoc_results[i].empty:
            f.write("PAIRWISE (ratios of geometric mean RT):\n")
            f.write(posthoc_results[i].to_string(index=False))
            f.write("\n\n")

        if not posthoc_results_collapsed[i].empty:
            collapse_extra = spec.get('collapse_over_extra', [])
            f.write(f"PAIRWISE COLLAPSED OVER {collapse_extra}:\n")
            f.write(posthoc_results_collapsed[i].to_string(index=False))
            f.write("\n\n")