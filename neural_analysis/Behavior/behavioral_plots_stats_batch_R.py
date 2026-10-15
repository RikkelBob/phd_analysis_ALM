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
import warnings
from itertools import product
from patsy import dmatrix
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from neural_analysis import plot_utils
from itertools import combinations
import config

plt.rcParams['font.family'] = 'sans-serif'
save_dir_dprime = config.FIGURE_DIR / "behavior/dprime"
save_dir = config.FIGURE_DIR / "behavior/responserate"
save_dir_dprime = save_dir_dprime.as_posix()
save_dir = save_dir.as_posix()


# =============================================================================
# WARNING HELPERS
# =============================================================================

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


def emmeans_vs_null_r(model_name_r, factors, model_type='lmm',
                      null_value=0, adjustment='holm',
                      collapse_over=None):
    """
    Test each estimated marginal mean against a null value (default 0).
    Uses the same df method as the rest of the model (KR for LMM).
    """
    if collapse_over is None:
        collapse_over = []
    keep_factors = [f for f in factors if f not in collapse_over]
    emmeans_spec = ' + '.join(keep_factors)

    type_arg = '"response"' if model_type == 'beta' else '"link"'
    mode_arg = '"asymptotic"' if model_type == 'beta' else '"kenward-roger"'

    try:
        ro.r(f'''
            library(emmeans)
            .emm <- emmeans({model_name_r}, ~ {emmeans_spec},
                            type = {type_arg}, mode = {mode_arg},
                            weights = "equal")
            .vs_null <- as.data.frame(
                test(.emm, null = {null_value}, adjust = "{adjustment}")
            )
        ''')
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.r('.vs_null'))

        df = df.rename(columns={
            'emmean': 'estimate', 'response': 'estimate',
            'SE': 'se', 'p.value': 'p_adjusted',
            't.ratio': 't_stat', 'z.ratio': 'z_stat',
        })
        df['significant'] = df['p_adjusted'] < 0.05
        df['null_value'] = null_value
        return df.reset_index(drop=True)
    except Exception as e:
        _warn_fallback(f"emmeans test vs null failed for {model_name_r} ({e})")
        return pd.DataFrame()

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

def diagnostics_r(model_name_r, model_idx, save_dir, model_type='lmm', n_sim=1000):
    """Save DHARMa diagnostic plots for any R model."""
    print(f"Saving diagnostics for {model_name_r}...")
    try:
        ro.r(f'''
            library(DHARMa)
            set.seed(42)
            tryCatch({{
                sim_res <- simulateResiduals(fittedModel = {model_name_r},
                                              plot = FALSE, n = {n_sim})
                png("{save_dir}/{model_name_r}_diagnostics.png",
                    width = 1200, height = 600)
                plot(sim_res)
                dev.off()

                png("{save_dir}/{model_name_r}_qq.png",
                    width = 800, height = 600)
                plotQQunif(sim_res)
                dev.off()

                png("{save_dir}/{model_name_r}_resid_vs_pred.png",
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


def plot_delta_left_right(delta_plot_dict, condition='unilateral', saliency='thr',
                          ax_size=(0.75, 1), alpha=0.4,
                          value='response rate', y_lim=None):
    """Plot delta values for left vs right responses for a single saliency level.

    ax_size: (width, height) of the data area in inches.
    """
    data = delta_plot_dict[condition]

    # Build figure with axes of exact physical size
    ax_w, ax_h = ax_size
    left, right = 0.4, 0.1
    bottom, top = 0.3, 0.2
    fig_w = left + ax_w + right
    fig_h = bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h,
                       ax_w / fig_w, ax_h / fig_h])

    x_pos = [0, 1]
    x_labels = ['Left', 'Right']

    # Pick colour based on saliency
    if saliency == 'thr':
        color = plot_utils.paper_colors("vis_thr")
        sal_label = 'Threshold'
    elif saliency == 'max':
        color = plot_utils.paper_colors("vis_max")
        sal_label = 'Maximum'
    else:
        raise ValueError(f"Unknown saliency: {saliency}")

    left_arr = np.array(data[saliency]['left'])
    right_arr = np.array(data[saliency]['right'])
    if len(left_arr) == len(right_arr):
        for l, r in zip(left_arr, right_arr):
            ax.plot([0, 1], [l, r], color=color, alpha=alpha,
                    linewidth=1, zorder=1)
    mean_left = np.mean(left_arr)
    mean_right = np.mean(right_arr)
    ax.plot([0, 1], [mean_left, mean_right],
            color=color, linewidth=2, marker='o', markersize=3,
            label=sal_label, zorder=3, alpha=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=6)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_ylabel(f'Δ {value}', fontsize=6)
    ax.set_xlabel('Side', fontsize=6)
    ax.set_title(f'{condition.capitalize()} — {sal_label}', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-0.15, 1.15])

    return fig, ax
    # DATA LOADING AND PREPARATION
# =============================================================================

parse = True
bool_stats = True
bool_plot = False
skip_RT = False
data_dir = config.DATA_DIR_EXP
figure_dir = config.FIGURE_DIR / "behavior/figures"
stats_dir = config.FIGURE_DIR / "behavior/stats"
pp_data_dir = config.DATA_DIR_PP

# Beta family choice: 'ordbeta' (handles 0/1 natively, recommended) or
# 'beta_smithson' (standard beta + Smithson-Verkuilen transform)
BETA_FAMILY = 'ordbeta'

# Multiple-comparison correction for pairwise contrasts
PAIRWISE_ADJUSTMENT = 'holm'

for d in [figure_dir, stats_dir, save_dir, save_dir_dprime]:
    if not os.path.isdir(d):
        os.makedirs(d)

# Load d-primes from matlab struct
data = utils.load_matlab_struct(
    pp_data_dir / "exp_beh_pp.mat"
)
data_df = pd.DataFrame(data['output'])

# Load raw data
trialData, sessionData, spikeData = utils.load_data(
    path_root=data_dir,
    experiment=['ALM_3', 'ALM_2'],
    animals=None,
    data_types=['trialData', 'sessionData'])

categorical_variables = ['hasphotostim', 'visualChangeSide', 'session_ID',
                         'mousename', 'optoNorm']

# Filter sessions with hit-rate < 0.3 for either modality
(trialData_filtered, sessionData_filtered, spikeData_filtered,
 sessions_exclude) = utils.filter_sessions_hitrate(
    trialData, sessionData=sessionData, spikeData=None, min_hitrate=0.3)

sessions_exclude = list(set(sessions_exclude))  # dedupe

for session in trialData_filtered['session_ID'].unique():
    trialData_ses = trialData[trialData['session_ID'] == session]
    if len(np.abs(trialData_ses['visualOriChange']).unique()) < 3:
        print(f"Session {session} excluded: <3 visual stimulus levels (need probe/thr/max)")
        sessions_exclude.append(session)
    if len(np.abs(trialData_ses['audioOctChange']).unique()) < 3:
        print(f"Session {session} excluded: <3 auditory stimulus levels (need probe/thr/max)")
        sessions_exclude.append(session)

sessions_exclude = list(set(sessions_exclude))
sessions_include = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]['session_ID']

sessions_unilat = data_df['session'][
    (data_df['opto'] == 'unilateral')
    & (~data_df['animal'].isin(["2063", "2065", "2066"]))
    & data_df['session'].isin(sessions_include)
].reset_index()

sessions_bilat = data_df['session'][
    (data_df['opto'] == 'bilateral')
    & data_df['session'].isin(sessions_include)
].reset_index()

sessions_bad_opto = data_df['session'][
    (data_df['animal'].isin(["2063", "2065", "2066"]))
    & (data_df['session'].isin(sessions_include))
].reset_index()

data_incl_unilat = data_df[data_df['session'].isin(sessions_unilat.session)].reset_index()
data_incl_bilat = data_df[data_df['session'].isin(sessions_bilat.session)].reset_index()
data_incl_bad = data_df[data_df['session'].isin(sessions_bad_opto.session)].reset_index().copy()
data_incl_bad['opto'] = 'exclude'

data_incl = pd.concat([data_incl_unilat, data_incl_bilat, data_incl_bad],
                      axis=0).reset_index()

df_long = utils.transform_df_long(data_incl)

# Drop NaN opto AND drop the explicitly-excluded "bad opto" animals
df_long = df_long[~df_long['opto'].isna()].copy()
df_long = df_long[df_long['opto'] != 'exclude'].copy()

# =============================================================================
# STATISTICAL MODELING
# =============================================================================

# =============================================================================
# 2x3 RESPONSE-RATE FIGURE: lick-left vs lick-right by opto condition
# Rows: visual / auditory / probe
# Cols: threshold / max  (probe duplicated across columns — see note)
# =============================================================================

def plot_lick_rates_opto(df_long, figure_dir=None, save_name='lick_rates_opto',
                        opto_conditions=('unilateral',),
                        figsize=(7, 7), alpha_lines=0.25):
    """
    Plot lick-left and lick-right rates for visual, auditory, and probe
    trials as a function of optogenetic condition.

    Parameters
    ----------
    df_long : pd.DataFrame
        Wide-format session-level dataframe with the response-rate columns
        listed in the script header (hitrate_*, errorrate_*, FArate_*,
        and their _opto counterparts).
    figure_dir : str or Path, optional
        If given, saves png + svg to figure_dir/png and figure_dir/svg.
    save_name : str
        Base filename (no extension).
    opto_conditions : tuple of str
        Which opto conditions to overlay against 'none'. Default: ('unilateral',).
        Pass ('unilateral', 'bilateral') to overlay both.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from scipy import stats as scistats

    # ---- column mapping ------------------------------------------------------
    # Each cell gives (lick_left_column, lick_right_column) for opto-OFF.
    # The opto-ON columns are the same names with '_opto' appended.
    col_map = {
        ('visual',   'thr'): ('hitrate_visual_thr',      'errorrate_visual_right_thr'),
        ('visual',   'max'): ('hitrate_visual_max',      'errorrate_visual_right_max'),
        ('auditory', 'thr'): ('errorrate_auditory_thr',  'hitrate_auditory_thr'),
        ('auditory', 'max'): ('errorrate_auditory_max',  'hitrate_auditory_max'),
        # Probe has no saliency split — duplicate FA rates in both columns
        ('probe',    'thr'): ('FArate_left',             'FArate_right'),
        ('probe',    'max'): ('FArate_left',             'FArate_right'),
    }

    row_labels = ['Visual', 'Auditory', 'Probe']
    col_labels = ['Threshold', 'Maximum']
    rows = ['visual', 'auditory', 'probe']
    cols = ['thr', 'max']

    # Colour scheme: black = opto off, then a palette for opto-on conditions
    opto_colors = {
        'none':       'black',
        'unilateral': '#1f77b4',   # blue
        'bilateral':  '#d62728',   # red
    }

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)

    for r, stim_type in enumerate(rows):
        for c, sal in enumerate(cols):
            ax = axes[r, c]
            left_col, right_col = col_map[(stim_type, sal)]

            # Restrict to sessions in opto_conditions (plus 'none' for off)
            opto_needed = ('none',) + tuple(opto_conditions)
            sub = df_long[df_long['opto'].isin(opto_needed)].copy()

            # --- opto-off: pool ALL included sessions (regardless of which
            # opto cohort they belong to) for the off baseline.
            off_left  = pd.to_numeric(sub[left_col],  errors='coerce').dropna()
            off_right = pd.to_numeric(sub[right_col], errors='coerce').dropna()

            # Plot opto-off mean ± SEM
            x = [0, 1]
            off_means = [off_left.mean(),  off_right.mean()]
            off_sems  = [off_left.sem(),   off_right.sem()]
            ax.errorbar(x, off_means, yerr=off_sems,
                        color=opto_colors['none'], marker='o', markersize=5,
                        linewidth=2, capsize=3, label='Control', zorder=3)

            # Faint per-session lines for opto-off
            paired_off = sub[[left_col, right_col]].apply(
                pd.to_numeric, errors='coerce').dropna()
            for _, row in paired_off.iterrows():
                ax.plot([0, 1], [row[left_col], row[right_col]],
                        color=opto_colors['none'], alpha=alpha_lines * 0.6,
                        linewidth=0.5, zorder=1)

            # --- opto-on: per-opto-condition overlay
            for opto in opto_conditions:
                sub_opto = sub[sub['opto'] == opto]
                left_opto_col  = left_col  + '_opto'
                right_opto_col = right_col + '_opto'
                if (left_opto_col not in sub_opto.columns
                        or right_opto_col not in sub_opto.columns):
                    continue

                on_left  = pd.to_numeric(sub_opto[left_opto_col],  errors='coerce').dropna()
                on_right = pd.to_numeric(sub_opto[right_opto_col], errors='coerce').dropna()

                on_means = [on_left.mean(),  on_right.mean()]
                on_sems  = [on_left.sem(),   on_right.sem()]
                ax.errorbar(x, on_means, yerr=on_sems,
                            color=opto_colors[opto], marker='o', markersize=5,
                            linewidth=2, capsize=3,
                            label=opto.capitalize(), zorder=3)

                # Faint per-session lines for this opto condition
                paired_on = sub_opto[[left_opto_col, right_opto_col]].apply(
                    pd.to_numeric, errors='coerce').dropna()
                for _, row in paired_on.iterrows():
                    ax.plot([0, 1], [row[left_opto_col], row[right_opto_col]],
                            color=opto_colors[opto], alpha=alpha_lines,
                            linewidth=0.5, zorder=1)

                # Paired Wilcoxon: off vs on, separately for left and right lick
                # (matched within session)
                for xi, (off_c, on_c) in enumerate(
                        [(left_col, left_opto_col), (right_col, right_opto_col)]):
                    paired = sub_opto[[off_c, on_c]].apply(
                        pd.to_numeric, errors='coerce').dropna()
                    if len(paired) >= 5 and (paired[off_c] != paired[on_c]).any():
                        try:
                            _, p = scistats.wilcoxon(paired[off_c], paired[on_c],
                                                     zero_method='zsplit')
                            if p < 0.001:
                                stars = '***'
                            elif p < 0.01:
                                stars = '**'
                            elif p < 0.05:
                                stars = '*'
                            else:
                                stars = ''
                            if stars:
                                y_top = max(off_means[xi], on_means[xi]) + 0.05
                                ax.text(xi, y_top, stars,
                                        ha='center', va='bottom',
                                        color=opto_colors[opto],
                                        fontsize=10, fontweight='bold')
                        except Exception:
                            pass

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Lick L', 'Lick R'])
            ax.set_ylim(-0.05, 1.05)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if c == 0:
                ax.set_ylabel(f'{row_labels[r]}\nresponse rate')
            if r == 0:
                ax.set_title(col_labels[c])

            # n indicator
            n_off = len(paired_off)
            ax.text(0.02, 0.98, f'n={n_off}',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=7, color='gray')

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()

    if figure_dir is not None:
        png_dir = os.path.join(figure_dir, 'png')
        svg_dir = os.path.join(figure_dir, 'svg')
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(svg_dir, exist_ok=True)
        fig.savefig(os.path.join(png_dir, save_name + '.png'),
                    format='png', dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(svg_dir, save_name + '.svg'),
                    format='svg', bbox_inches='tight')
        print(f"Saved: {save_name}.png / .svg in {figure_dir}")

    return fig, axes

fig, axes = plot_lick_rates_opto(
    data_incl,                          # <-- wide-format df with _opto columns
    figure_dir=figure_dir,
    save_name='lick_rates_vis_aud_probe_opto',
    opto_conditions=('unilateral',),    # change to ('unilateral','bilateral') if wanted
)


if bool_stats:
    # Define model specifications using a list of dicts
    model_specs = [
        # ----- D-prime models (Gaussian LMM) -----
        {
            'name': "Visual d-prime opto effect left vs right",
            'family': 'lmm',
            'fixed': "value ~ direction * opto * saliency",
            'random': "(1|session)",
            'filters': {
                'value_type': 'prime', 'modality': 'visual',
                'is_2ADC': True,
                'direction': ['left', 'right'],
                'saliency': ['thr', 'max'],
            },
            'collapse_over': [],
            'collapse_over_extra': ['direction'],
        },
        {
            'name': "Auditory d-prime",
            'family': 'lmm',
            'fixed': "value ~ saliency * opto",
            'random': "(1|session)",
            'filters': {
                'value_type': 'prime', 'modality': 'auditory',
                'is_2ADC': True,
                'saliency': ['thr', 'max'],
            },
            'collapse_over': [],
        },
        # ----- Response-rate models (Beta GLMM) -----
        {
            'name': "Visual responses opto left right",
            'family': 'beta',
            'fixed': "value ~ direction * opto * saliency",
            'random': "(1|session)",
            'filters': {
                'value_type': 'responserate', 'modality': 'visual',
                'response_direction': 'left',
                'direction': ['left', 'right'],
                'saliency': ['thr', 'max'],
            },
            'collapse_over': [],
            'collapse_over_extra': ['direction'],
        },
        {
            'name': "Auditory response",
            'family': 'beta',
            'fixed': "value ~ saliency * opto",
            'random': "(1|session)",
            'filters': {
                'value_type': 'responserate', 'modality': 'auditory',
                'response_direction': 'right',
                'saliency': ['thr', 'max'],
            },
            'collapse_over': [],
        },
        {
            'name': "False alarm rates under opto conditions",
            'family': 'beta',
            'fixed': "value ~ opto * response_direction",
            'random': "(1|session)",
            'filters': {
                'value_type': 'responserate', 'modality': 'none',
            },
            'collapse_over': [],
        },
    ]

    # Build per-model datasets
    row_data_dict = {}
    for i, spec in enumerate(model_specs):
        row_data = df_long.copy()
        for col, val in spec['filters'].items():
            if col not in df_long.columns:
                continue
            if isinstance(val, list):
                row_data = row_data[row_data[col].isin(val)]
            elif pd.notna(val):
                row_data = row_data[row_data[col] == val]
        row_data = row_data.copy()
        row_data['value'] = pd.to_numeric(row_data['value'], errors='coerce')
        row_data = row_data.dropna(subset=['value'])
        row_data_dict[i] = row_data

    # Initialise R packages
    base = importr('base')
    importr('lmerTest')
    importr('glmmTMB')
    importr('emmeans')
    importr('car')
    try:
        importr('performance')
    except Exception:
        _warn_fallback(
            "R package 'performance' not installed. R² values will be missing. "
            "Install with: install.packages('performance')"
        )
    importr('DHARMa')

    # Fit all models
    fitted_models = {}  # idx -> R object name
    for i, spec in enumerate(model_specs):
        # Determine which columns to keep
        factors_in_model = re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
        factors_in_model = [f for f in factors_in_model
                            if f != 'value' and f in row_data_dict[i].columns]
        factors_in_model = list(dict.fromkeys(factors_in_model))
        random_cols = ['session', 'animal']
        all_cols = list(set(factors_in_model + random_cols))

        data = row_data_dict[i][['value'] + all_cols].copy()

        if spec['family'] == 'lmm':
            r_name = fit_lmm_r(i, spec['fixed'], spec['random'],
                               data, all_cols)
        else:
            r_name = fit_beta_r(i, spec['fixed'], spec['random'],
                                data, all_cols, family=BETA_FAMILY)
        fitted_models[i] = r_name

    # Diagnostics
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    for i, spec in enumerate(model_specs):
        out_dir = save_dir_dprime if spec['family'] == 'lmm' else save_dir
        diagnostics_r(fitted_models[i], i, out_dir, model_type=spec['family'])

    print(f"Diagnostics plotted in {out_dir}")

    # Omnibus tests
    print("\n" + "=" * 60)
    print("OMNIBUS TESTS")
    print("=" * 60)
    omnibus_results = {}
    for i, spec in enumerate(model_specs):
        print(f"\n--- Model {i}: {spec['name']} ---")
        omni = omnibus_tests_r(fitted_models[i], model_type=spec['family'])
        omnibus_results[i] = omni
        if not omni.empty:
            print(omni.to_string(index=False))
        else:
            print("(no omnibus results — see warnings above)")

    # Marginal means and pairwise comparisons
    print("\n" + "=" * 60)
    print("MARGINAL MEANS AND PAIRWISE COMPARISONS")
    print("=" * 60)

    marginal_means = {}
    posthoc_results = {}
    for i, spec in enumerate(model_specs):
        print(f"\n--- Model {i}: {spec['name']} ---")
        factors = re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
        factors = [f for f in factors if f != 'value']
        factors = list(dict.fromkeys(factors))

        mm = get_marginal_means_r(fitted_models[i], factors,
                                   model_type=spec['family'])
        marginal_means[i] = mm
        if not mm.empty:
            print("\nMarginal Means:")
            cols = ['condition', 'marginal_mean', 'se', 'ci_lower', 'ci_upper']
            cols = [c for c in cols if c in mm.columns]
            print(mm[cols].to_string(index=False))

        ph = pairwise_comparisons_r(
            fitted_models[i], factors, model_type=spec['family'],
            adjustment=PAIRWISE_ADJUSTMENT,
            collapse_over=spec['collapse_over'],
        )
        posthoc_results[i] = ph
        if not ph.empty:
            print("\nPairwise Comparisons:")
            cols = ['condition_1', 'condition_2', 'diff', 'ratio',
                    'p_adjusted', 'significant']
            cols = [c for c in cols if c in ph.columns]
            print(ph[cols].to_string(index=False))

    # =========================================================================
    # COLLAPSED MARGINAL MEANS AND PAIRWISE COMPARISONS
    # (average over factors listed in `collapse_over_extra`)
    # =========================================================================
    print("\n" + "=" * 60)
    print("COLLAPSED MARGINAL MEANS AND PAIRWISE COMPARISONS")
    print("=" * 60)

    marginal_means_collapsed = {}
    posthoc_results_collapsed = {}

    for i, spec in enumerate(model_specs):
        collapse_extra = spec.get('collapse_over_extra', [])
        if not collapse_extra:
            marginal_means_collapsed[i] = pd.DataFrame()
            posthoc_results_collapsed[i] = pd.DataFrame()
            continue

        print(f"\n--- Model {i}: {spec['name']} "
              f"(collapsed over {collapse_extra}) ---")

        factors = re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
        factors = [f for f in factors if f != 'value']
        factors = list(dict.fromkeys(factors))

        mm_c = get_marginal_means_r(
            fitted_models[i], factors,
            model_type=spec['family'],
            collapse_over=collapse_extra,
        )
        marginal_means_collapsed[i] = mm_c
        if not mm_c.empty:
            print("\nCollapsed Marginal Means:")
            cols = ['condition', 'marginal_mean', 'se', 'ci_lower', 'ci_upper']
            cols = [c for c in cols if c in mm_c.columns]
            print(mm_c[cols].to_string(index=False))

        ph_c = pairwise_comparisons_r(
            fitted_models[i], factors,
            model_type=spec['family'],
            adjustment=PAIRWISE_ADJUSTMENT,
            collapse_over=collapse_extra,
        )
        posthoc_results_collapsed[i] = ph_c
        if not ph_c.empty:
            print("\nCollapsed Pairwise Comparisons:")
            cols = ['condition_1', 'condition_2', 'diff', 'ratio',
                    'p_adjusted', 'significant']
            cols = [c for c in cols if c in ph_c.columns]
            print(ph_c[cols].to_string(index=False))

        # =========================================================================
        # TESTS AGAINST NULL (zero for d-prime, 0.5 / link=0 for response rate)
        # =========================================================================
        print("\n" + "=" * 60)
        print("TESTS AGAINST NULL")
        print("=" * 60)

        vs_null_results = {}
        vs_null_results_collapsed = {}

        for i, spec in enumerate(model_specs):
            # Only test d-prime models against zero — for response rates,
            # zero isn't a meaningful null and the boundary makes the SE
            # asymmetric. Skip beta models here.
            if spec['family'] != 'lmm':
                vs_null_results[i] = pd.DataFrame()
                vs_null_results_collapsed[i] = pd.DataFrame()
                continue

            print(f"\n--- Model {i}: {spec['name']} (vs zero) ---")
            factors = re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
            factors = [f for f in factors if f != 'value']
            factors = list(dict.fromkeys(factors))

            vn = emmeans_vs_null_r(
                fitted_models[i], factors,
                model_type=spec['family'],
                null_value=0,
                adjustment=PAIRWISE_ADJUSTMENT,
            )
            vs_null_results[i] = vn
            if not vn.empty:
                cols = ['condition', 'estimate', 'se', 't_stat',
                        'p_adjusted', 'significant']
                cols = [c for c in cols if c in vn.columns]
                print(vn[cols].to_string(index=False))

            # Collapsed version, if requested
            collapse_extra = spec.get('collapse_over_extra', [])
            if collapse_extra:
                print(f"--- Model {i}: vs zero (collapsed over {collapse_extra}) ---")
                vn_c = emmeans_vs_null_r(
                    fitted_models[i], factors,
                    model_type=spec['family'],
                    null_value=0,
                    adjustment=PAIRWISE_ADJUSTMENT,
                    collapse_over=collapse_extra,
                )
                vs_null_results_collapsed[i] = vn_c
                if not vn_c.empty:
                    cols = ['condition', 'estimate', 'se', 't_stat',
                            'p_adjusted', 'significant']
                    cols = [c for c in cols if c in vn_c.columns]
                    print(vn_c[cols].to_string(index=False))
            else:
                vs_null_results_collapsed[i] = pd.DataFrame()

    # =============================================================================
    # CLAIMS 1 & 2: direction effects — difference test only (no equivalence/TOST).
    #
    # Claim 1: left vs right at opto = 'none' (simple effect of direction, opto OFF)
    # Claim 2: differential opto effect across sides for a given opto level
    #          (direction x opto interaction contrast = difference-of-differences)
    #
    # Both are estimable contrasts -> invariant to factor coding (contr.treatment
    # vs contr.sum give identical estimates/SEs), so no contr.sum needed here.
    #
    # Scale:
    #   - LMM (d-prime): identity scale, KR df. Estimate in d' units.
    #   - Beta (response rate): regridded to RESPONSE (probability) scale, so the
    #     contrast is a difference of probabilities. df asymptotic (z).
    #
    # Interpretation:
    #   p < alpha and 95% CI excludes 0 -> 'different'
    #   otherwise                       -> 'no significant difference'
    #
    # PLACEMENT: paste inside `if bool_stats:` AFTER the model-fitting loop has
    # populated `fitted_models` and `model_specs`. (Set bool_stats=True.)
    # =============================================================================

    ALPHA = 0.05

    def _scale_setup(model_type):
        """Return (emmeans mode arg, whether to regrid to response scale)."""
        if model_type == 'lmm':
            return '"kenward-roger"', False
        return '"asymptotic"', True


    _STAT_RENAME = {
        'SE': 'se', 'p.value': 'p_value',
        't.ratio': 'stat', 'z.ratio': 'stat',
        'lower.CL': 'ci_low', 'upper.CL': 'ci_high',
        'asymp.LCL': 'ci_low', 'asymp.UCL': 'ci_high',
    }


    def _tidy(diff, alpha, scale_label):
        out = diff.rename(columns={k: v for k, v in _STAT_RENAME.items()
                                   if k in diff.columns}).copy()
        out['scale'] = scale_label
        out['significant'] = out['p_value'] < alpha if 'p_value' in out.columns else False
        return out.reset_index(drop=True)


    def claim1_direction_at_off(model_name_r, model_type='lmm',
                                off_level='none', alpha=ALPHA, scale_label=''):
        """Claim 1: left vs right at opto = off_level, per saliency AND pooled."""
        mode_arg, regrid = _scale_setup(model_type)
        regrid_line = '.base <- regrid(.base)' if regrid else ''
        level = 1 - alpha

        frames = []
        grids = [('~ direction * saliency', None),  # per-saliency
                 ('~ direction', 'pooled')]  # averaged over saliency

        for report_spec, sal_tag in grids:
            try:
                ro.r(f'''
                    library(emmeans)
                    .base <- emmeans({model_name_r}, ~ direction * saliency,
                                     at = list(opto = c("{off_level}")),
                                     mode = {mode_arg}, weights = "equal")
                    {regrid_line}
                    .emm  <- emmeans(.base, {report_spec})
                    .c1   <- contrast(.emm, "pairwise", simple = "direction")
                    .diff <- as.data.frame(summary(.c1, infer = c(TRUE, TRUE),
                                                   level = {level}))
                ''')
                with localconverter(ro.default_converter + pandas2ri.converter):
                    diff = ro.conversion.rpy2py(ro.r('.diff'))
            except Exception as e:
                _warn_fallback(f"claim1 ({model_name_r}, grid={report_spec}) "
                               f"failed: {e}")
                continue

            tidy = _tidy(diff, alpha, scale_label)
            if sal_tag == 'pooled':
                tidy['saliency'] = 'pooled'
            tidy.insert(0, 'opto', off_level)
            frames.append(tidy)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


    def claim2_differential_opto(model_name_r, model_type='lmm',
                                 opto_level='unilateral', off_level='none',
                                 alpha=ALPHA, scale_label=''):
        """Claim 2: does the opto effect differ between left and right?

        Difference-of-differences:
            [ (off - opto_level) | left ] - [ (off - opto_level) | right ]
        Read opto_pairwise / direction_pairwise label columns for the sign.
        emmeans orders by level order (none<unilateral, left<right), so the
        estimate is (none-unilateral)|left  -  (none-unilateral)|right.
        """
        mode_arg, regrid = _scale_setup(model_type)
        regrid_line = '.base <- regrid(.base)' if regrid else ''
        level = 1 - alpha

        frames = []
        # (reported grid, extra `by` arg for the contrast, saliency tag)
        grids = [('~ direction * opto * saliency', ', by = "saliency"', None),  # per-saliency
                 ('~ direction * opto', '', 'pooled')]

        for report_spec, by_arg, sal_tag in grids:
            try:
                ro.r(f'''
                    library(emmeans)
                    .base <- emmeans({model_name_r}, ~ direction * opto * saliency,
                                     at = list(opto = c("{off_level}", "{opto_level}")),
                                     mode = {mode_arg}, weights = "equal")
                    {regrid_line}
                    .emm <- emmeans(.base, {report_spec})
                    .ic  <- contrast(.emm,
                                     interaction = c(opto = "pairwise",
                                                     direction = "pairwise"){by_arg})
                    .diff <- as.data.frame(summary(.ic, infer = c(TRUE, TRUE),
                                                   level = {level}))
                ''')
                with localconverter(ro.default_converter + pandas2ri.converter):
                    diff = ro.conversion.rpy2py(ro.r('.diff'))
            except Exception as e:
                _warn_fallback(f"claim2 ({model_name_r}, grid={report_spec}) "
                               f"failed: {e}")
                continue

            tidy = _tidy(diff, alpha, scale_label)
            if sal_tag == 'pooled':
                tidy['saliency'] = 'pooled'
            tidy.insert(0, 'opto_contrast', f'{off_level} vs {opto_level}')
            frames.append(tidy)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


    # =============================================================================
    # DRIVER — runs both claims on every model with direction AND opto
    # (visual d-prime LMM and visual response-rate beta model).
    # =============================================================================
    claim1_results = {}
    claim2_results = {}

    print("\n" + "=" * 60)
    print(f"CLAIMS 1 & 2 — DIFFERENCE TEST (alpha = {ALPHA})")
    print("=" * 60)

    for i, spec in enumerate(model_specs):
        facs = re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
        facs = set(f for f in facs if f != 'value')
        if not {'direction', 'opto'} <= facs:
            continue

        mt = spec['family']
        scale_label = "d-prime (d')" if mt == 'lmm' else "response rate (prob.)"
        print(f"\n--- Model {i}: {spec['name']}  [{scale_label}] ---")

        c1 = claim1_direction_at_off(fitted_models[i], mt, scale_label=scale_label)
        c2 = claim2_differential_opto(fitted_models[i], mt,
                                      opto_level='unilateral',
                                      scale_label=scale_label)
        claim1_results[i] = c1
        claim2_results[i] = c2

        cols = ['saliency', 'estimate', 'se', 'ci_low', 'ci_high',
                'stat', 'p_value', 'significant']
        print("\n  CLAIM 1 — left vs right, opto OFF:")
        if not c1.empty:
            print(c1[[c for c in cols if c in c1.columns]].to_string(index=False))
        print("\n  CLAIM 2 — differential unilateral-opto effect across sides:")
        if not c2.empty:
            print(c2[[c for c in cols if c in c2.columns]].to_string(index=False))


    def joint_tests_marginal_r(model_name_r, model_type='lmm', alpha=0.05):
        """Coding-invariant marginal Type-III omnibus via emmeans::joint_tests.

        LMM:  Kenward-Roger denominator df (matches the rest of your d-prime models).
        Beta: asymptotic (df2 = Inf), since KR doesn't apply to glmmTMB.

        Returns tidy df: term, df1, df2, F, p_value, significant.
        """
        # Only LMMs get lmer.df; passing it to a glmmTMB ref_grid would error.
        extra = ', lmer.df = "kenward-roger"' if model_type == 'lmm' else ''
        try:
            ro.r(f'''
                library(emmeans)
                .jt <- as.data.frame(joint_tests({model_name_r}{extra}))
            ''')
            with localconverter(ro.default_converter + pandas2ri.converter):
                df = ro.conversion.rpy2py(ro.r('.jt'))
        except Exception as e:
            _warn_fallback(f"joint_tests failed for {model_name_r} ({e})")
            return pd.DataFrame()

        # Normalise column names (rpy2 turns 'model term' into 'model.term')
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if 'term' in cl or cl.startswith('model'):
                rename[c] = 'term'
            elif c == 'F.ratio':
                rename[c] = 'F'
            elif c == 'p.value':
                rename[c] = 'p_value'
        df = df.rename(columns=rename)
        if 'term' in df.columns:
            df['term'] = df['term'].astype(str).str.strip()
        if 'p_value' in df.columns:
            df['significant'] = df['p_value'] < alpha
        return df.reset_index(drop=True)

    # --- driver: marginal omnibus for every model -------------------------------
    joint_test_results = {}
    print("\n" + "=" * 60)
    print("MARGINAL OMNIBUS (joint_tests, KR df for LMMs)")
    print("=" * 60)
    for i, spec in enumerate(model_specs):
        print(f"\n--- Model {i}: {spec['name']} ---")
        jt = joint_tests_marginal_r(fitted_models[i], model_type=spec['family'])
        joint_test_results[i] = jt
        if not jt.empty:
            cols = [c for c in ['term', 'df1', 'df2', 'F', 'p_value', 'significant']
                    if c in jt.columns]
            print(jt[cols].to_string(index=False))

    # =========================================================================
    # UNIFIED EXCEL WORKBOOK
    # =========================================================================
    from openpyxl.styles import PatternFill, Font, Alignment

    SECTION_FILL = PatternFill(start_color='305496', end_color='305496', fill_type='solid')
    HEADER_FILL = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')
    DIFF1_FILL = PatternFill(start_color='FFF2CC', end_color='FFF2CC', fill_type='solid')

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

    def _write_section(ws, start_row, title, df, header_fill, section_fill):
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

        return header_row + 1 + len(df) + 2

    results_workbook_path = os.path.join(stats_dir, 'behavioral_results.xlsx')

    # Build overview
    overview_rows = []
    for i, spec in enumerate(model_specs):
        rd = row_data_dict[i]
        overview_rows.append({
            'model_idx': i,
            'model_name': spec['name'],
            'family': 'LMM (Gaussian, lmerTest)' if spec['family'] == 'lmm'
                      else f'Beta GLMM ({BETA_FAMILY}, glmmTMB)',
            'fixed_formula': spec['fixed'],
            'random_formula': spec['random'],
            'n_obs': len(rd),
            'n_sessions': rd['session'].nunique(),
            'n_animals': rd['animal'].nunique() if 'animal' in rd.columns else None,
            'pairwise_adjustment': PAIRWISE_ADJUSTMENT,
        })
    overview_df = pd.DataFrame(overview_rows)

    # README content
    readme_lines = [
        "Behavioral statistics workbook",
        "",
        "Each subsequent sheet corresponds to one statistical model.",
        "Sheets contain the following sections:",
        "  - MODEL INFO: design, n, formulas, software",
        "  - FIT STATISTICS: AIC/BIC/logLik, R² (marginal/conditional),",
        "    random-effects variance components, and beta dispersion (φ)",
        "  - FIXED EFFECTS: coefficients on the link/identity scale",
        "  - OMNIBUS TESTS: Type-III tests",
        "      - LMM: F-tests with Kenward-Roger denominator df",
        "      - Beta: Wald χ² via car::Anova",
        "  - MARGINAL MEANS: emmeans estimates, on response scale for beta",
        "  - PAIRWISE COMPARISONS: emmeans contrasts, "
        f"adjusted with {PAIRWISE_ADJUSTMENT}",
        "      Sign convention: higher_opto - lower_opto",
        "      Yellow rows: contrasts where conditions differ in only 1 factor",
        "  - CELL COUNTS: n observations per condition",
        "  - ANIMAL BREAKDOWN: n sessions per animal × condition",
        "",
        "Software:",
        f"  - Beta family: {BETA_FAMILY}",
        f"  - Pairwise adjustment: {PAIRWISE_ADJUSTMENT}",
        f"  - Random-effects structure: nested (1|session)",
        f"  - LMM df: Kenward-Roger via lmerTest",
        "",
        "Exclusion criteria applied to source data:",
        "  - Sessions with hit-rate < 0.3 in either modality",
        "  - Sessions with < 3 unique stimulus levels in either modality",
        "  - Animals 2063, 2065, 2066 excluded (bad opto)",
        "  - MARGINAL MEANS (collapsed over <factor>): emmeans estimates",
        "    averaged over the listed factor with equal weights, on the same",
        "    fitted model (no refit). Used when an interaction with that",
        "    factor is not significant and a simpler summary is desired.",
        "  - PAIRWISE COMPARISONS (collapsed over <factor>): same logic",
        f"    applied to contrasts, adjusted with {PAIRWISE_ADJUSTMENT}.",
        "  - TESTS VS NULL: each marginal mean tested against 0 (chance",
        "    d-prime). Two-sided t-tests using the same KR df as the",
        f"    rest of the model. Adjusted with {PAIRWISE_ADJUSTMENT} within",
        "    each model. Only reported for d-prime (LMM) models — for",
        "    response rates, 0 is a boundary, not a meaningful null.",
    ]
    readme_df = pd.DataFrame({'README': readme_lines})

    # Compute direction claims 1 & 2 so they can be written into the per-model sheets
    claim1_results, claim2_results = {}, {}
    for i, spec in enumerate(model_specs):
        facs = set(f for f in re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
                   if f != 'value')
        if not {'direction', 'opto'} <= facs:
            continue
        mt = spec['family']
        scale_label = "d-prime (d')" if mt == 'lmm' else "response rate (prob.)"
        claim1_results[i] = claim1_direction_at_off(
            fitted_models[i], mt, scale_label=scale_label)
        claim2_results[i] = claim2_differential_opto(
            fitted_models[i], mt, opto_level='unilateral', scale_label=scale_label)

    with pd.ExcelWriter(results_workbook_path, engine='openpyxl', mode='w') as writer:
        # README sheet
        readme_df.to_excel(writer, sheet_name='README', index=False)
        ws_rm = writer.sheets['README']
        ws_rm.column_dimensions['A'].width = 90

        # Overview sheet
        overview_df.to_excel(writer, sheet_name='Overview', index=False)
        ws_ov = writer.sheets['Overview']
        for j in range(1, overview_df.shape[1] + 1):
            c = ws_ov.cell(row=1, column=j)
            c.font = Font(bold=True)
            c.fill = HEADER_FILL
        widths = {'model_idx': 10, 'model_name': 40, 'family': 30,
                  'fixed_formula': 45, 'random_formula': 22,
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

            # Section 1: Model info
            info_df = pd.DataFrame({
                'field': [
                    'Model index', 'Model name', 'Family', 'Fixed formula',
                    'Random formula', 'N observations', 'N sessions',
                    'N animals', 'Beta family',
                    'Pairwise adjustment', 'LMM df method',
                ],
                'value': [
                    i, spec['name'],
                    'LMM (Gaussian, lmerTest)' if spec['family'] == 'lmm'
                    else f'Beta GLMM ({BETA_FAMILY}, glmmTMB)',
                    spec['fixed'], spec['random'],
                    len(rd), rd['session'].nunique(),
                    rd['animal'].nunique() if 'animal' in rd.columns else None,
                    BETA_FAMILY if spec['family'] == 'beta' else 'n/a',
                    PAIRWISE_ADJUSTMENT,
                    'Kenward-Roger' if spec['family'] == 'lmm' else 'asymptotic',
                ]
            })
            next_row = _write_section(ws, 1, 'MODEL INFO', info_df,
                                      HEADER_FILL, SECTION_FILL)

            # Section 2: Fit statistics
            fit_stats = get_model_fit_stats(fitted_models[i],
                                             model_type=spec['family'])
            fit_df = pd.DataFrame({'statistic': list(fit_stats.keys()),
                                   'value': list(fit_stats.values())})
            next_row = _write_section(ws, next_row, 'FIT STATISTICS',
                                      fit_df, HEADER_FILL, SECTION_FILL)

            # Section 3: Fixed effects
            fe_df = get_fixed_effects_r(fitted_models[i],
                                         model_type=spec['family'])
            next_row = _write_section(ws, next_row, 'FIXED EFFECTS',
                                      fe_df, HEADER_FILL, SECTION_FILL)

            # Section 4: Omnibus tests DEPRECIATED
            omni_title = ('OMNIBUS TESTS (Type-III, Kenward-Roger F-tests)'
                          if spec['family'] == 'lmm'
                          else 'OMNIBUS TESTS (Type-III Wald χ² via car::Anova)')
            next_row = _write_section(ws, next_row, omni_title,
                                      omnibus_results[i],
                                      HEADER_FILL, SECTION_FILL)
            # NEW MARGINAL OMNIBUS
            next_row = _write_section(
                ws, next_row,
                'MARGINAL OMNIBUS (joint_tests, Type-III, KR df)',
                joint_test_results.get(i, pd.DataFrame()),
                HEADER_FILL, SECTION_FILL)

            # Section 5: Marginal means
            mm_df = marginal_means[i].copy()
            if not mm_df.empty and 'condition' in mm_df.columns:
                cols = ['condition'] + [c for c in mm_df.columns if c != 'condition']
                mm_df = mm_df[cols]
            next_row = _write_section(ws, next_row, 'MARGINAL MEANS',
                                      mm_df, HEADER_FILL, SECTION_FILL)

            # Section 6: Pairwise comparisons
            ph_df = posthoc_results[i]
            ph_start = next_row
            next_row = _write_section(ws, next_row, 'PAIRWISE COMPARISONS',
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

            # Section: Direction claims 1 & 2 (models with direction AND opto)
            _facs = set(f for f in re.findall(r'\b\w+\b',
                                              spec['fixed'].split('~')[1]) if f != 'value')
            if {'direction', 'opto'} <= _facs:
                next_row = _write_section(
                    ws, next_row,
                    'CLAIM 1 — left vs right, opto OFF (difference test)',
                    claim1_results.get(i, pd.DataFrame()),
                    HEADER_FILL, SECTION_FILL)
                next_row = _write_section(
                    ws, next_row,
                    'CLAIM 2 — differential unilateral-opto effect across sides',
                    claim2_results.get(i, pd.DataFrame()),
                    HEADER_FILL, SECTION_FILL)

            # Section 6b: Collapsed marginal means
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
                    f'MARGINAL MEANS (collapsed over {collapsed_label})',
                    mm_c_df, HEADER_FILL, SECTION_FILL)

                # Section 6c: Collapsed pairwise comparisons
                ph_c_df = posthoc_results_collapsed[i]
                ph_c_start = next_row
                next_row = _write_section(
                    ws, next_row,
                    f'PAIRWISE COMPARISONS (collapsed over {collapsed_label})',
                    ph_c_df, HEADER_FILL, SECTION_FILL)
                if (not ph_c_df.empty
                        and {'condition_1', 'condition_2'}.issubset(
                            ph_c_df.columns)):
                    data_start = ph_c_start + 2
                    for r_offset, (_, row) in enumerate(ph_c_df.iterrows()):
                        if count_differences(row['condition_1'],
                                             row['condition_2']) == 1:
                            for j in range(1, ph_c_df.shape[1] + 1):
                                ws.cell(row=data_start + r_offset,
                                        column=j).fill = DIFF1_FILL

            # Section 6d: Tests vs null (d-prime models only)
            if spec['family'] == 'lmm':
                vn_df = vs_null_results[i].copy()
                if not vn_df.empty and 'condition' in vn_df.columns:
                    cols = ['condition'] + [c for c in vn_df.columns
                                            if c != 'condition']
                    vn_df = vn_df[cols]
                next_row = _write_section(
                    ws, next_row,
                    'TESTS VS NULL (d-prime vs 0, chance performance)',
                    vn_df, HEADER_FILL, SECTION_FILL)

                # Collapsed version
                collapse_extra = spec.get('collapse_over_extra', [])
                if collapse_extra:
                    collapsed_label = ', '.join(collapse_extra)
                    vn_c_df = vs_null_results_collapsed[i].copy()
                    if (not vn_c_df.empty
                            and 'condition' in vn_c_df.columns):
                        cols = ['condition'] + [c for c in vn_c_df.columns
                                                if c != 'condition']
                        vn_c_df = vn_c_df[cols]
                    next_row = _write_section(
                        ws, next_row,
                        f'TESTS VS NULL (collapsed over {collapsed_label})',
                        vn_c_df, HEADER_FILL, SECTION_FILL)

            # Section 7: Cell counts
            try:
                factors_for_count = [
                    f for f in re.findall(r'\b\w+\b', spec['fixed'].split('~')[1])
                    if f != 'value' and f in rd.columns
                ]
                factors_for_count = list(dict.fromkeys(factors_for_count))
                if factors_for_count:
                    cell_counts = (rd.groupby(factors_for_count, observed=True)
                                     .size().reset_index(name='n_obs'))
                else:
                    cell_counts = pd.DataFrame({'n_obs': [len(rd)]})
            except Exception:
                cell_counts = pd.DataFrame()
            next_row = _write_section(ws, next_row, 'CELL COUNTS',
                                      cell_counts, HEADER_FILL, SECTION_FILL)

            # Section 8: Animal breakdown
            try:
                if 'animal' in rd.columns:
                    animal_breakdown = (
                        rd.groupby(['animal'] + factors_for_count, observed=True)
                          .agg(n_obs=('value', 'size'),
                               n_sessions=('session', 'nunique'))
                          .reset_index()
                    )
                else:
                    animal_breakdown = pd.DataFrame()
            except Exception:
                animal_breakdown = pd.DataFrame()
            next_row = _write_section(ws, next_row, 'ANIMAL BREAKDOWN',
                                      animal_breakdown,
                                      HEADER_FILL, SECTION_FILL)

            # Column widths
            for col_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                ws.column_dimensions[col_letter].width = 18
            ws.column_dimensions['A'].width = 28

    print(f"\nUnified results workbook saved to: {results_workbook_path}")

    # Also write a plain-text dump for archival / quick inspection
    with open(os.path.join(stats_dir, 'model_results_detailed.txt'), 'w') as f:
        for i, spec in enumerate(model_specs):
            f.write("=" * 80 + "\n")
            f.write(f"MODEL {i}: {spec['name']}\n")
            f.write(f"Family:  {spec['family']}\n")
            f.write(f"Fixed:   {spec['fixed']}\n")
            f.write(f"Random:  {spec['random']}\n")
            f.write("=" * 80 + "\n\n")

            rd = row_data_dict[i]
            f.write(f"N observations: {len(rd)}\n")
            f.write(f"N sessions:     {rd['session'].nunique()}\n")
            f.write(f"N animals:      "
                    f"{rd['animal'].nunique() if 'animal' in rd.columns else 'n/a'}\n\n")

            f.write("FULL R SUMMARY:\n")
            try:
                summary_text = ro.r(f'capture.output(summary({fitted_models[i]}))')
                for line in summary_text:
                    f.write(line + "\n")
            except Exception as e:
                f.write(f"(summary capture failed: {e})\n")
            f.write("\n")

            if not omnibus_results[i].empty:
                f.write("OMNIBUS TESTS:\n")
                f.write(omnibus_results[i].to_string(index=False))
                f.write("\n\n")

            if not posthoc_results[i].empty:
                f.write("PAIRWISE COMPARISONS:\n")
                f.write(posthoc_results[i].to_string(index=False))
                f.write("\n\n")


# =============================================================================
# PLOTTING
# =============================================================================

if bool_plot:
    fig_opto, axes_opto = plt.subplots(2, 2, figsize=(3.5, 4))
    colors_vis_resp = {
        "probe": plot_utils.paper_colors("probe"),
        "thr": plot_utils.paper_colors("vis_thr"),
        "max": plot_utils.paper_colors("vis_max"),
    }
    colors_aud_resp = {
        "probe": plot_utils.paper_colors("probe"),
        "thr": plot_utils.paper_colors("aud_thr"),
        "max": plot_utils.paper_colors("aud_max"),
    }
    colors_vis_dprime = {
        "thr": plot_utils.paper_colors("vis_thr"),
        "max": plot_utils.paper_colors("vis_max"),
    }
    colors_aud_dprime = {
        "thr": plot_utils.paper_colors("aud_thr"),
        "max": plot_utils.paper_colors("aud_max"),
    }

    # --- Visual responses for probe/thr/max ---
    print("Visual responses for probe/thr/max")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['opto'] == 'none')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    plt_vis_resp = utils.plot_pairwise_effects_multiple_conditions(
        df_long_plot, condition_column='saliency',
        conditions=['probe', 'thr', 'max'],
        animal_column='animal', selected_value_type='responserate')
    filename = "vis_resp"
    plt_vis_resp.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_vis_resp.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Auditory responses for probe/thr/max ---
    print("Auditory responses for probe/thr/max")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['opto'] == 'none')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'auditory'])]
    plt_aud_resp = utils.plot_pairwise_effects_multiple_conditions(
        df_long_plot, condition_column='saliency',
        conditions=['probe', 'thr', 'max'],
        animal_column='animal', selected_value_type='responserate')
    filename = "aud_resp"
    plt_aud_resp.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_aud_resp.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Visual errors ---
    print("Visual errors")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all'])
                           & (df_long['modality']).isin(['visual'])
                           & (df_long['saliency'] == 'all')]
    plt_vis_err = utils.plot_pairwise_effects_multiple_conditions(
        df_long_plot, condition_column='opto',
        conditions=['none', 'unilateral', 'bilateral'],
        animal_column='animal', selected_value_type='responserate')
    filename = "vis_err"
    plt_vis_err.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_vis_err.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Auditory errors ---
    print("Auditory errors")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all'])
                           & (df_long['modality']).isin(['auditory'])
                           & (df_long['saliency'] == 'all')]
    plt_aud_err = utils.plot_pairwise_effects_multiple_conditions(
        df_long_plot, condition_column='opto',
        conditions=['none', 'unilateral', 'bilateral'],
        animal_column='animal', selected_value_type='responserate')
    filename = "aud_err"
    plt_aud_err.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_aud_err.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Response rates for opto conditions - Visual ---
    print("Response rates for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='saliency', conditions=['probe', 'thr', 'max'],
        grouping_column='opto', groups=['none', 'unilateral', 'bilateral'],
        selected_value_type='responserate', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['Probe', 'Threshold', 'Maximum'])
    ax.set_ylabel("Response rate")
    filename = "vis_resp_multi_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- MAIN FIGURE Response rates for opto conditions - Visual - alt ---
    print("Response rates for opto conditions - Visual (alt)")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['probe', 'thr', 'max'],
        selected_value_type='responserate', alpha_marker=0.5, error_type='scatter',
        ax=axes_opto[0, 0], show_legend=False, group_colors=colors_vis_resp)
    ax.set_xticklabels([])
    ax.set_ylabel("Response rate")
    ax.set_xlabel("")

    # --- Response rates for opto conditions - Visual - Left alt ---
    print("Response rates for opto conditions - Visual Left")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['left', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['probe', 'thr', 'max'],
        selected_value_type='responserate', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['None', 'Unilateral', 'Bilateral'])
    ax.set_ylabel("Response rate")
    filename = "vis_resp_left_multi_opto_alt"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Response rates for opto conditions - Visual - Right alt ---
    print("Response rates for opto conditions - Visual Right")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['right', 'none'])
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['probe', 'thr', 'max'],
        selected_value_type='responserate', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['None', 'Unilateral', 'Bilateral'])
    ax.set_ylabel("Response rate")
    filename = "vis_resp_right_multi_opto_alt"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Response rates for opto conditions - Auditory ---
    print("Response rates for opto conditions - Auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'auditory'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='saliency', conditions=['probe', 'thr', 'max'],
        grouping_column='opto', groups=['none', 'unilateral', 'bilateral'],
        selected_value_type='responserate', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['Probe', 'Threshold', 'Maximum'])
    ax.set_ylabel("Response rate")
    filename = "aud_resp_multi_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- MAIN FIGURE Response rates - Auditory - alt ---
    print("Response rates for opto conditions - Auditory (alt)")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['modality']).isin(['none', 'auditory'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['probe', 'thr', 'max'],
        selected_value_type='responserate', alpha_marker=0.5, error_type='scatter',
        ax=axes_opto[0, 1], show_legend=False, group_colors=colors_aud_resp)
    ax.set_xticklabels([])
    ax.set_ylabel("")
    ax.set_xlabel("")

    # --- D-primes for opto conditions - Visual ---
    print("D-primes for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='saliency', conditions=['thr', 'max'],
        grouping_column='opto', groups=['none', 'unilateral', 'bilateral'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['Threshold', 'Maximum'])
    ax.set_ylabel("Performance (d')")
    filename = "vis_dprime_multi_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- MAIN FIGURE D-primes - Visual ---
    print("D-primes for opto conditions - Visual (main)")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['thr', 'max'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter',
        ax=axes_opto[1, 0], show_legend=False, group_colors=colors_vis_dprime)
    ax.set_xticklabels(['Control', 'Unilateral', 'Bilateral'])
    ax.set_ylabel("Performance (d')")
    ax.set_xlabel("")

    # --- D-primes - Auditory ---
    print("D-primes for opto conditions - Auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'auditory'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='saliency', conditions=['thr', 'max'],
        grouping_column='opto', groups=['none', 'unilateral', 'bilateral'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['Threshold', 'Maximum'])
    ax.set_ylabel("Performance (d')")
    filename = "aud_dprime_multi_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- MAIN FIGURE D-primes - Auditory ---
    print("D-primes for opto conditions - Auditory (main)")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'auditory'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['thr', 'max'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter',
        ax=axes_opto[1, 1], show_legend=False, group_colors=colors_aud_dprime)
    ax.set_xticklabels(['Control', 'Unilateral', 'Bilateral'])
    ax.set_ylabel("")
    ax.set_xlabel("")

    # --- D-primes for left/right visual conditions ---
    print("D-primes for left/right visual conditions")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['left', 'right'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['opto'] == 'none')
                           & (df_long['modality']).isin(['visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='direction', conditions=['left', 'right'],
        grouping_column='saliency', groups=['thr', 'max'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_ylabel("Performance (d')")
    filename = "vis_direction_saliency"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- D-primes for thr/max visual ---
    print("D-primes for thr/max visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])
                           & (df_long['opto_condition'] == False)]
    plt_vis_dprime_saliency = utils.plot_pairwise_effects_multiple_conditions(
        df_long_plot, condition_column='saliency', conditions=['thr', 'max'],
        animal_column='animal', selected_value_type='prime', alpha_marker=0.2)
    ax = plt_vis_dprime_saliency.gca()
    ax.set_xticklabels(['Threshold', 'Maximum'])
    ax.set_ylabel("Performance (d')")
    ax.set_ylim(-1, 4)  # FIXED: was plt.ylim on Figure
    filename = "vis_dprime_saliency"
    plt_vis_dprime_saliency.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_vis_dprime_saliency.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- D-primes for thr/max auditory ---
    print("D-primes for thr/max auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'auditory'])
                           & (df_long['opto_condition'] == False)]
    plt_aud_dprime_saliency = utils.plot_pairwise_effects_multiple_conditions(
        df_long_plot, condition_column='saliency', conditions=['thr', 'max'],
        animal_column='animal', selected_value_type='prime', alpha_marker=0.2)
    ax = plt_aud_dprime_saliency.gca()
    ax.set_xticklabels(['Threshold', 'Maximum'])
    ax.set_ylabel("Performance (d')")
    ax.set_ylim(-1, 4)  # FIXED
    filename = "aud_dprime_saliency"
    plt_aud_dprime_saliency.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    plt_aud_dprime_saliency.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- D-primes for opto conditions - Visual Left ---
    print("D-primes for opto conditions - Visual Left")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['left', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='saliency', conditions=['thr', 'max'],
        grouping_column='opto', groups=['none', 'unilateral', 'bilateral'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['Threshold', 'Maximum'])
    filename = "vis_dprime_multi_left_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Visual Left alt ---
    print("D-primes for opto conditions - Visual Left (alt)")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['left', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['thr', 'max'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['None', 'Unilateral', 'Bilateral'])
    filename = "vis_dprime_multi_left_opto_alt"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Visual Right ---
    print("D-primes for opto conditions - Visual Right")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['right', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='saliency', conditions=['thr', 'max'],
        grouping_column='opto', groups=['none', 'unilateral', 'bilateral'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['Threshold', 'Maximum'])
    filename = "vis_dprime_multi_right_opto"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Visual Right alt ---
    print("D-primes for opto conditions - Visual Right (alt)")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['right', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'visual'])]
    ax = utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['thr', 'max'],
        selected_value_type='prime', alpha_marker=0.5, error_type='scatter')
    ax.set_xticklabels(['None', 'Unilateral', 'Bilateral'])  # FIXED: was 'Maximum'
    filename = "vis_dprime_multi_right_opto_alt"
    fig = ax.get_figure()
    fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'), format='png')
    fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'), format='svg')

    # --- Save combined figure ---
    fig_opto.subplots_adjust(hspace=0.1, wspace=0.2)
    axes_opto[0, 0].set_ylim(-0.05, 1.05)
    axes_opto[0, 1].set_ylim(-0.05, 1.05)
    axes_opto[1, 0].set_ylim(-1.05, 3.65)
    axes_opto[1, 1].set_ylim(-1.05, 3.65)
    for ax in axes_opto.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(6)
        ax.title.set_size(8)
        ax.tick_params(axis='both', which='major', pad=2, length=3)
    fig_opto.savefig(os.path.join(figure_dir, 'png', "resp_dprimes_vis_aud_opto.png"), format='png')
    fig_opto.savefig(os.path.join(figure_dir, 'svg', "resp_dprimes_vis_aud_opto.svg"), format='svg')

    # --- Criterion figure ---
    fig_crit, axes_crit = plt.subplots(1, 2, figsize=(3.5, 2))
    print("Criterion for opto conditions - Visual")
    df_long_plot = df_long[(df_long['response_direction'] == 'left')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['visual'])
                           & (df_long['saliency'] == 'all')]
    utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['all'],
        selected_value_type='c', alpha_marker=0.5, error_type='scatter',
        ax=axes_crit[0], show_legend=False)
    axes_crit[0].set_xticklabels(['Control', 'Unilateral', 'Bilateral'])
    axes_crit[0].set_ylabel("Criterion (c)")
    axes_crit[0].set_xlabel("")

    print("Criterion for opto conditions - Auditory")
    df_long_plot = df_long[(df_long['response_direction'] == 'right')
                           & (df_long['direction']).isin(['all', 'none'])
                           & (df_long['is_2ADC'] == True)
                           & (df_long['modality']).isin(['none', 'auditory'])
                           & (df_long['saliency'] == 'all')]
    utils.plot_mean_lines_with_errorbars_ax(
        df_long_plot,
        condition_column='opto', conditions=['none', 'unilateral', 'bilateral'],
        grouping_column='saliency', groups=['all'],
        selected_value_type='c', alpha_marker=0.5, error_type='scatter',
        ax=axes_crit[1], show_legend=False)
    axes_crit[1].set_xticklabels(['Control', 'Unilateral', 'Bilateral'])
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

    # --- Delta plots ---
    for value_type in ['prime', 'responserate']:
        if value_type == 'prime':
            y_lim = [-3, 1.5]
        else:
            y_lim = [-1, 1]
        delta_plot_dict = {
            'unilateral': {'thr': {'left': [], 'right': []},
                           'max': {'left': [], 'right': []}},
            'bilateral': {'thr': {'left': [], 'right': []},
                          'max': {'left': [], 'right': []}},
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
                        opto_off = df_long_plot[
                            (df_long_plot['session'] == session)
                            & (df_long_plot['opto'] == 'none')
                            & (df_long_plot['saliency'] == saliency)
                            & (df_long_plot['direction'] == direction)
                            & (df_long_plot['value_type'] == value_type)]
                        opto_on = df_long_plot[
                            (df_long_plot['session'] == session)
                            & (df_long_plot['opto'] == opto)
                            & (df_long_plot['saliency'] == saliency)
                            & (df_long_plot['direction'] == direction)
                            & (df_long_plot['value_type'] == value_type)]

                        if len(opto_off) == 0 or len(opto_on) == 0:
                            continue

                        # Use mean to handle potential duplicates safely
                        opto_off_value = opto_off['value'].mean()
                        opto_on_value = opto_on['value'].mean()

                        delta = opto_on_value - opto_off_value
                        delta_plot_dict[opto][saliency][direction].append(delta)

            # One figure per saliency
            for saliency in ['thr', 'max']:
                fig, ax = plot_delta_left_right(delta_plot_dict, condition=opto,
                                                saliency=saliency,
                                                value=value_type, y_lim=y_lim)
                filename = f"vis_{value_type}_opto_delta_left_right_{opto}_{saliency}"
                fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'),
                            format='png', dpi=300)
                fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'),
                            format='svg')

            # --- Visual response rate: left vs right stimuli (control only) ---
            # Split threshold and max into separate figures
            print("Visual response rate - left vs right stimuli (control)")
            df_long_plot_ctrl = df_long[(df_long['response_direction'] == 'left')
                                        & (df_long['direction']).isin(['left', 'right'])
                                        & (df_long['opto'] == 'none')
                                        & (df_long['modality'] == 'visual')]

            saliency_colors_ctrl = {
                'thr': plot_utils.paper_colors("vis_thr"),
                'max': plot_utils.paper_colors("vis_max"),
            }
            saliency_labels_ctrl = {'thr': 'Threshold', 'max': 'Maximum'}

            for saliency in ['thr', 'max']:
                # Build figure with axes of exact physical size
                ax_w, ax_h = 0.75, 1
                left_m, right_m = 0.4, 0.1
                bottom_m, top_m = 0.3, 0.2
                fig_w = left_m + ax_w + right_m
                fig_h = bottom_m + ax_h + top_m
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_axes([left_m / fig_w, bottom_m / fig_h,
                                   ax_w / fig_w, ax_h / fig_h])

                color = saliency_colors_ctrl[saliency]
                alpha_lines = 0.4

                left_vals, right_vals = [], []
                for session in df_long_plot_ctrl['session'].unique():
                    left_data = df_long_plot_ctrl[
                        (df_long_plot_ctrl['session'] == session)
                        & (df_long_plot_ctrl['saliency'] == saliency)
                        & (df_long_plot_ctrl['direction'] == 'left')
                        & (df_long_plot_ctrl['value_type'] == 'responserate')]
                    right_data = df_long_plot_ctrl[
                        (df_long_plot_ctrl['session'] == session)
                        & (df_long_plot_ctrl['saliency'] == saliency)
                        & (df_long_plot_ctrl['direction'] == 'right')
                        & (df_long_plot_ctrl['value_type'] == 'responserate')]

                    if len(left_data) == 0 or len(right_data) == 0:
                        continue
                    left_vals.append(left_data['value'].mean())
                    right_vals.append(right_data['value'].mean())

                left_arr = np.array(left_vals)
                right_arr = np.array(right_vals)

                for l, r in zip(left_arr, right_arr):
                    ax.plot([0, 1], [l, r], color=color, alpha=alpha_lines,
                            linewidth=1, zorder=1)
                ax.plot([0, 1], [np.mean(left_arr), np.mean(right_arr)],
                        color=color, linewidth=2, marker='o', markersize=3,
                        label=saliency_labels_ctrl[saliency], zorder=3, alpha=1)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Left', 'Right'], fontsize=6)
                ax.set_ylabel("Response rate", fontsize=6)
                ax.set_xlabel('Visual stimulus', fontsize=6)
                ax.set_title(f'Control — {saliency_labels_ctrl[saliency]}', fontsize=7)
                ax.tick_params(axis='both', labelsize=6)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xlim([-0.15, 1.15])
                ax.set_ylim([-0.05, 1.05])

                filename = f"vis_resp_direction_saliency_control_{saliency}"
                fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'),
                            format='png', dpi=300)
                fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'),
                            format='svg')

            # --- Visual d-prime: left vs right stimuli (control only) ---
            # Split threshold and max into separate figures
            print("Visual d-prime - left vs right stimuli (control)")
            df_long_plot_ctrl = df_long[(df_long['response_direction'] == 'left')
                                        & (df_long['direction']).isin(['left', 'right'])
                                        & (df_long['opto'] == 'none')
                                        & (df_long['is_2ADC'] == True)
                                        & (df_long['modality'] == 'visual')]

            for saliency in ['thr', 'max']:
                # Build figure with axes of exact physical size
                ax_w, ax_h = 0.75, 1
                left_m, right_m = 0.4, 0.1
                bottom_m, top_m = 0.3, 0.2
                fig_w = left_m + ax_w + right_m
                fig_h = bottom_m + ax_h + top_m
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_axes([left_m / fig_w, bottom_m / fig_h,
                                   ax_w / fig_w, ax_h / fig_h])

                color = saliency_colors_ctrl[saliency]
                alpha_lines = 0.4

                left_vals, right_vals = [], []
                for session in df_long_plot_ctrl['session'].unique():
                    left_data = df_long_plot_ctrl[
                        (df_long_plot_ctrl['session'] == session)
                        & (df_long_plot_ctrl['saliency'] == saliency)
                        & (df_long_plot_ctrl['direction'] == 'left')
                        & (df_long_plot_ctrl['value_type'] == 'prime')]
                    right_data = df_long_plot_ctrl[
                        (df_long_plot_ctrl['session'] == session)
                        & (df_long_plot_ctrl['saliency'] == saliency)
                        & (df_long_plot_ctrl['direction'] == 'right')
                        & (df_long_plot_ctrl['value_type'] == 'prime')]

                    if len(left_data) == 0 or len(right_data) == 0:
                        continue
                    left_vals.append(left_data['value'].mean())
                    right_vals.append(right_data['value'].mean())

                left_arr = np.array(left_vals)
                right_arr = np.array(right_vals)

                for l, r in zip(left_arr, right_arr):
                    ax.plot([0, 1], [l, r], color=color, alpha=alpha_lines,
                            linewidth=1, zorder=1)
                ax.plot([0, 1], [np.mean(left_arr), np.mean(right_arr)],
                        color=color, linewidth=2, marker='o', markersize=3,
                        label=saliency_labels_ctrl[saliency], zorder=3, alpha=1)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Left', 'Right'], fontsize=6)
                ax.set_ylabel("Performance (d')", fontsize=6)
                ax.set_xlabel('Visual stimulus', fontsize=6)
                ax.set_title(f'Control — {saliency_labels_ctrl[saliency]}', fontsize=7)
                ax.tick_params(axis='both', labelsize=6)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xlim([-0.15, 1.15])
                ax.set_ylim([-0.6, 2.5])

                filename = f"vis_dprime_direction_saliency_control_{saliency}"
                fig.savefig(os.path.join(figure_dir, 'png', filename + '.png'),
                            format='png', dpi=300)
                fig.savefig(os.path.join(figure_dir, 'svg', filename + '.svg'),
                            format='svg')

