"""
Pipeline factories and trainer for the SVM All-Star predictor.

Two halves:

  * Pipeline factories (`get_*_pipeline`) — small wrappers around sklearn
    Pipelines so the notebook can sweep over (architecture × C) cleanly.
    Each pipeline does its own feature transform → scaler → SVC, so the
    architecture choice itself becomes a hyperparameter.

  * `SVMTrainer` — rolling-window cross-validation + final refit on
    train+val. Mirrors the strategy used in log_reg.py: single-fold val
    tuning is noisy at ~158 positives, so we walk three expanding
    windows through the train+val block to pick `(architecture, C)`,
    then refit once on the full block.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC


# =========================================================
# PIPELINE FACTORIES
# =========================================================
def get_base_pipeline(cfg, c_val=1.0, weights=None):
    """Linear SVM on the already-cleaned features.

    Class weighting defaults to 1:3 (positives upweighted ~3x) because
    All-Stars are ~7-8% of the population. Without this, the linear
    margin tilts toward the trivial "predict 0" boundary and the
    structured selection sees badly-ranked marginal candidates.
    """
    if weights is None:
        weights = {0: 1, 1: 3}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=c_val,
            kernel='linear',
            probability=cfg.PROBABILITY,
            max_iter=cfg.MAX_ITER,
            tol=cfg.TOL,
            random_state=cfg.SEED,
            class_weight=weights,
        ))
    ])


def get_poly_pipeline(cfg, c_val=0.1, weights=None):
    """Linear SVM on degree-2 polynomial features.

    Useful for capturing interaction effects (PTS * Win%, AST * TRB,
    etc.) without the kernel trick. We re-scale *after* generating the
    polynomial features so the SVM doesn't choke on the squared
    magnitudes.
    """
    if weights is None:
        weights = {0: 1, 1: 3}
    return Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=c_val,
            kernel='linear',
            probability=cfg.PROBABILITY,
            max_iter=cfg.MAX_ITER,
            tol=cfg.TOL,
            random_state=cfg.SEED,
            class_weight=weights,
        ))
    ])


def get_rbf_pipeline(cfg, c_val=1.0, gamma='scale', weights=None):
    """RBF kernel SVM.

    Lets the model carve curved decision boundaries that linear /
    polynomial layouts can't fit. Practical trade-off: slower to train
    and to score, and the coefficients aren't directly interpretable.
    """
    if weights is None:
        weights = {0: 1, 1: 3}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=c_val,
            kernel='rbf',
            gamma=gamma,
            probability=cfg.PROBABILITY,
            max_iter=cfg.MAX_ITER,
            tol=cfg.TOL,
            random_state=cfg.SEED,
            class_weight=weights,
        ))
    ])


def get_kmeans_pipeline(n_clusters=10, c_val=1.0, weights=None):
    """K-Means archetype distances → LinearSVC.

    Replaces raw player stats with their distances to N league
    archetypes (centroids). The intuition is that "scoring guard who
    rebounds" is a more useful representation than 20 correlated raw
    columns, but in practice this discards too much signal to be
    competitive with the baseline.
    """
    if weights is None:
        weights = {0: 1, 1: 3}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, n_init=10)),
        ('svm', LinearSVC(C=c_val, class_weight=weights, dual=False, max_iter=20000))
    ])


def get_pca_pipeline(n_components=5, c_val=1.0, weights=None):
    """PCA → LinearSVC.

    Compresses the feature space before fitting. Helpful for diagnostic
    purposes (how much signal is in the top components) but the linear
    SVM already does its own implicit regularisation, so PCA usually
    just throws away information.
    """
    if weights is None:
        weights = {0: 1, 1: 3}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('svm', LinearSVC(C=c_val, class_weight=weights, dual=False, max_iter=20000))
    ])


# =========================================================
# STRUCTURED SELECTION
# =========================================================
def structured_selection(df, score_col='prob'):
    """Top-2 BC + top-3 FC starters + top-7 reserves per (Season, Conf).

    The model itself is unconstrained; structure is enforced post hoc.
    This is what closes the gap between AUC and recall — without it,
    threshold-based selection can put 14 players into the East and 10
    into the West, which is never how the real ballot ends.
    """
    df = df.copy()
    df['pred'] = 0
    for _, group in df.groupby(['Season Ending Year', 'Conference_East']):
        bc = group[group['PosGroup_Backcourt'] == 1]
        fc = group[group['PosGroup_Frontcourt'] == 1]
        starters = pd.concat([
            bc.sort_values(score_col, ascending=False).head(2),
            fc.sort_values(score_col, ascending=False).head(3),
        ])
        remaining = group.drop(index=starters.index)
        reserves = remaining.sort_values(score_col, ascending=False).head(7)
        selected = pd.concat([starters, reserves])
        df.loc[selected.index, 'pred'] = 1
    return df


# =========================================================
# TRAINER (rolling-window CV)
# =========================================================
class SVMTrainer:
    """Rolling-window CV trainer for SVM pipelines.

    Single-fold validation against 2016–2021 is noisy: only ~158
    All-Stars across 12 (season, conference) groups, so changing one
    selection per group moves TopK recall by a full point. Different
    hyperparameters tie on TopK constantly, and the AUC tiebreaker
    doesn't transfer to test reliably.

    Three expanding-window folds walk through the train+val block:

        fold 1: train ≤ 2013, score 2014–2015
        fold 2: train ≤ 2015, score 2016–2018
        fold 3: train ≤ 2018, score 2019–2021

    A pipeline/C pair has to do well *across* folds to win, which kills
    most of the noise-driven ties. After tuning, we refit on the full
    train + val block so the final model has seen the recent seasons.
    """

    CV_FOLDS = [
        (2013, 2014, 2015),
        (2015, 2016, 2018),
        (2018, 2019, 2021),
    ]

    def __init__(self, cfg):
        self.cfg = cfg

    def _score_one(self, pipeline_func, c_val, X_tr, y_tr, X_va, y_va, df_va):
        """Fit on (X_tr, y_tr), score on val, return (TopK, AUC)."""
        pipeline = pipeline_func(self.cfg, c_val=c_val)
        pipeline.fit(X_tr, y_tr)

        # SVC with probability=False exposes decision_function; with
        # probability=True (or LinearSVC wrappers) exposes predict_proba.
        if hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_va)
        else:
            scores = pipeline.predict_proba(X_va)[:, 1]

        auc = roc_auc_score(y_va, scores)
        tmp = df_va.copy()
        tmp['prob'] = scores
        tmp = structured_selection(tmp)
        topk = ((tmp['pred'] == 1) & (y_va.values == 1)).sum() / max(y_va.sum(), 1)
        return topk, auc

    def _cv_score(self, pipeline_func, c_val, X_all, y_all, df_all):
        """Mean (TopK + 0.5 * AUC) across the rolling-window folds.

        TopK is the primary signal, AUC is a small (continuous) tiebreaker
        that nudges the selection toward configurations with stronger
        overall ranking.
        """
        scores = []
        for train_end, val_start, val_end in self.CV_FOLDS:
            tr_mask = (df_all['Season Ending Year'] <= train_end).values
            va_mask = (
                (df_all['Season Ending Year'] >= val_start)
                & (df_all['Season Ending Year'] <= val_end)
            ).values
            X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
            X_va, y_va = X_all[va_mask], y_all[va_mask]
            df_va = df_all[va_mask].reset_index(drop=True)
            topk, auc = self._score_one(pipeline_func, c_val,
                                        X_tr, y_tr, X_va, y_va, df_va)
            scores.append(topk + 0.5 * auc)
        return float(np.mean(scores))

    def tune_architecture(self, architectures, X_train, y_train, train_df,
                          X_val, y_val, val_df, verbose=True):
        """Joint sweep over (architecture, C).

        `architectures` is a dict of name → pipeline-factory. For each
        (name, C) we evaluate the rolling-window CV score, and return
        the winning combination plus a full results log for plotting.
        """
        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        df_all = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

        results_log = []
        best_name, best_C, best_score = None, None, -np.inf

        for name, pipe_func in architectures.items():
            if verbose:
                print(f"\n>>> CV-tuning {name} ...")
            for C in self.cfg.C_VALUES:
                score = self._cv_score(pipe_func, C, X_all, y_all, df_all)
                results_log.append({
                    "name": name,
                    "C": C,
                    "cv_score": score,
                })
                if verbose:
                    print(f"   C={C:9.5f} | CV score={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_name, best_C = name, C

        if verbose:
            print(f"\nBest: {best_name} @ C={best_C} (CV score={best_score:.4f})")
        return best_name, best_C, results_log

    def train_final(self, pipeline_func, C, X_train, y_train,
                    X_val=None, y_val=None):
        """Refit on train + val once (architecture, C) is locked.

        Extra recent data — especially 2016–2021 — moves the SVM's
        support vectors toward the era it will actually be scored on.
        Train-only loses six years of recent signal; we never touch the
        2022–2025 test rows.
        """
        if X_val is not None and y_val is not None:
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        else:
            X_full, y_full = X_train, y_train

        final_pipeline = pipeline_func(self.cfg, c_val=C)
        final_pipeline.fit(X_full, y_full)
        return final_pipeline

    # -----------------------------------------------------
    # Legacy single-fold API (kept for backwards-compat with
    # earlier notebook cells).
    # -----------------------------------------------------
    def tune(self, pipeline_func, X_train, y_train, train_df,
             X_val, y_val, val_df, verbose=True):
        """Single-fold tune (legacy). Use `tune_architecture` instead."""
        results = []
        for C in self.cfg.C_VALUES:
            topk, auc = self._score_one(pipeline_func, C,
                                        X_train, y_train, X_val, y_val, val_df)
            pipeline = pipeline_func(self.cfg, c_val=C)
            pipeline.fit(X_train, y_train)
            results.append((C, auc, topk, pipeline))
            if verbose:
                print(f"C={C:.5f} | AUC={auc:.4f} | TopK={topk:.4f}")

        best_entry = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)[0]
        if verbose:
            print(f"\nBest C: {best_entry[0]} (val TopK={best_entry[2]:.4f})")
        return best_entry[3], best_entry[0], results
