"""
Hybrid logistic-regression All-Star predictor.

Two linear models are fit on the same feature set:

  * a pointwise classifier P(All-Star | x),
  * a pairwise classifier P(x_i ranked above x_j | x_i - x_j),

and their probabilities are blended at inference. The pointwise model is
good at calibrated absolute probability; the pairwise model is good at
the relative ordering that structured top-K selection actually depends
on. Blending lets us keep both.

Selection itself is done structurally: within each (Season, Conference)
group we pick the top-2 backcourt + top-3 frontcourt as starters and the
top-7 remaining as reserves, matching the real All-Star roster shape.

Run:
    python scripts/log_reg.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


class Config:
    SEED = 47

    # Regularization sweep for both heads. log-spaced so we get a reasonable
    # range from heavy shrinkage (C=1e-3) to nearly unregularized (C=1e2).
    C_VALUES = np.logspace(-3, 2, 12)

    # Alpha sweep for the pointwise / pairwise blend. We tune this on the
    # validation set; in practice the optimum lives in the middle, not the
    # endpoints, which is the whole reason for keeping both heads.
    ALPHA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Temperature sweep for sharpening / softening the blended logits.
    TEMP_VALUES = [0.3, 0.5, 0.7, 1.0]

    TRAIN_END = 2015
    VAL_END = 2021

    PENALTY = 'l2'
    SOLVER = 'lbfgs'
    MAX_ITER = 4000
    TOL = 1e-6

    # Cap pairs per (Season, Conference) group to keep the pairwise problem
    # tractable. The training distribution is hugely imbalanced (~12 pos per
    # group vs hundreds of neg), so the full cross-product is wasteful.
    MAX_PAIRS = 400

    # Class weighting is intentionally None for the pointwise head: under
    # structured top-K selection we never threshold on probability, only
    # *rank*, so balancing distorts the well-calibrated absolute scores
    # without helping selection. The pairwise head sees a balanced pair
    # set by construction either way.
    CLASS_WEIGHT = None


# =========================================================
# PAIRWISE DATA BUILDER
# =========================================================
def build_pairwise_dataset(X, y, df, max_pairs_per_group, rng):
    """Build (x_i - x_j, 1) and (x_j - x_i, 0) pairs within each group.

    Pairs are restricted to within a (Season, Conference) group because the
    real selection process is competitive within that group — ranking a
    1985 Eastern player against a 2019 Western player is not a meaningful
    comparison.
    """
    df = df.reset_index(drop=True)
    y_np = y.values

    X_pairs = []
    y_pairs = []

    for _, idx in df.groupby(['Season Ending Year', 'Conference_East']).groups.items():
        idx = list(idx)
        pos_idx = [i for i in idx if y_np[i] == 1]
        neg_idx = [i for i in idx if y_np[i] == 0]

        if not pos_idx or not neg_idx:
            continue

        pairs = [(i, j) for i in pos_idx for j in neg_idx]

        if len(pairs) > max_pairs_per_group:
            chosen = rng.choice(len(pairs), max_pairs_per_group, replace=False)
            pairs = [pairs[k] for k in chosen]

        for i, j in pairs:
            X_pairs.append(X[i] - X[j])
            y_pairs.append(1)
            X_pairs.append(X[j] - X[i])
            y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs)


# =========================================================
# DATA PIPELINE
# =========================================================
class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def load(self, path):
        df = pd.read_csv(path)
        return self._features(df)

    def _features(self, df):
        """Add features that the cleaned dataset doesn't already carry.

        Kept deliberately compact — linear models are sensitive to
        collinearity, so we only add features that capture a different
        axis of signal than what's already in the data:

          * Volume composites (Usage) and their interactions with team
            success (Usage_x_Win, Impact_on_winning).
          * Conference-relative scoring (z-score, share-of-leader,
            absolute deviation from group mean).
          * A single conference rank on Usage (role-on-team proxy).
          * Career-form lags (1- and 2-season rolling All-Star count).

        BoxLoad is already season-z'd in the cleaning pipeline, so the
        modelling script doesn't double-normalize it.
        """
        group_keys = ['Season Ending Year', 'Conference_East']

        df['Usage'] = df['FGA per game'] + 0.44 * df['FTA per game']
        df['Usage_x_Win'] = df['Usage'] * df['Team Win %']
        df['Impact_on_winning'] = df['PTS per game'] * df['Team Win %']

        pts_mean = df.groupby(group_keys)['PTS per game'].transform('mean')
        pts_std = df.groupby(group_keys)['PTS per game'].transform('std')
        df['PTS_conf_z'] = (df['PTS per game'] - pts_mean) / (pts_std + 1e-8)
        df['PTS_minus_conf_avg'] = df['PTS per game'] - pts_mean

        pts_max = df.groupby(group_keys)['PTS per game'].transform('max')
        df['PTS_conf_share'] = df['PTS per game'] / (pts_max + 1e-8)

        impact_mean = df.groupby(group_keys)['Impact_on_winning'].transform('mean')
        impact_std = df.groupby(group_keys)['Impact_on_winning'].transform('std')
        df['Impact_conf_z'] = (
            (df['Impact_on_winning'] - impact_mean) / (impact_std + 1e-8)
        )

        df['Usage_rank_conf'] = df.groupby(group_keys)['Usage'].rank(pct=True)

        # Team-success interactions on rebounding / playmaking.
        df['Win_x_AST'] = df['Team Win %'] * df['AST per game']
        df['Win_x_TRB'] = df['Team Win %'] * df['TRB per game']

        df = df.sort_values(['Player', 'Season Ending Year']).reset_index(drop=True)
        df['AllStar_Last_Year'] = (
            df.groupby('Player')['All Star'].shift(1).fillna(0)
        )
        df['AllStar_last_2'] = (
            df.groupby('Player')['All Star']
            .shift(1).rolling(2).sum().fillna(0)
        )
        df['Impact_x_LastYear'] = df['Impact_on_winning'] * df['AllStar_Last_Year']

        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def split(self, df):
        train_df = df[df['Season Ending Year'] <= self.cfg.TRAIN_END].copy()
        val_df = df[
            (df['Season Ending Year'] > self.cfg.TRAIN_END) &
            (df['Season Ending Year'] <= self.cfg.VAL_END)
        ].copy()
        test_df = df[df['Season Ending Year'] > self.cfg.VAL_END].copy()
        return train_df, val_df, test_df

    def _group_center(self, X, df):
        """Center each feature within its (Season, Conference) group.

        This makes the linear model's coefficients describe *relative*
        deviation from the group mean, which is the right quantity for a
        ranking problem. Globally a 25 PPG line might be elite; in a
        loaded conference it might be third-best.
        """
        X_df = pd.DataFrame(X, index=df.index)
        group_means = X_df.groupby(
            [df['Season Ending Year'], df['Conference_East']]
        ).transform('mean')
        centered = (X_df - group_means).fillna(0.0)
        return centered.values

    def prepare(self, train_df, val_df, test_df):
        # 'Prev All Stars' carries the same signal as our shifted features
        # but in a much higher-variance form (career totals). Dropping
        # avoids letting a single feature dominate the linear model.
        # Position-group one-hots are also dropped from features because
        # they're used downstream for structured selection, not scoring.
        #
        # TS%, BoxLoad, GamesFrac, PrimeAge are deliberately excluded:
        # they're informative for the neural network but they're nearly
        # linear combinations of features already present (PTS+TRB+AST+...,
        # PTS/(2*(FGA+.44*FTA)), Games/#TeamGames, age bucket), which
        # destabilises the linear model's coefficients without adding
        # signal. The NN handles them just fine in its own pipeline.
        drop_cols = [
            'All Star', 'Player', 'Season Ending Year',
            'Prev All Stars', 'Conference_East',
            'PosGroup_Backcourt', 'PosGroup_Frontcourt',
            'TS%', 'BoxLoad', 'GamesFrac', 'PrimeAge',
        ]

        def split_xy(df):
            X = df.drop(columns=drop_cols, errors='ignore')
            y = df['All Star']
            return X, y

        X_train_df, y_train = split_xy(train_df)
        X_val_df, y_val = split_xy(val_df)
        X_test_df, y_test = split_xy(test_df)

        self.feature_names = X_train_df.columns.tolist()

        X_train = self.imputer.fit_transform(X_train_df)
        X_val = self.imputer.transform(X_val_df)
        X_test = self.imputer.transform(X_test_df)

        X_train = self._group_center(X_train, train_df)
        X_val = self._group_center(X_val, val_df)
        X_test = self._group_center(X_test, test_df)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return (
            X_train, y_train, train_df,
            X_val, y_val, val_df,
            X_test, y_test, test_df,
        )


# =========================================================
# HYBRID MODEL
# =========================================================
class HybridModel:
    def __init__(self, cfg, C, alpha, temp, rng):
        self.cfg = cfg
        self.alpha = alpha
        self.temp = temp
        self.rng = rng

        common = dict(
            C=C,
            penalty=cfg.PENALTY,
            solver=cfg.SOLVER,
            max_iter=cfg.MAX_ITER,
            tol=cfg.TOL,
            n_jobs=-1,
        )
        self.pointwise = LogisticRegression(class_weight=cfg.CLASS_WEIGHT, **common)
        # The pairwise classifier sees a balanced positive/negative set
        # by construction, so class_weight isn't useful here.
        self.pairwise = LogisticRegression(**common)

    def fit(self, X, y, df):
        self.pointwise.fit(X, y)
        X_pair, y_pair = build_pairwise_dataset(
            X, y, df, self.cfg.MAX_PAIRS, self.rng
        )
        self.pairwise.fit(X_pair, y_pair)

    def predict_proba(self, X, df):
        p_point = self.pointwise.predict_proba(X)[:, 1]

        # The pairwise classifier's linear score, evaluated on a single
        # player (not a pair), is the per-player ranking logit. Sigmoid
        # turns it back into a probability-like value in [0, 1].
        logits_pair = (X @ self.pairwise.coef_.T + self.pairwise.intercept_).squeeze()
        p_pair = 1.0 / (1.0 + np.exp(-logits_pair))

        probs = self.alpha * p_point + (1 - self.alpha) * p_pair

        # Temperature scaling on the blend. T < 1 sharpens; T > 1 softens.
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        logits = np.log(probs) - np.log(1 - probs)
        logits = logits / self.temp
        return 1.0 / (1.0 + np.exp(-logits))


# =========================================================
# STRUCTURED SELECTION
# =========================================================
def structured_selection(df):
    """Pick the canonical 12-player roster within each (Season, Conf).

    Selection mirrors the real ballot: top-2 backcourt + top-3 frontcourt
    starters, then top-7 reserves from the remaining pool. Doing this
    explicitly (vs thresholding on prob) is what closes the gap between
    AUC and recall — the model can hand back well-ordered probabilities
    and still miss roster slots if you select by threshold.
    """
    df = df.copy()
    df['pred'] = 0

    for _, group in df.groupby(['Season Ending Year', 'Conference_East']):
        bc = group[group['PosGroup_Backcourt'] == 1]
        fc = group[group['PosGroup_Frontcourt'] == 1]

        starters = pd.concat([
            bc.sort_values('prob', ascending=False).head(2),
            fc.sort_values('prob', ascending=False).head(3),
        ])

        remaining = group.drop(index=starters.index)
        reserves = remaining.sort_values('prob', ascending=False).head(7)

        selected = pd.concat([starters, reserves])
        df.loc[selected.index, 'pred'] = 1

    return df


# =========================================================
# TRAINER (rolling-window cross-validation)
# =========================================================
class Trainer:
    """Hyperparameter selection via expanding-window CV.

    Single-fold validation against the 2016–2021 block is noisy: only
    ~158 All-Stars across 12 (season, conference) groups, so changing one
    selection per group moves TopK recall by a full point. Different
    hyperparameter settings tie on TopK constantly, and the tie-breaker
    (AUC) doesn't reliably correlate with held-out F1 on the 2022–2025
    test block.

    To get a more stable signal we walk forward through the validation
    period in three folds:

        fold 1: train on ≤2013, score 2014–2015
        fold 2: train on ≤2015, score 2016–2018
        fold 3: train on ≤2018, score 2019–2021

    A hyperparameter set has to do well *across* folds to win, which kills
    most of the noise-driven ties.
    """

    CV_FOLDS = [
        (2013, 2014, 2015),
        (2015, 2016, 2018),
        (2018, 2019, 2021),
    ]

    def __init__(self, cfg):
        self.cfg = cfg

    def _cv_score(self, X_all, y_all, df_all, C, alpha, temp, rng):
        """Mean (TopK + 0.5 * AUC) across the rolling-window folds."""
        scores = []
        for train_end, val_start, val_end in self.CV_FOLDS:
            tr_mask = (df_all['Season Ending Year'] <= train_end).values
            va_mask = (
                (df_all['Season Ending Year'] >= val_start)
                & (df_all['Season Ending Year'] <= val_end)
            ).values

            X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
            X_va, y_va = X_all[va_mask], y_all[va_mask]
            df_tr = df_all[tr_mask].reset_index(drop=True)
            df_va = df_all[va_mask].reset_index(drop=True)

            model = HybridModel(self.cfg, C, alpha=alpha, temp=temp, rng=rng)
            model.fit(X_tr, y_tr, df_tr)
            probs = model.predict_proba(X_va, df_va)

            auc = roc_auc_score(y_va, probs)
            temp_df = df_va.copy()
            temp_df['prob'] = probs
            temp_df = structured_selection(temp_df)
            topk = ((temp_df['pred'] == 1) & (y_va.values == 1)).sum() / max(y_va.sum(), 1)
            scores.append(topk + 0.5 * auc)
        return float(np.mean(scores))

    def tune(self, X_train, y_train, train_df, X_val, y_val, val_df):
        """Sweep (C, alpha, temp) by mean rolling-window CV score.

        We stack train + val and let the CV folds carve their own windows
        out of the combined block; the held-out 2022–2025 test set is
        never touched here.
        """
        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        df_all = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

        rng = np.random.default_rng(self.cfg.SEED)

        print("=== Sweeping C (alpha=1.0, temp=0.5) ===")
        best_C, best_score = None, -np.inf
        for C in self.cfg.C_VALUES:
            score = self._cv_score(X_all, y_all, df_all, C, alpha=1.0, temp=0.5, rng=rng)
            print(f"  C={C:8.4f} | CV score={score:.4f}")
            if score > best_score:
                best_score, best_C = score, C

        print(f"\nBest C (CV): {best_C}")

        print("\n=== Sweeping (alpha, temp) at best C ===")
        best_alpha, best_temp = 1.0, 0.5
        for alpha in self.cfg.ALPHA_VALUES:
            for temp in self.cfg.TEMP_VALUES:
                score = self._cv_score(X_all, y_all, df_all, best_C, alpha=alpha, temp=temp, rng=rng)
                if score > best_score:
                    best_score = score
                    best_alpha, best_temp = alpha, temp
                print(f"  a={alpha:.2f} | T={temp:.2f} | CV score={score:.4f}")

        print(f"\nBest: C={best_C}, alpha={best_alpha}, temp={best_temp}")
        return best_C, best_alpha, best_temp

    def train_final(self, C, alpha, temp, X_train, y_train, train_df,
                    X_val=None, y_val=None, val_df=None):
        """Refit on train + val once hyperparameters are locked.

        Extra recent data — especially the 2016–2021 window — moves the
        model's coefficients toward the era it will actually be scored
        on, without leaking test-set rows.
        """
        rng = np.random.default_rng(self.cfg.SEED)
        if X_val is not None:
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
            df_full = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        else:
            X_full, y_full, df_full = X_train, y_train, train_df

        model = HybridModel(self.cfg, C, alpha=alpha, temp=temp, rng=rng)
        model.fit(X_full, y_full, df_full)
        return model


# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, X_test, y_test, test_df):
    probs = model.predict_proba(X_test, test_df)

    temp_df = test_df.copy()
    temp_df['prob'] = probs
    temp_df = structured_selection(temp_df)

    y_true = y_test.values
    y_pred = temp_df['pred'].values

    print("\n=== FINAL RESULTS ===")
    print(f"AUC      : {roc_auc_score(y_true, probs):.4f}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")

    print("\n=== Per-Season (Conference Split) ===")
    for (season, conf), group in temp_df.groupby(['Season Ending Year', 'Conference_East']):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        prob_g = group['prob'].values
        auc = roc_auc_score(y_g, prob_g) if len(np.unique(y_g)) > 1 else np.nan
        prec = precision_score(y_g, p_g, zero_division=0)
        rec = recall_score(y_g, p_g, zero_division=0)
        f1 = f1_score(y_g, p_g, zero_division=0)
        topk = ((p_g == 1) & (y_g == 1)).sum() / max(y_g.sum(), 1)
        top12 = ((p_g == 1) & (y_g == 1)).sum() / max(p_g.sum(), 1)
        print(
            f"{season} Conf {conf} | AUC {auc:.4f} | "
            f"P {prec:.3f} | R {rec:.3f} | F1 {f1:.3f} | "
            f"TopK {topk:.3f} | Top12 {top12:.3f}"
        )

    print("\n=== Per-Season (Overall) ===")
    season_metrics = []
    for season, group in temp_df.groupby('Season Ending Year'):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        prob_g = group['prob'].values
        auc = roc_auc_score(y_g, prob_g) if len(np.unique(y_g)) > 1 else np.nan
        acc = accuracy_score(y_g, p_g)
        prec = precision_score(y_g, p_g, zero_division=0)
        rec = recall_score(y_g, p_g, zero_division=0)
        f1 = f1_score(y_g, p_g, zero_division=0)
        topk = ((p_g == 1) & (y_g == 1)).sum() / max(y_g.sum(), 1)
        top12 = ((p_g == 1) & (y_g == 1)).sum() / max(p_g.sum(), 1)
        season_metrics.append((auc, acc, prec, rec, f1, topk, top12))
        print(
            f"{season} | AUC {auc:.4f} | "
            f"P {prec:.3f} | R {rec:.3f} | F1 {f1:.3f} | "
            f"TopK {topk:.3f} | Top12 {top12:.3f}"
        )

    season_metrics = np.array(season_metrics)
    print("\n=== Avg Per-Season Metrics ===")
    print(f"AUC          : {np.nanmean(season_metrics[:, 0]):.4f}")
    print(f"Accuracy     : {season_metrics[:, 1].mean():.4f}")
    print(f"Precision    : {season_metrics[:, 2].mean():.4f}")
    print(f"Recall       : {season_metrics[:, 3].mean():.4f}")
    print(f"F1           : {season_metrics[:, 4].mean():.4f}")
    print(f"TopK Recall  : {season_metrics[:, 5].mean():.4f}")
    print(f"Top12 Acc    : {season_metrics[:, 6].mean():.4f}")


def print_feature_importance(model, feature_names, top_k=10):
    print("\n=== FEATURE IMPORTANCE (POINTWISE) ===")
    coefs = model.pointwise.coef_.flatten()
    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
    })
    print("\nTop Positive:")
    print(df.sort_values("coef", ascending=False).head(top_k).to_string(index=False))
    print("\nTop Negative:")
    print(df.sort_values("coef", ascending=True).head(top_k).to_string(index=False))

    print("\n=== FEATURE IMPORTANCE (PAIRWISE) ===")
    coefs_pair = model.pairwise.coef_.flatten()
    df_pair = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs_pair,
    })
    print("\nTop Positive (helps beat others):")
    print(df_pair.sort_values("coef", ascending=False).head(top_k).to_string(index=False))
    print("\nTop Negative:")
    print(df_pair.sort_values("coef", ascending=True).head(top_k).to_string(index=False))


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = Config()
    np.random.seed(cfg.SEED)

    pipeline = DataPipeline(cfg)
    df = pipeline.load('source/cleaned/cleaned_data.csv')

    train_df, val_df, test_df = pipeline.split(df)

    (X_train, y_train, train_df,
     X_val, y_val, val_df,
     X_test, y_test, test_df) = pipeline.prepare(train_df, val_df, test_df)

    trainer = Trainer(cfg)
    best_C, best_alpha, best_temp = trainer.tune(
        X_train, y_train, train_df, X_val, y_val, val_df
    )

    model = trainer.train_final(
        best_C, best_alpha, best_temp,
        X_train, y_train, train_df,
        X_val, y_val, val_df,
    )

    print_feature_importance(model, pipeline.feature_names)
    evaluate(model, X_test, y_test, test_df)


if __name__ == "__main__":
    main()
