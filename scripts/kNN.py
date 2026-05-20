"""
Per-conference k-NN All-Star predictor.

The intuition: All-Star selection is competitive within a conference,
so a k-NN trained only on East data ranks Eastern candidates against
their actual peers, and likewise for the West. Pooling the two
conferences before fitting smears out the very signal we care about
(a 22 PPG line means different things in a top-heavy East than in a
loaded West).

Pipeline:

  * Same feature set as log_reg / svm (group-centred, season z-scored,
    career-form lags, conference-relative scoring composites).
  * Per-conference k-NN with a tunable distance-weighting power and
    temperature.
  * Rolling-window CV over (K, distance power, temperature) on
    train+val (three folds across 2014–2021).
  * Refit on train+val once the hyperparameters are locked.
  * Structured top-2 BC + top-3 FC + top-7 reserves selection on test.

Run:
    python scripts/kNN.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class Config:
    SEED = 47

    # k, distance-weighting power, and softening temperature each get
    # their own sweep. They interact: a small k with a high power is
    # roughly equivalent to a larger k with a lower power, so we tune
    # them jointly rather than one at a time.
    K_VALUES = [5, 7, 9, 11, 13, 15, 19]
    DIST_POWERS = [1.0, 1.5, 2.0, 3.0]
    TEMPS = [0.5]

    TRAIN_END = 2015
    VAL_END = 2021

    METRIC = 'euclidean'
    EPS = 1e-6


# =========================================================
# FEATURE PIPELINE
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
        """Same feature set as the linear models, kept aligned so
        cross-model ensembling stays apples-to-apples.

        Volume composites, conference-relative scoring stats, and
        career-form lags. We deliberately don't compute team-success
        interactions inside the kNN — distance metrics already weight
        the raw stats appropriately when group-centred.
        """
        group_keys = ['Season Ending Year', 'Conference_East']

        df['Usage'] = df['FGA per game'] + 0.44 * df['FTA per game']
        df['Usage_x_Win'] = df['Usage'] * df['Team Win %']
        df['Impact_on_winning'] = df['PTS per game'] * df['Team Win %']

        pts_mean = df.groupby(group_keys)['PTS per game'].transform('mean')
        pts_std = df.groupby(group_keys)['PTS per game'].transform('std')
        df['PTS_conf_z'] = (df['PTS per game'] - pts_mean) / (pts_std + 1e-8)

        pts_max = df.groupby(group_keys)['PTS per game'].transform('max')
        df['PTS_conf_share'] = df['PTS per game'] / (pts_max + 1e-8)

        impact_mean = df.groupby(group_keys)['Impact_on_winning'].transform('mean')
        impact_std = df.groupby(group_keys)['Impact_on_winning'].transform('std')
        df['Impact_conf_z'] = (
            (df['Impact_on_winning'] - impact_mean) / (impact_std + 1e-8)
        )

        df['Usage_rank_conf'] = df.groupby(group_keys)['Usage'].rank(pct=True)
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

        return df.replace([np.inf, -np.inf], np.nan)

    def split(self, df):
        train = df[df['Season Ending Year'] <= self.cfg.TRAIN_END].copy()
        val = df[
            (df['Season Ending Year'] > self.cfg.TRAIN_END)
            & (df['Season Ending Year'] <= self.cfg.VAL_END)
        ].copy()
        test = df[df['Season Ending Year'] > self.cfg.VAL_END].copy()
        return train, val, test

    def prepare(self, train_df, val_df, test_df):
        # Same set of drops as log_reg / svm. The cleaning-added
        # features (TS%, BoxLoad, GamesFrac, PrimeAge) are near-linear
        # combinations of features already present and add noise to
        # the kNN distance metric without contributing signal.
        drop_cols = [
            'All Star', 'Player', 'Season Ending Year',
            'Prev All Stars', 'Conference_East',
            'PosGroup_Backcourt', 'PosGroup_Frontcourt',
            'TS%', 'BoxLoad', 'GamesFrac', 'PrimeAge',
        ]

        def split_xy(df):
            X = df.drop(columns=drop_cols, errors='ignore')
            return X, df['All Star']

        X_train_df, y_train = split_xy(train_df)
        X_val_df, y_val = split_xy(val_df)
        X_test_df, y_test = split_xy(test_df)
        self.feature_names = X_train_df.columns.tolist()

        X_train = self.imputer.fit_transform(X_train_df)
        X_val = self.imputer.transform(X_val_df)
        X_test = self.imputer.transform(X_test_df)

        # Group-centering matters for k-NN: distances in raw feature
        # space mix together era-level differences (60s pace vs 2010s
        # pace) with within-conference comparisons. Centring per
        # (Season, Conference) leaves only the within-pool deltas,
        # which is what the structured selection actually depends on.
        X_train = self._group_center(X_train, train_df)
        X_val = self._group_center(X_val, val_df)
        X_test = self._group_center(X_test, test_df)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return (X_train, y_train, train_df,
                X_val, y_val, val_df,
                X_test, y_test, test_df)

    def _group_center(self, X, df):
        X_df = pd.DataFrame(X, index=df.index)
        means = X_df.groupby(
            [df['Season Ending Year'], df['Conference_East']]
        ).transform('mean')
        return (X_df - means).fillna(0.0).values


# =========================================================
# STRUCTURED SELECTION
# =========================================================
def structured_selection(df, score_col='prob'):
    """Top-2 BC + top-3 FC + top-7 reserves per (Season, Conf)."""
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
# kNN MODEL (per-conference)
# =========================================================
class ConferenceKNN:
    """Two k-NN models, one for East and one for West.

    Per-conference fitting matters here because the All-Star ballot
    runs separate East/West votes; the natural distance neighbours of
    a Western forward are other Western players, not the entire
    league. A pooled k-NN would dilute that.

    Distance weighting uses a softened inverse-power kernel:

        w(d) = 1 / (d/T + eps) ** p

    Larger `p` makes the nearest neighbour dominate; larger `T` (the
    temperature) smooths the contribution of distant neighbours.
    """

    def __init__(self, K, power, temp, metric='euclidean', eps=1e-6):
        self.K = K
        self.power = power
        self.temp = temp
        self.metric = metric
        self.eps = eps
        self.east_model = None
        self.west_model = None

    def _weights(self, distances):
        d = distances / self.temp
        return 1.0 / (np.power(d + self.eps, self.power))

    def fit(self, X, y, df):
        east_mask = (df['Conference_East'] == 1).values
        west_mask = ~east_mask
        self.east_model = KNeighborsClassifier(
            n_neighbors=self.K, weights=self._weights, metric=self.metric,
        )
        self.west_model = KNeighborsClassifier(
            n_neighbors=self.K, weights=self._weights, metric=self.metric,
        )
        self.east_model.fit(X[east_mask], y[east_mask])
        self.west_model.fit(X[west_mask], y[west_mask])
        return self

    def predict_proba(self, X, df):
        probs = np.zeros(len(df))
        east_mask = (df['Conference_East'] == 1).values
        west_mask = ~east_mask
        if east_mask.any():
            probs[east_mask] = self.east_model.predict_proba(X[east_mask])[:, 1]
        if west_mask.any():
            probs[west_mask] = self.west_model.predict_proba(X[west_mask])[:, 1]
        return probs


# =========================================================
# TRAINER (rolling-window CV)
# =========================================================
class Trainer:
    """Same rolling-window CV strategy used by log_reg / svm.

    Three expanding-window folds carved out of the combined train+val:
        fold 1: train ≤ 2013, score 2014–2015
        fold 2: train ≤ 2015, score 2016–2018
        fold 3: train ≤ 2018, score 2019–2021

    Single-fold validation against 2016–2021 ties on TopK constantly
    (158 positives, 12 selection groups). Three folds give the tuner
    a stable signal to pick from.
    """

    CV_FOLDS = [
        (2013, 2014, 2015),
        (2015, 2016, 2018),
        (2018, 2019, 2021),
    ]

    def __init__(self, cfg):
        self.cfg = cfg

    def _cv_score(self, K, power, temp, X_all, y_all, df_all):
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

            model = ConferenceKNN(K=K, power=power, temp=temp,
                                  metric=self.cfg.METRIC, eps=self.cfg.EPS)
            model.fit(X_tr, y_tr, df_tr)
            probs = model.predict_proba(X_va, df_va)

            auc = roc_auc_score(y_va, probs)
            tmp = df_va.copy()
            tmp['prob'] = probs
            tmp = structured_selection(tmp)
            topk = ((tmp['pred'] == 1) & (y_va.values == 1)).sum() / max(y_va.sum(), 1)
            scores.append(topk + 0.5 * auc)
        return float(np.mean(scores))

    def tune(self, X_train, y_train, train_df, X_val, y_val, val_df):
        """Joint sweep over (K, power, temp) by mean rolling-window CV."""
        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        df_all = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

        best = (-np.inf, None)
        print("=== Sweeping (K, power, temp) ===")
        for K in self.cfg.K_VALUES:
            for power in self.cfg.DIST_POWERS:
                for temp in self.cfg.TEMPS:
                    score = self._cv_score(K, power, temp, X_all, y_all, df_all)
                    print(f"  K={K:3d} | p={power:.1f} | T={temp:.1f} | CV score={score:.4f}")
                    if score > best[0]:
                        best = (score, (K, power, temp))
        print(f"\nBest: K={best[1][0]}, p={best[1][1]}, T={best[1][2]} (CV={best[0]:.4f})")
        return best[1]

    def train_final(self, K, power, temp,
                    X_train, y_train, train_df,
                    X_val=None, y_val=None, val_df=None):
        if X_val is not None:
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
            df_full = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        else:
            X_full, y_full, df_full = X_train, y_train, train_df

        model = ConferenceKNN(K=K, power=power, temp=temp,
                              metric=self.cfg.METRIC, eps=self.cfg.EPS)
        model.fit(X_full, y_full, df_full)
        return model


# =========================================================
# EVAL
# =========================================================
def evaluate(model, X_test, y_test, test_df):
    probs = model.predict_proba(X_test, test_df)
    tdf = test_df.copy()
    tdf['prob'] = probs
    tdf = structured_selection(tdf)
    yt = y_test.values
    yp = tdf['pred'].values

    print("\n=== FINAL RESULTS ===")
    print(f"AUC      : {roc_auc_score(yt, probs):.4f}")
    print(f"Accuracy : {accuracy_score(yt, yp):.4f}")
    print(f"Precision: {precision_score(yt, yp, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(yt, yp, zero_division=0):.4f}")
    print(f"F1       : {f1_score(yt, yp, zero_division=0):.4f}")

    print("\n=== Per-Season (Overall) ===")
    season_metrics = []
    for season, group in tdf.groupby('Season Ending Year'):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        s_g = group['prob'].values
        auc = roc_auc_score(y_g, s_g) if len(np.unique(y_g)) > 1 else np.nan
        prec = precision_score(y_g, p_g, zero_division=0)
        rec = recall_score(y_g, p_g, zero_division=0)
        f1 = f1_score(y_g, p_g, zero_division=0)
        topk = ((p_g == 1) & (y_g == 1)).sum() / max(y_g.sum(), 1)
        top12 = ((p_g == 1) & (y_g == 1)).sum() / max(p_g.sum(), 1)
        season_metrics.append((auc, accuracy_score(y_g, p_g), prec, rec, f1, topk, top12))
        print(f"{season} | AUC {auc:.4f} | P {prec:.3f} | R {rec:.3f} | F1 {f1:.3f} | "
              f"TopK {topk:.3f} | Top12 {top12:.3f}")

    season_metrics = np.array(season_metrics)
    print("\n=== Avg Per-Season ===")
    print(f"AUC          : {np.nanmean(season_metrics[:, 0]):.4f}")
    print(f"Accuracy     : {season_metrics[:, 1].mean():.4f}")
    print(f"Precision    : {season_metrics[:, 2].mean():.4f}")
    print(f"Recall       : {season_metrics[:, 3].mean():.4f}")
    print(f"F1           : {season_metrics[:, 4].mean():.4f}")
    print(f"TopK Recall  : {season_metrics[:, 5].mean():.4f}")
    print(f"Top12 Acc    : {season_metrics[:, 6].mean():.4f}")


def main():
    cfg = Config()
    pipeline = DataPipeline(cfg)
    df = pipeline.load("source/cleaned/cleaned_data.csv")
    train_df, val_df, test_df = pipeline.split(df)

    (X_train, y_train, train_df,
     X_val, y_val, val_df,
     X_test, y_test, test_df) = pipeline.prepare(train_df, val_df, test_df)

    trainer = Trainer(cfg)
    K, power, temp = trainer.tune(X_train, y_train, train_df,
                                  X_val, y_val, val_df)

    model = trainer.train_final(
        K, power, temp,
        X_train, y_train, train_df,
        X_val, y_val, val_df,
    )

    evaluate(model, X_test, y_test, test_df)


if __name__ == "__main__":
    main()
