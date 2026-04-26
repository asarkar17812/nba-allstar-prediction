import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score
)

class Config:
    SEED = 47
    C_VALUES = np.logspace(-3, 2, 10)
    TRAIN_END = 2015
    VAL_END = 2021
    PENALTY = 'l2'
    SOLVER = 'lbfgs'
    MAX_ITER = 2000
    TOL = 1e-5
    TEMP = 0.5
    MAX_PAIRS = 200   
    ALPHA = 1.0


# =========================================================
# PAIRWISE DATA BUILDER
# =========================================================
def build_pairwise_dataset(X, y, df, max_pairs_per_group=200):
    df = df.reset_index(drop=True)
    y_np = y.values

    X_pairs = []
    y_pairs = []

    for (_, _), idx in df.groupby(['Season Ending Year', 'Conference_East']).groups.items():
        idx = list(idx)

        pos_idx = [i for i in idx if y_np[i] == 1]
        neg_idx = [i for i in idx if y_np[i] == 0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        pairs = [(i, j) for i in pos_idx for j in neg_idx]

        if len(pairs) > max_pairs_per_group:
            pairs = [pairs[k] for k in np.random.choice(len(pairs), max_pairs_per_group, replace=False)]

        for i, j in pairs:
            X_pairs.append(X[i] - X[j])
            y_pairs.append(1)

            X_pairs.append(X[j] - X[i])
            y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs)

class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def load(self, path):
        df = pd.read_csv(path)
        return self._features(df)

    def _features(self, df):
        group_keys = ['Season Ending Year', 'Conference_East']

        df['Usage'] = df['FGA per game'] + 0.44 * df['FTA per game']
        df['Usage_x_Win'] = df['Usage'] * df['Team Win %']
        df['Impact_on_winning'] = df['PTS per game'] * df['Team Win %']

        pts_mean = df.groupby(group_keys)['PTS per game'].transform('mean')
        pts_std  = df.groupby(group_keys)['PTS per game'].transform('std')

        df['PTS_conf_z'] = (df['PTS per game'] - pts_mean) / (pts_std + 1e-8)
        df['PTS_minus_conf_avg'] = df['PTS per game'] - pts_mean

        pts_max = df.groupby(group_keys)['PTS per game'].transform('max')
        df['PTS_conf_share'] = df['PTS per game'] / (pts_max + 1e-8)

        impact_mean = df.groupby(group_keys)['Impact_on_winning'].transform('mean')
        impact_std  = df.groupby(group_keys)['Impact_on_winning'].transform('std')

        df['Impact_conf_z'] = (
            (df['Impact_on_winning'] - impact_mean) / (impact_std + 1e-8)
        )

        df['Usage_rank_conf'] = df.groupby(group_keys)['Usage'].rank(pct=True)

        df['Win_x_AST'] = df['Team Win %'] * df['AST per game']
        df['Win_x_TRB'] = df['Team Win %'] * df['TRB per game']

        df = df.sort_values(['Player', 'Season Ending Year']).reset_index(drop=True)

        df['AllStar_Last_Year'] = df.groupby('Player')['All Star'].shift(1).fillna(0)
        df['AllStar_last_2'] = (
            df.groupby('Player')['All Star']
            .shift(1)
            .rolling(2)
            .sum()
            .fillna(0)
        )

        df['Impact_x_LastYear'] = df['Impact_on_winning'] * df['AllStar_Last_Year']

        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def split(self, df):
        train_df = df[df['Season Ending Year'] <= self.cfg.TRAIN_END].copy()
        val_df = df[(df['Season Ending Year'] > self.cfg.TRAIN_END) & (df['Season Ending Year'] <= self.cfg.VAL_END)].copy()
        test_df = df[df['Season Ending Year'] > self.cfg.VAL_END].copy()
        return train_df, val_df, test_df

    def _group_center(self, X, df):
        X_df = pd.DataFrame(X, index=df.index)

        group_means = X_df.groupby(
            [df['Season Ending Year'], df['Conference_East']]
        ).transform('mean')

        centered = (X_df - group_means).fillna(0.0)
        return centered.values

    def prepare(self, train_df, val_df, test_df):
        drop_cols = [
            'All Star', 'Player', 'Season Ending Year',
            'Prev All Stars', 'Conference_East',
            'PosGroup_Backcourt', 'PosGroup_Frontcourt'
        ]

        def split_xy(df):
            X = df.drop(columns=drop_cols)
            y = df['All Star']
            return X, y

        X_train_df, y_train = split_xy(train_df)
        X_val_df, y_val = split_xy(val_df)
        X_test_df, y_test = split_xy(test_df)

        self.feature_names = X_train_df.columns.tolist()

        X_train = self.imputer.fit_transform(X_train_df)
        X_val   = self.imputer.transform(X_val_df)
        X_test  = self.imputer.transform(X_test_df)

        X_train = self._group_center(X_train, train_df)
        X_val   = self._group_center(X_val, val_df)
        X_test  = self._group_center(X_test, test_df)

        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)

        return (
            X_train, y_train, train_df,
            X_val, y_val, val_df,
            X_test, y_test, test_df
        )

class HybridModel:
    def __init__(self, cfg, C):
        self.cfg = cfg

        # pointwise model
        self.pointwise = LogisticRegression(
            C=C,
            penalty=cfg.PENALTY,
            solver=cfg.SOLVER,
            max_iter=cfg.MAX_ITER,
            tol=cfg.TOL,
            n_jobs=-1
        )

        # pairwise model
        self.pairwise = LogisticRegression(
            C=C,
            penalty=cfg.PENALTY,
            solver=cfg.SOLVER,
            max_iter=cfg.MAX_ITER,
            tol=cfg.TOL,
            n_jobs=-1
        )

        # learnable blend weight (can tune later)
        self.alpha = cfg.ALPHA

    def fit(self, X, y, df):
        # pointwise fit
        self.pointwise.fit(X, y)

        # pairwise fit
        X_pair, y_pair = build_pairwise_dataset(X, y, df, self.cfg.MAX_PAIRS)
        self.pairwise.fit(X_pair, y_pair)

    def predict_proba(self, X, df):
        # pointwise scores
        p_point = self.pointwise.predict_proba(X)[:, 1]

        # pairwise scores (linear score)
        logits_pair = X @ self.pairwise.coef_.T + self.pairwise.intercept_
        logits_pair = logits_pair.squeeze()
        p_pair = 1 / (1 + np.exp(-logits_pair))

        # combine
        probs = self.alpha * p_point + (1 - self.alpha) * p_pair

        # temperature scaling
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        logits = np.log(probs) - np.log(1 - probs)
        logits = logits / self.cfg.TEMP
        probs = 1 / (1 + np.exp(-logits))

        return probs

def structured_selection(df):
    df = df.copy()
    df['pred'] = 0

    for (season, conf), group in df.groupby(['Season Ending Year', 'Conference_East']):
        bc = group[group['PosGroup_Backcourt'] == 1]
        fc = group[group['PosGroup_Frontcourt'] == 1]

        starters = pd.concat([
            bc.sort_values('prob', ascending=False).head(2),
            fc.sort_values('prob', ascending=False).head(3)
        ])

        remaining = group.drop(index=starters.index)
        reserves = remaining.sort_values('prob', ascending=False).head(7)

        selected = pd.concat([starters, reserves])
        df.loc[selected.index, 'pred'] = 1

    return df

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def tune(self, X_train, y_train, train_df, X_val, y_val, val_df):
        results = []

        for C in self.cfg.C_VALUES:
            model = HybridModel(self.cfg, C)
            model.fit(X_train, y_train, train_df)

            probs = model.predict_proba(X_val, val_df)
            auc = roc_auc_score(y_val, probs)

            temp_df = val_df.copy()
            temp_df['prob'] = probs
            temp_df = structured_selection(temp_df)

            true_allstars = y_val.sum()
            selected_correct = ((temp_df['pred'] == 1) & (y_val == 1)).sum()
            topk_recall = selected_correct / (true_allstars + 1e-8)

            results.append((C, auc, topk_recall))
            print(f"C={C:.4f} | AUC={auc:.4f} | TopK={topk_recall:.4f}")

        best_C = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)[0][0]
        print("\nBest C:", best_C)
        return best_C

    def train_final(self, C, X_train, y_train, train_df):
        model = HybridModel(self.cfg, C)
        model.fit(X_train, y_train, train_df)
        return model


def evaluate(model, X_test, y_test, test_df):
    probs = model.predict_proba(X_test, test_df)

    temp_df = test_df.copy()
    temp_df['prob'] = probs
    temp_df = structured_selection(temp_df)

    y_true = y_test.values
    y_pred = temp_df['pred'].values

    # =========================================================
    # GLOBAL METRICS
    # =========================================================
    print("\n=== FINAL RESULTS ===")
    print("AUC:", roc_auc_score(y_true, probs))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))

    # =========================================================
    # PER (SEASON, CONFERENCE)
    # =========================================================
    print("\n=== Per-Season (Conference Split) ===")

    season_conf_metrics = []

    for (season, conf), group in temp_df.groupby(['Season Ending Year', 'Conference_East']):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        prob_g = group['prob'].values

        if len(np.unique(y_g)) > 1:
            auc = roc_auc_score(y_g, prob_g)
        else:
            auc = np.nan

        acc = accuracy_score(y_g, p_g)
        prec = precision_score(y_g, p_g, zero_division=0)
        rec = recall_score(y_g, p_g, zero_division=0)
        f1 = f1_score(y_g, p_g, zero_division=0)

        true_allstars = y_g.sum()
        selected_correct = ((p_g == 1) & (y_g == 1)).sum()
        topk_recall = selected_correct / (true_allstars + 1e-8)

        total_selected = p_g.sum()
        top12_acc = selected_correct / (total_selected + 1e-8)

        season_conf_metrics.append((auc, acc, prec, rec, f1, topk_recall, top12_acc))

        print(
            f"{season} Conf {conf} | "
            f"AUC {auc:.4f} | "
            f"Acc {acc:.3f} | "
            f"P {prec:.3f} | R {rec:.3f} | F1 {f1:.3f} | "
            f"TopK {topk_recall:.3f} | Top12Acc {top12_acc:.3f}"
        )

    # =========================================================
    # PER SEASON (COMBINED)
    # =========================================================
    print("\n=== Per-Season (Overall) ===")

    season_metrics = []

    for season, group in temp_df.groupby('Season Ending Year'):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        prob_g = group['prob'].values

        if len(np.unique(y_g)) > 1:
            auc = roc_auc_score(y_g, prob_g)
        else:
            auc = np.nan

        acc = accuracy_score(y_g, p_g)
        prec = precision_score(y_g, p_g, zero_division=0)
        rec = recall_score(y_g, p_g, zero_division=0)
        f1 = f1_score(y_g, p_g, zero_division=0)

        true_allstars = y_g.sum()
        selected_correct = ((p_g == 1) & (y_g == 1)).sum()
        topk_recall = selected_correct / (true_allstars + 1e-8)

        total_selected = p_g.sum()
        top12_acc = selected_correct / (total_selected + 1e-8)

        season_metrics.append((auc, acc, prec, rec, f1, topk_recall, top12_acc))

        print(
            f"{season} | "
            f"AUC {auc:.4f} | "
            f"Acc {acc:.3f} | "
            f"P {prec:.3f} | R {rec:.3f} | F1 {f1:.3f} | "
            f"TopK {topk_recall:.3f} | Top12Acc {top12_acc:.3f}"
        )

    # =========================================================
    # AVG PER-SEASON
    # =========================================================
    season_metrics = np.array(season_metrics)

    print("\n=== Avg Per-Season Metrics ===")
    print("AUC        :", np.nanmean(season_metrics[:, 0]))
    print("Accuracy   :", season_metrics[:, 1].mean())
    print("Precision  :", season_metrics[:, 2].mean())
    print("Recall     :", season_metrics[:, 3].mean())
    print("F1         :", season_metrics[:, 4].mean())
    print("Top 12 Recall:", season_metrics[:, 5].mean())
    print("Top 12 Accuracy:", season_metrics[:, 6].mean())

def print_feature_importance(model, feature_names, top_k=10):
        print("\n=== FEATURE IMPORTANCE (POINTWISE) ===")

        coefs = model.pointwise.coef_.flatten()

        df = pd.DataFrame({
            "feature": feature_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs)
        }).sort_values("abs_coef", ascending=False)

        print("\nTop Positive Features:")
        print(df.sort_values("coef", ascending=False).head(top_k)[["feature", "coef"]])

        print("\nTop Negative Features:")
        print(df.sort_values("coef", ascending=True).head(top_k)[["feature", "coef"]])

        # =========================
        # PAIRWISE
        # =========================
        print("\n=== FEATURE IMPORTANCE (PAIRWISE) ===")

        coefs_pair = model.pairwise.coef_.flatten()

        df_pair = pd.DataFrame({
            "feature": feature_names,
            "coef": coefs_pair,
            "abs_coef": np.abs(coefs_pair)
        }).sort_values("abs_coef", ascending=False)

        print("\nTop Positive (helps beat others):")
        print(df_pair.sort_values("coef", ascending=False).head(top_k)[["feature", "coef"]])

        print("\nTop Negative (hurts ranking):")
        print(df_pair.sort_values("coef", ascending=True).head(top_k)[["feature", "coef"]])

# =========================================================
# MAIN
# =========================================================
def main():
    cfg = Config()

    pipeline = DataPipeline(cfg)
    df = pipeline.load('source\\cleaned\\cleaned_data.csv')

    train_df, val_df, test_df = pipeline.split(df)

    (X_train, y_train, train_df,
     X_val, y_val, val_df,
     X_test, y_test, test_df) = pipeline.prepare(train_df, val_df, test_df)

    trainer = Trainer(cfg)

    best_C = trainer.tune(
        X_train, y_train, train_df,
        X_val, y_val, val_df
    )

    model = trainer.train_final(best_C, X_train, y_train, train_df)

    print_feature_importance(model, pipeline.feature_names)

    evaluate(model, X_test, y_test, test_df)


if __name__ == "__main__":
    main()