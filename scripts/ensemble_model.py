"""
Ensemble of the four base models.

The four models we've already built — logistic regression, neural net,
linear SVM, per-conference k-NN — make different mistakes:

  * LR / SVM are linear in the engineered feature space and rank
    well-globally but miss interaction effects.
  * NN catches roster-shape interactions through attention and the
    starter/reserve heads.
  * k-NN captures local density: it's the only model that uses
    *peer-similarity* directly, which catches some of the marginal
    reserve picks the parametric models miss.

If those error patterns are partially independent, averaging their
predictions will beat the best single model. We test that here.

Pipeline:

  1. Fit each base model on the train split using the hyperparameters
     each model's own script tunes to (hard-coded below so this
     script runs without rerunning the full per-model sweeps).
  2. Score val with each base model.
  3. Tune ensemble weights (w_lr, w_nn, w_svm, w_knn) on the unit
     simplex by maximising val TopK recall.
  4. Refit each base model on train + val, score test, blend with the
     tuned weights, run structured selection, evaluate.

Run:
    python scripts/ensemble_model.py
"""

import itertools
import sys
from pathlib import Path  # noqa: F401

import numpy as np
import pandas as pd
import torch
import torch.nn as nn_module
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from svm_pipelines import get_base_pipeline, structured_selection


# =========================================================
# CONFIG
# =========================================================
class Config:
    SEED = 47
    TRAIN_END = 2015
    VAL_END = 2021

    # Hard-coded best hyperparameters from the per-model CV sweeps.
    # Update these when re-tuning the individual models.
    LR_C = 0.066
    LR_ALPHA = 0.7
    LR_TEMP = 1.0
    LR_MAX_PAIRS = 400

    SVM_C = 10.0
    # The SVM factory pulls these off the cfg; matched to svm_pipelines.py.
    MAX_ITER = 100000
    TOL = 1e-3
    PROBABILITY = False

    KNN_K = 13
    KNN_POWER = 3.0
    KNN_TEMP = 0.5
    KNN_METRIC = 'euclidean'
    KNN_EPS = 1e-6

    NN_LR = 5e-5
    NN_WEIGHT_DECAY = 1e-5
    NN_EPOCHS = 60          # single-seed compromise; full nn.py uses ensembling
    NN_HIDDEN_DIMS = [256, 128, 64]
    NN_DROPOUT = 0.1
    NN_ATTN_HEADS = 4
    NN_ATTN_DIM = 128
    NN_TOPK_TEMP = 0.32
    NN_PAIRWISE_WEIGHT = 0.3
    NN_BCE_WEIGHT = 0.4
    NN_GRAD_CLIP = 1.0

    # Weight sweep granularity. Ten steps per axis = 220 distinct
    # 4-simplex weight vectors after the sum-to-1 constraint, which
    # runs in well under a second of post-hoc blending.
    WEIGHT_GRID = np.arange(0, 1.001, 0.1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# DATA PIPELINE
# =========================================================
class DataPipeline:
    """Shared preprocessing for all four base models.

    Identical to the log_reg / svm / kNN pipelines so the per-model
    predictions are comparable. Group-centred features and median
    imputation, with TS%, BoxLoad, GamesFrac, PrimeAge dropped (those
    destabilise the linear models — see log_reg.py).
    """

    DROP_COLS = [
        'All Star', 'Player', 'Season Ending Year',
        'Prev All Stars', 'Conference_East',
        'PosGroup_Backcourt', 'PosGroup_Frontcourt',
        'TS%', 'BoxLoad', 'GamesFrac', 'PrimeAge',
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler_linear = StandardScaler()
        self.scaler_nn = StandardScaler()
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

        return df.replace([np.inf, -np.inf], np.nan)

    def split(self, df):
        train = df[df['Season Ending Year'] <= self.cfg.TRAIN_END].copy()
        val = df[
            (df['Season Ending Year'] > self.cfg.TRAIN_END)
            & (df['Season Ending Year'] <= self.cfg.VAL_END)
        ].copy()
        test = df[df['Season Ending Year'] > self.cfg.VAL_END].copy()
        return train, val, test

    def _group_center(self, X, df):
        X_df = pd.DataFrame(X, index=df.index)
        means = X_df.groupby(
            [df['Season Ending Year'], df['Conference_East']]
        ).transform('mean')
        return (X_df - means).fillna(0.0).values

    def prepare(self, train_df, val_df, test_df):
        def split_xy(df):
            X = df.drop(columns=self.DROP_COLS, errors='ignore')
            return X, df['All Star']

        X_train_df, y_train = split_xy(train_df)
        X_val_df, y_val = split_xy(val_df)
        X_test_df, y_test = split_xy(test_df)
        self.feature_names = X_train_df.columns.tolist()

        # Median imputation (fit on train, apply to all).
        X_train = self.imputer.fit_transform(X_train_df)
        X_val = self.imputer.transform(X_val_df)
        X_test = self.imputer.transform(X_test_df)

        # Two flavours of post-processing:
        #
        #   * "linear" variant — group-centred + global-scaled. This is
        #     what LR and SVM expect.
        #   * "nn" variant — globally scaled only, no group centring.
        #     The NN's attention layer handles the cross-player
        #     comparison itself; group-centring on top of attention
        #     was empirically a wash and made the embedding inputs
        #     less interpretable. k-NN uses this same variant.
        X_train_lin = self._group_center(X_train, train_df)
        X_val_lin = self._group_center(X_val, val_df)
        X_test_lin = self._group_center(X_test, test_df)
        X_train_lin = self.scaler_linear.fit_transform(X_train_lin)
        X_val_lin = self.scaler_linear.transform(X_val_lin)
        X_test_lin = self.scaler_linear.transform(X_test_lin)

        X_train_nn = self.scaler_nn.fit_transform(X_train)
        X_val_nn = self.scaler_nn.transform(X_val)
        X_test_nn = self.scaler_nn.transform(X_test)

        return dict(
            lin=(X_train_lin, X_val_lin, X_test_lin),
            nn=(X_train_nn, X_val_nn, X_test_nn),
            y=(y_train, y_val, y_test),
            df=(train_df, val_df, test_df),
        )


# =========================================================
# HYBRID LR (pointwise + pairwise blend)
# =========================================================
def build_pairwise_dataset(X, y, df, max_pairs, rng):
    df = df.reset_index(drop=True)
    y_np = y.values
    X_pairs, y_pairs = [], []
    for _, idx in df.groupby(['Season Ending Year', 'Conference_East']).groups.items():
        idx = list(idx)
        pos_idx = [i for i in idx if y_np[i] == 1]
        neg_idx = [i for i in idx if y_np[i] == 0]
        if not pos_idx or not neg_idx:
            continue
        pairs = [(i, j) for i in pos_idx for j in neg_idx]
        if len(pairs) > max_pairs:
            chosen = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[k] for k in chosen]
        for i, j in pairs:
            X_pairs.append(X[i] - X[j]); y_pairs.append(1)
            X_pairs.append(X[j] - X[i]); y_pairs.append(0)
    return np.array(X_pairs), np.array(y_pairs)


class HybridLR:
    def __init__(self, cfg):
        self.cfg = cfg
        common = dict(C=cfg.LR_C, penalty='l2', solver='lbfgs',
                      max_iter=4000, tol=1e-6, n_jobs=-1)
        self.pointwise = LogisticRegression(**common)
        self.pairwise = LogisticRegression(**common)

    def fit(self, X, y, df):
        rng = np.random.default_rng(self.cfg.SEED)
        self.pointwise.fit(X, y)
        Xp, yp = build_pairwise_dataset(X, y, df, self.cfg.LR_MAX_PAIRS, rng)
        self.pairwise.fit(Xp, yp)
        return self

    def predict_proba(self, X, df):
        p_point = self.pointwise.predict_proba(X)[:, 1]
        logits_pair = (X @ self.pairwise.coef_.T + self.pairwise.intercept_).squeeze()
        p_pair = 1.0 / (1.0 + np.exp(-logits_pair))
        probs = self.cfg.LR_ALPHA * p_point + (1 - self.cfg.LR_ALPHA) * p_pair
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        logits = (np.log(probs) - np.log(1 - probs)) / self.cfg.LR_TEMP
        return 1.0 / (1.0 + np.exp(-logits))


# =========================================================
# PER-CONFERENCE kNN
# =========================================================
class ConferenceKNN:
    def __init__(self, cfg):
        self.cfg = cfg
        self.east = self.west = None

    def _weights(self, distances):
        d = distances / self.cfg.KNN_TEMP
        return 1.0 / (np.power(d + self.cfg.KNN_EPS, self.cfg.KNN_POWER))

    def fit(self, X, y, df):
        from sklearn.neighbors import KNeighborsClassifier
        east_mask = (df['Conference_East'] == 1).values
        west_mask = ~east_mask
        self.east = KNeighborsClassifier(
            n_neighbors=self.cfg.KNN_K, weights=self._weights,
            metric=self.cfg.KNN_METRIC,
        )
        self.west = KNeighborsClassifier(
            n_neighbors=self.cfg.KNN_K, weights=self._weights,
            metric=self.cfg.KNN_METRIC,
        )
        self.east.fit(X[east_mask], y[east_mask])
        self.west.fit(X[west_mask], y[west_mask])
        return self

    def predict_proba(self, X, df):
        probs = np.zeros(len(df))
        east_mask = (df['Conference_East'] == 1).values
        west_mask = ~east_mask
        if east_mask.any():
            probs[east_mask] = self.east.predict_proba(X[east_mask])[:, 1]
        if west_mask.any():
            probs[west_mask] = self.west.predict_proba(X[west_mask])[:, 1]
        return probs


# =========================================================
# NEURAL NETWORK (single seed, single snapshot)
# =========================================================
# Mirrors the architecture in nn.py but trains one seed only — full
# multi-seed × snapshot ensembling lives in nn.py. The ensemble script
# wants a quick representative NN signal, not the most accurate one.

def _pos5_from_dummies(df):
    pos5 = np.full(len(df), 5, dtype=np.int64)
    for col, idx in [('Pos_C', 0), ('Pos_PF', 1), ('Pos_SF', 2),
                     ('Pos_SG', 3), ('Pos_PG', 4)]:
        if col in df.columns:
            pos5[(df[col] == 1).values] = idx
    return pos5


def _build_group_indices(df, device):
    groups = {}
    for i, k in enumerate(map(tuple, df[['Season Ending Year', 'Conference_East']].values)):
        groups.setdefault(k, []).append(i)
    return [torch.tensor(v, dtype=torch.long, device=device) for v in groups.values()]


class AllStarNN(nn_module.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.conf_emb = nn_module.Embedding(2, 4)
        self.pos_emb = nn_module.Embedding(6, 6)
        self.season_emb = nn_module.Embedding(60, 8)
        emb_dim = 4 + 6 + 8

        prev = input_dim + emb_dim
        layers = []
        for h in cfg.NN_HIDDEN_DIMS:
            layers += [nn_module.Linear(prev, h), nn_module.ReLU(),
                       nn_module.Dropout(cfg.NN_DROPOUT)]
            prev = h
        self.mlp = nn_module.Sequential(*layers)
        self.to_attn = nn_module.Linear(prev, cfg.NN_ATTN_DIM)
        self.attn = nn_module.MultiheadAttention(
            embed_dim=cfg.NN_ATTN_DIM, num_heads=cfg.NN_ATTN_HEADS, batch_first=True,
        )
        self.ln1 = nn_module.LayerNorm(cfg.NN_ATTN_DIM)
        self.ln2 = nn_module.LayerNorm(cfg.NN_ATTN_DIM)
        self.ff = nn_module.Sequential(
            nn_module.Linear(cfg.NN_ATTN_DIM, cfg.NN_ATTN_DIM),
            nn_module.ReLU(),
            nn_module.Linear(cfg.NN_ATTN_DIM, cfg.NN_ATTN_DIM),
        )
        self.starter_head = nn_module.Linear(cfg.NN_ATTN_DIM, 1)
        self.reserve_head = nn_module.Linear(cfg.NN_ATTN_DIM, 1)

    def forward_group(self, x, conf, pos5, season):
        h = torch.cat([
            x, self.conf_emb(conf), self.pos_emb(pos5), self.season_emb(season),
        ], dim=1)
        h = self.mlp(h)
        h = self.to_attn(h).unsqueeze(0)
        attn_out, _ = self.attn(h, h, h)
        h = self.ln1(h + attn_out)
        h = self.ln2(h + self.ff(h))
        h = h.squeeze(0)
        return self.starter_head(h), self.reserve_head(h)


def _soft_selection(scores, k, temp):
    scores = scores.view(-1)
    return k * torch.softmax(scores / temp, dim=0)


def _selection_loss(scores, labels, k, temp):
    scores = scores.view(-1); labels = labels.view(-1)
    if labels.sum() == 0:
        return torch.zeros(1, device=scores.device)
    z = _soft_selection(scores, k, temp)
    target = k * labels / (labels.sum() + 1e-8)
    return torch.mean((z - target) ** 2)


def _pairwise_loss(scores, labels, margin=0.01):
    scores = scores.view(-1); labels = labels.view(-1)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return torch.zeros(1, device=scores.device)
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return torch.relu(margin - diff).mean()


class NNWrapper:
    """One-seed NN with the same training objective as nn.py.

    Deliberately simpler than the production nn.py runner: no
    multi-seed averaging, no per-seed snapshot ensembling. The
    ensemble script just needs a single representative NN signal —
    the full 9-model production NN can be plugged in if you want the
    best possible NN contribution.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def _prep_tensors(self, X, df):
        device = self.cfg.DEVICE
        pos5 = _pos5_from_dummies(df)
        min_year = df['Season Ending Year'].min()
        season = (df['Season Ending Year'] - min_year).astype(int)
        conf = df['Conference_East'].astype(int)
        return (
            torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(conf.values, dtype=torch.long, device=device),
            torch.tensor(pos5, dtype=torch.long, device=device),
            torch.tensor(season.values, dtype=torch.long, device=device),
        )

    def fit(self, X, y, df):
        torch.manual_seed(self.cfg.SEED)
        np.random.seed(self.cfg.SEED)

        X_t, c_t, p5_t, s_t = self._prep_tensors(X, df)
        y_t = torch.tensor(y.values, dtype=torch.float32, device=self.cfg.DEVICE)
        groups = _build_group_indices(df.reset_index(drop=True), self.cfg.DEVICE)

        self.model = AllStarNN(X.shape[1], self.cfg).to(self.cfg.DEVICE)
        for m in self.model.modules():
            if isinstance(m, nn_module.Linear):
                nn_module.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn_module.init.zeros_(m.bias)

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.NN_LR, weight_decay=self.cfg.NN_WEIGHT_DECAY,
        )
        bce = nn_module.BCEWithLogitsLoss()
        pg = torch.tensor(
            (df['PosGroup_Backcourt'] == 1).astype(int).values,
            dtype=torch.long, device=self.cfg.DEVICE,
        )

        for _ in range(self.cfg.NN_EPOCHS):
            self.model.train()
            for idx in groups:
                opt.zero_grad()
                s, r = self.model.forward_group(
                    X_t[idx], c_t[idx], p5_t[idx], s_t[idx]
                )
                yg = y_t[idx]
                pg_g = pg[idx]
                bc = (pg_g == 1); fc = (pg_g == 0)
                loss = torch.zeros(1, device=s.device)
                if bc.sum() > 0:
                    loss = loss + _selection_loss(s[bc], yg[bc], 2, self.cfg.NN_TOPK_TEMP)
                    loss = loss + self.cfg.NN_PAIRWISE_WEIGHT * _pairwise_loss(s[bc], yg[bc])
                if fc.sum() > 0:
                    loss = loss + _selection_loss(s[fc], yg[fc], 3, self.cfg.NN_TOPK_TEMP)
                    loss = loss + self.cfg.NN_PAIRWISE_WEIGHT * _pairwise_loss(s[fc], yg[fc])
                loss = loss + _selection_loss(r, yg, 7, self.cfg.NN_TOPK_TEMP)
                loss = loss + self.cfg.NN_PAIRWISE_WEIGHT * _pairwise_loss(r, yg)
                combined = (s + r).squeeze()
                loss = loss + self.cfg.NN_BCE_WEIGHT * bce(combined, yg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.NN_GRAD_CLIP)
                opt.step()
        return self

    def predict_proba(self, X, df):
        self.model.eval()
        X_t, c_t, p5_t, s_t = self._prep_tensors(X, df)
        groups = _build_group_indices(df.reset_index(drop=True), self.cfg.DEVICE)
        scores = torch.zeros(len(df), device=self.cfg.DEVICE)
        with torch.no_grad():
            for idx in groups:
                s, r = self.model.forward_group(
                    X_t[idx], c_t[idx], p5_t[idx], s_t[idx]
                )
                blended = (0.3 * s + 0.7 * r).squeeze()
                scores[idx] = torch.sigmoid(blended)
        return scores.cpu().numpy()


# =========================================================
# ENSEMBLE BLENDING
# =========================================================
def normalize_probs(p):
    """Min-max into [0, 1] so different score scales blend coherently."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (p - p.min()) / (p.max() - p.min() + 1e-8)


def rank_within_groups(scores, df):
    """Convert scores to within-(season, conference) percentile ranks.

    This is what actually matters for structured selection — only the
    ordering within each group counts. Ranking removes scale
    differences between models entirely, so a Borda-style average
    across models is a clean and overfit-resistant blend.
    """
    s = pd.Series(scores, index=df.index)
    return s.groupby(
        [df['Season Ending Year'], df['Conference_East']]
    ).rank(pct=True).values


def topk_recall_from_probs(probs, y, df):
    tmp = df.copy()
    tmp['prob'] = probs
    tmp = structured_selection(tmp)
    return ((tmp['pred'] == 1) & (y.values == 1)).sum() / max(y.sum(), 1)


def tune_weights(val_probs, y_val, val_df, weight_grid):
    """Grid search over the 4-simplex for the best convex combination.

    This is genuinely overfit-prone on the 158-positive val set — a
    weight vector that wins on val by a fraction often loses on test.
    We report it for transparency but the *recommended* blend is the
    equal-weighted rank-averaged ensemble below, which doesn't pick
    any of these dials on val at all.
    """
    names = list(val_probs.keys())
    best = (-np.inf, None)
    for combo in itertools.product(weight_grid, repeat=len(names)):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        blended = np.zeros_like(next(iter(val_probs.values())))
        for w, name in zip(combo, names):
            blended = blended + w * val_probs[name]
        topk = topk_recall_from_probs(blended, y_val, val_df)
        if topk > best[0]:
            best = (topk, dict(zip(names, combo)))
    return best


# =========================================================
# EVAL
# =========================================================
def evaluate(probs, y_test, test_df, tag=""):
    tdf = test_df.copy()
    tdf['prob'] = probs
    tdf = structured_selection(tdf)
    yt = y_test.values
    yp = tdf['pred'].values
    print(f"\n=== {tag} ===")
    print(f"AUC      : {roc_auc_score(yt, probs):.4f}")
    print(f"Accuracy : {accuracy_score(yt, yp):.4f}")
    print(f"Precision: {precision_score(yt, yp, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(yt, yp, zero_division=0):.4f}")
    print(f"F1       : {f1_score(yt, yp, zero_division=0):.4f}")
    return tdf


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = Config()
    pipeline = DataPipeline(cfg)
    df = pipeline.load("source/cleaned/cleaned_data.csv")
    train_df, val_df, test_df = pipeline.split(df)
    data = pipeline.prepare(train_df, val_df, test_df)

    X_train_lin, X_val_lin, X_test_lin = data['lin']
    X_train_nn,  X_val_nn,  X_test_nn  = data['nn']
    y_train, y_val, y_test = data['y']
    train_df, val_df, test_df = data['df']

    # ---- Phase 1: fit each base model on train, score val ----
    print("Phase 1: fitting base models on train, scoring val ...")

    lr = HybridLR(cfg).fit(X_train_lin, y_train, train_df)
    lr_val = normalize_probs(lr.predict_proba(X_val_lin, val_df))

    svm_pipe = get_base_pipeline(cfg, c_val=cfg.SVM_C)
    svm_pipe.fit(X_train_lin, y_train)
    svm_val_raw = svm_pipe.decision_function(X_val_lin)
    svm_val = normalize_probs(svm_val_raw)

    knn = ConferenceKNN(cfg).fit(X_train_lin, y_train, train_df)
    knn_val = normalize_probs(knn.predict_proba(X_val_lin, val_df))

    nn_wrap = NNWrapper(cfg).fit(X_train_nn, y_train, train_df)
    nn_val = normalize_probs(nn_wrap.predict_proba(X_val_nn, val_df))

    # ---- Phase 2: tune ensemble weights on val ----
    print("\nPhase 2: tuning ensemble weights on val (4-simplex grid) ...")
    val_probs = {"lr": lr_val, "nn": nn_val, "svm": svm_val, "knn": knn_val}
    best_topk, best_weights = tune_weights(val_probs, y_val, val_df, cfg.WEIGHT_GRID)
    print(f"\nBest val TopK = {best_topk:.4f}")
    print("Best weights  =", {k: round(v, 2) for k, v in best_weights.items()})

    # ---- Phase 3: refit on train+val, score test, blend ----
    print("\nPhase 3: refitting base models on train + val ...")

    full_lin = np.concatenate([X_train_lin, X_val_lin], axis=0)
    full_nn = np.concatenate([X_train_nn, X_val_nn], axis=0)
    full_y = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    lr_full = HybridLR(cfg).fit(full_lin, full_y, full_df)
    svm_full = get_base_pipeline(cfg, c_val=cfg.SVM_C).fit(full_lin, full_y)
    knn_full = ConferenceKNN(cfg).fit(full_lin, full_y, full_df)
    nn_full = NNWrapper(cfg).fit(full_nn, full_y, full_df)

    lr_test = normalize_probs(lr_full.predict_proba(X_test_lin, test_df))
    svm_test = normalize_probs(svm_full.decision_function(X_test_lin))
    knn_test = normalize_probs(knn_full.predict_proba(X_test_lin, test_df))
    nn_test = normalize_probs(nn_full.predict_proba(X_test_nn, test_df))

    # ---- Per-model test eval (for the comparison table) ----
    evaluate(lr_test,  y_test, test_df, tag="LR (in ensemble)")
    evaluate(svm_test, y_test, test_df, tag="SVM (in ensemble)")
    evaluate(knn_test, y_test, test_df, tag="kNN (in ensemble)")
    evaluate(nn_test,  y_test, test_df, tag="NN single-seed (in ensemble)")

    # ---- Dump per-model selections for downstream figures ----
    # The agreement matrix figure in the README is built from these.
    pred_dump_path = Path("assets/figures/test_predictions.csv")
    pred_dump_path.parent.mkdir(parents=True, exist_ok=True)

    def _selections(probs):
        tdf = test_df.copy()
        tdf['prob'] = probs
        return structured_selection(tdf)['pred'].values

    dump = test_df[['Player', 'Season Ending Year', 'Conference_East', 'All Star']].copy()
    dump['pred_lr']  = _selections(lr_test)
    dump['pred_svm'] = _selections(svm_test)
    dump['pred_knn'] = _selections(knn_test)
    dump['pred_nn']  = _selections(nn_test)
    dump['score_lr']  = lr_test
    dump['score_svm'] = svm_test
    dump['score_knn'] = knn_test
    dump['score_nn']  = nn_test
    dump.to_csv(pred_dump_path, index=False)
    print(f"\nDumped per-model test predictions to {pred_dump_path}")

    # ---- Three ensemble flavours ----
    print("\n" + "=" * 60)
    print("Ensemble variants:")
    print("=" * 60)

    # (a) Val-tuned weights — overfit-prone but a useful diagnostic.
    blended_tuned = (
        best_weights["lr"]  * lr_test  +
        best_weights["nn"]  * nn_test  +
        best_weights["svm"] * svm_test +
        best_weights["knn"] * knn_test
    )
    evaluate(blended_tuned, y_test, test_df,
             tag=f"Val-tuned blend {best_weights}")

    # (b) Equal-weighted average of the normalized probabilities. No
    # tuning, no overfit risk. With four diverse models this is often
    # the strongest "fair" ensemble.
    blended_equal = 0.25 * (lr_test + nn_test + svm_test + knn_test)
    evaluate(blended_equal, y_test, test_df, tag="Equal-weighted (0.25 each)")

    # (c) Rank-averaged Borda count within each (Season, Conference)
    # group. Removes scale differences between models entirely so it's
    # the most overfit-resistant blend we have.
    lr_rank  = rank_within_groups(lr_test,  test_df)
    svm_rank = rank_within_groups(svm_test, test_df)
    knn_rank = rank_within_groups(knn_test, test_df)
    nn_rank  = rank_within_groups(nn_test,  test_df)
    borda = (lr_rank + svm_rank + knn_rank + nn_rank) / 4.0
    evaluate(borda, y_test, test_df,
             tag="Rank-averaged (Borda count, within-group)")


if __name__ == "__main__":
    main()
