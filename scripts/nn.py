"""
Selection-aware neural network for NBA All-Star prediction.

The architecture is a small MLP with a self-attention block over each
(Season, Conference) group. Attention is the load-bearing component:
because All-Star selection is competitive *within* a group, scoring a
player in isolation discards the comparative signal the selection
committees actually use. Letting the model attend over its peers lets
the score for one player depend on who else is in the pool that year.

The output side has two heads:

  * a "starter" head, trained to win 2-of-N (backcourt) and 3-of-N
    (frontcourt) positional contests,
  * a "reserve" head, trained to win the remaining 7-of-N slots.

Both are optimised with:

  * a *selection loss* — soft top-K that matches the actual roster
    cardinality,
  * a margin pairwise loss for ordering robustness,
  * a BCE term on the combined score for absolute calibration,
  * small auxiliary penalties on positional balance and head-overlap.

Two stability levers, in addition:

  * **Early stopping** on validation TopK recall (training-loss / AUC
    are weak proxies for the structured-selection objective).
  * **Multi-seed averaging** — train N independent models, average
    their final-layer scores. A single seed is loud on a test set with
    only ~100 positives; averaging 3 trims the seed variance roughly
    by sqrt(3).

Run:
    python scripts/nn.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================
class Config:
    SEED = 47

    # Multi-seed averaging. 3 seeds is the cheapest win — ~sqrt(3)
    # variance reduction. Going past 3 helps less than the runtime
    # cost suggests because the snapshot ensemble inside each seed
    # is already picking up most of the noise this is meant to dampen.
    N_SEEDS = 3
    SEEDS = [47, 91, 173]

    # Per-seed snapshot ensemble: keep the top-K val checkpoints and
    # average their predictions. K=3 is the sweet spot — fewer loses
    # the noise-damping benefit, more starts dragging in checkpoints
    # whose val performance dropped meaningfully below the best.
    SNAPSHOT_ENSEMBLE_K = 3

    # Long enough to converge, with patience for the noisy val signal.
    # Past ~80 epochs the val TopK is essentially flat; training further
    # just memorises training noise.
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 40

    # LR / regularization. We keep the baseline LR and weight decay
    # (those worked) and add gradient clipping for stability.
    LR = 5e-5
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    DROPOUT = 0.1

    # Architecture. Same trunk shape as the baseline; the only knob
    # we move is the embedding richness — using the full 5-way
    # position id (PG/SG/SF/PF/C) instead of the binary backcourt
    # flag preserves more roster signal.
    HIDDEN_DIMS = [256, 128, 64]
    ATTN_HEADS = 4
    ATTN_DIM = ATTN_HEADS * 32
    POS_EMB_DIM = 6
    SEASON_EMB_DIM = 8
    CONF_EMB_DIM = 4

    # Soft-selection temperature. Sweep on val showed 0.32 narrowly
    # beats 0.34 (the baseline) and 0.30 — concentrating soft top-K
    # mass on the true candidates without saturating the gradient.
    TOPK_TEMP = 0.32

    PAIRWISE_WEIGHT = 0.3
    BCE_WEIGHT = 0.4
    BALANCE_WEIGHT = 0.005
    OVERLAP_WEIGHT = 0.005

    TRAIN_END = 2015
    VAL_END = 2021

    # Inference-time score blend. Reserves take 7 of the 12 roster
    # slots, so the reserve head dominates.
    STARTER_BLEND = 0.30
    RESERVE_BLEND = 0.70

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_group_indices(df, device):
    """Return a list of index tensors, one per (Season, Conference).

    Used to slice each batch into the right competitive pool for the
    attention block and the selection losses.
    """
    keys = df[['Season Ending Year', 'Conference_East']].values
    groups = {}
    for i, k in enumerate(map(tuple, keys)):
        groups.setdefault(k, []).append(i)
    return [
        torch.tensor(v, dtype=torch.long, device=device)
        for v in groups.values()
    ]


# =========================================================
# DATA PIPELINE
# =========================================================
class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()

    def load(self, path):
        df = pd.read_csv(path)
        return self._features(df)

    def _features(self, df):
        """Pre-attention feature engineering.

        We add:

          * Categorical IDs for the embedding tables (Conf, Pos5, Season).
            Using the 5-way position id (PG/SG/SF/PF/C) instead of the
            2-way backcourt/frontcourt flag preserves more roster signal —
            a PG plays a different game than an SG.
          * Volume / efficiency composites (Usage, Impact_on_winning,
            Scoring_efficiency).
          * Conference-relative z-scores and ranks on the stats that
            drive selection: scoring, win-impact, BoxLoad.
          * Career-form lags (1- and 2-season All-Star rolling sums).
        """
        df['Conf_ID'] = df['Conference_East'].astype(int)

        # 5-class position id, with C=0, PF=1, SF=2, SG=3, PG=4. Last
        # bucket (5) is the rare "Other" position that we leave open.
        pos_map = {
            ('Pos_C', 1): 0, ('Pos_PF', 1): 1, ('Pos_SF', 1): 2,
            ('Pos_SG', 1): 3, ('Pos_PG', 1): 4,
        }
        pos5 = np.full(len(df), 5, dtype=np.int64)
        for (col, val), idx in pos_map.items():
            mask = (df[col] == val).values
            pos5[mask] = idx
        df['Pos5_ID'] = pos5

        # Backwards-compat shorthand: the structured-selection routine
        # uses the binary backcourt flag.
        df['Pos_ID'] = df['PosGroup_Backcourt'].astype(int)

        min_year = df['Season Ending Year'].min()
        df['Season_ID'] = (df['Season Ending Year'] - min_year).astype(int)

        # Volume / efficiency composites.
        df['Usage'] = df['FGA per game'] + 0.44 * df['FTA per game']
        df['Usage_x_Win'] = df['Usage'] * df['Team Win %']
        df['Impact_on_winning'] = df['PTS per game'] * df['Team Win %']
        df['Scoring_efficiency'] = df['PTS per game'] * df['TS%']

        group_keys = ['Season Ending Year', 'Conference_East']

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

        # BoxLoad is already season-z'd by the cleaning pipeline; we add
        # the conference-rank view (which the cleaning doesn't know
        # about), and that's all.
        df['BoxLoad_rank_conf'] = df.groupby(group_keys)['BoxLoad'].rank(pct=True)
        df['Usage_rank_conf'] = df.groupby(group_keys)['Usage'].rank(pct=True)

        df = df.sort_values(['Player', 'Season Ending Year']).reset_index(drop=True)

        df['AllStar_Last_Year'] = (
            df.groupby('Player')['All Star'].shift(1).fillna(0)
        )
        df['AllStar_last_2'] = (
            df.groupby('Player')['All Star']
            .shift(1).rolling(2).sum().fillna(0)
        )
        df['AllStar_last_3'] = (
            df.groupby('Player')['All Star']
            .shift(1).rolling(3).sum().fillna(0)
        )

        return df

    def split(self, df):
        train_df = df[df['Season Ending Year'] <= self.cfg.TRAIN_END].reset_index(drop=True)
        val_df = df[
            (df['Season Ending Year'] > self.cfg.TRAIN_END) &
            (df['Season Ending Year'] <= self.cfg.VAL_END)
        ].reset_index(drop=True)
        test_df = df[df['Season Ending Year'] > self.cfg.VAL_END].reset_index(drop=True)
        return train_df, val_df, test_df

    def prepare(self, train_df, val_df, test_df):
        # Drop columns that aren't continuous features. Categorical IDs
        # (Conf, Pos5, Season) are passed in separately for the
        # embedding tables.
        drop_cols = [
            'All Star', 'Player', 'Season Ending Year',
            'Prev All Stars', 'Conference_East',
            'PosGroup_Backcourt', 'PosGroup_Frontcourt',
            'Conf_ID', 'Pos5_ID', 'Pos_ID', 'Season_ID',
        ]

        def split_xy(df):
            X = df.drop(columns=drop_cols, errors='ignore')
            y = df['All Star']
            return (
                X, y,
                df['Conf_ID'], df['Pos5_ID'], df['Pos_ID'], df['Season_ID'],
            )

        X_train, y_train, c_train, p5_train, p_train, s_train = split_xy(train_df)
        X_val, y_val, c_val, p5_val, p_val, s_val = split_xy(val_df)
        X_test, y_test, c_test, p5_test, p_test, s_test = split_xy(test_df)

        # Median fill from the train slice keeps val/test out of the
        # imputer's statistics.
        med = X_train.median()
        X_train = X_train.fillna(med)
        X_val = X_val.fillna(med)
        X_test = X_test.fillna(med)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        device = self.cfg.DEVICE

        def to_t(arr, dtype):
            return torch.tensor(arr, dtype=dtype).to(device)

        return (
            to_t(X_train, torch.float32),
            to_t(y_train.values, torch.float32),
            to_t(c_train.values, torch.long),
            to_t(p5_train.values, torch.long),
            to_t(p_train.values, torch.long),
            to_t(s_train.values, torch.long),

            to_t(X_val, torch.float32),
            to_t(y_val.values, torch.float32),
            to_t(c_val.values, torch.long),
            to_t(p5_val.values, torch.long),
            to_t(p_val.values, torch.long),
            to_t(s_val.values, torch.long),

            to_t(X_test, torch.float32),
            to_t(y_test.values, torch.float32),
            to_t(c_test.values, torch.long),
            to_t(p5_test.values, torch.long),
            to_t(p_test.values, torch.long),
            to_t(s_test.values, torch.long),
        )


# =========================================================
# MODEL
# =========================================================
class AllStarNN(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()

        self.cfg = cfg

        # Categorical embeddings.
        self.conf_emb = nn.Embedding(2, cfg.CONF_EMB_DIM)
        self.pos_emb = nn.Embedding(6, cfg.POS_EMB_DIM)   # 5 positions + 1 padding
        self.season_emb = nn.Embedding(60, cfg.SEASON_EMB_DIM)

        emb_dim = cfg.CONF_EMB_DIM + cfg.POS_EMB_DIM + cfg.SEASON_EMB_DIM

        # MLP trunk. Linear/ReLU/Dropout, narrowing toward the attention
        # dimension. Layer-norms are deferred to inside the attention
        # block where the additive residual makes them load-bearing.
        prev = input_dim + emb_dim
        layers = []
        for h in cfg.HIDDEN_DIMS:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(cfg.DROPOUT),
            ]
            prev = h
        self.mlp = nn.Sequential(*layers)

        self.to_attn = nn.Linear(prev, cfg.ATTN_DIM)

        # Post-norm transformer-style block: attention -> residual ->
        # LayerNorm, then FF -> residual -> LayerNorm. The baseline used
        # this layout and it trained fine for our depth, so we stay with
        # it rather than chasing pre-norm.
        self.ln1 = nn.LayerNorm(cfg.ATTN_DIM)
        self.ln2 = nn.LayerNorm(cfg.ATTN_DIM)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.ATTN_DIM,
            num_heads=cfg.ATTN_HEADS,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.ATTN_DIM, cfg.ATTN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.ATTN_DIM, cfg.ATTN_DIM),
        )

        self.starter_head = nn.Linear(cfg.ATTN_DIM, 1)
        self.reserve_head = nn.Linear(cfg.ATTN_DIM, 1)

    def forward_group(self, x, conf, pos5, season):
        """Score a single (Season, Conference) group of players."""
        device = next(self.parameters()).device
        x = x.to(device)
        conf = conf.to(device=device, dtype=torch.long)
        pos5 = pos5.to(device=device, dtype=torch.long)
        season = season.to(device=device, dtype=torch.long)

        h = torch.cat(
            [
                x,
                self.conf_emb(conf),
                self.pos_emb(pos5),
                self.season_emb(season),
            ],
            dim=1,
        )

        h = self.mlp(h)
        h = self.to_attn(h)

        # Insert a batch dim of size 1 so MultiheadAttention treats the
        # whole group as one sequence.
        h = h.unsqueeze(0)
        attn_out, _ = self.attn(h, h, h)
        h = self.ln1(h + attn_out)
        h = self.ln2(h + self.ff(h))
        h = h.squeeze(0)

        return self.starter_head(h), self.reserve_head(h)


# =========================================================
# LOSSES
# =========================================================
def soft_selection(scores, k, temp):
    """Differentiable top-K — returns a probability distribution that
    sums to k, peaking on the highest-scoring players.
    """
    scores = scores.view(-1)
    probs = torch.softmax(scores / temp, dim=0)
    return k * probs


def selection_loss(scores, labels, k, temp):
    """MSE between the soft top-K distribution and the (k-weighted)
    label distribution. Trains the model to put its mass on the k
    true positives without forcing hard thresholding.
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if labels.sum() == 0:
        return torch.zeros(1, device=scores.device)
    z = soft_selection(scores, k, temp)
    target = k * labels / (labels.sum() + 1e-8)
    return torch.mean((z - target) ** 2)


def pairwise_loss(scores, labels, margin=0.01):
    """Margin-based ranking loss over (pos, neg) score pairs."""
    scores = scores.view(-1)
    labels = labels.view(-1)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return torch.zeros(1, device=scores.device)
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return torch.relu(margin - diff).mean()


# =========================================================
# STRUCTURED SELECTION
# =========================================================
def structured_selection(df, score_col='score'):
    """Top-2 BC + top-3 FC starters + top-7 reserves per (Season, Conf)."""
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
# TRAINER
# =========================================================
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss()

    def _compute_loss(self, s, r, yg, pg):
        """All terms for one (Season, Conference) group.

        Five components, in order of magnitude:

          1. selection loss on the starter head, split by position group
             with the right k (2 backcourt, 3 frontcourt),
          2. selection loss on the reserve head with k=7,
          3. pairwise margin loss on each head,
          4. BCE on the combined score for absolute calibration,
          5. small auxiliary penalties:
             - positional balance: encourages the 2/3 backcourt/frontcourt
               split in the combined top-12,
             - overlap: discourages a player from being soft-selected by
               *both* heads at once (the heads should specialise).
        """
        cfg = self.cfg
        loss = torch.zeros(1, device=s.device)

        bc = (pg == 1)
        fc = (pg == 0)

        if bc.sum() > 0:
            loss += selection_loss(s[bc], yg[bc], k=2, temp=cfg.TOPK_TEMP)
            loss += cfg.PAIRWISE_WEIGHT * pairwise_loss(s[bc], yg[bc])

        if fc.sum() > 0:
            loss += selection_loss(s[fc], yg[fc], k=3, temp=cfg.TOPK_TEMP)
            loss += cfg.PAIRWISE_WEIGHT * pairwise_loss(s[fc], yg[fc])

        loss += selection_loss(r, yg, k=7, temp=cfg.TOPK_TEMP)
        loss += cfg.PAIRWISE_WEIGHT * pairwise_loss(r, yg)

        combined = (s + r).squeeze()
        loss += cfg.BCE_WEIGHT * self.bce(combined, yg)

        # Positional balance: the real ballot puts 2 BC + 3 FC starters
        # into the 12-player roster, so backcourt should make up roughly
        # 2/12 of the soft-selected mass.
        z_total = soft_selection(combined, k=12, temp=cfg.TOPK_TEMP)
        actual_bc = z_total[bc].sum() / 12
        loss += cfg.BALANCE_WEIGHT * (actual_bc - 2 / 12) ** 2

        # Head specialisation: penalise mass that lands in both heads
        # at once.
        z_starters = torch.zeros_like(s.squeeze())
        if bc.sum() > 0:
            z_starters[bc] = soft_selection(s[bc], k=2, temp=cfg.TOPK_TEMP)
        if fc.sum() > 0:
            z_starters[fc] = soft_selection(s[fc], k=3, temp=cfg.TOPK_TEMP)
        z_reserves = soft_selection(r.squeeze(), k=7, temp=cfg.TOPK_TEMP)
        loss += cfg.OVERLAP_WEIGHT * (z_starters * z_reserves).sum()

        return loss

    def _val_topk_and_auc(self, model, X_val, y_val, c_val, p5_val, s_val,
                          val_df, val_groups):
        """Evaluate structured TopK *and* AUC on val.

        TopK is the right tuning signal because the test metric reduces
        to it under structured selection, but it ties frequently on a
        small val set. AUC is the stable secondary key — higher AUC at
        the same TopK means the model is ranking marginal candidates
        more confidently, which usually transfers to test.
        """
        model.eval()
        scores = torch.zeros_like(y_val)
        with torch.no_grad():
            for idx in val_groups:
                s_, r_ = model.forward_group(
                    X_val[idx], c_val[idx], p5_val[idx], s_val[idx]
                )
                blended = (
                    self.cfg.STARTER_BLEND * s_ + self.cfg.RESERVE_BLEND * r_
                ).squeeze()
                scores[idx] = torch.sigmoid(blended)

        scores_np = scores.cpu().numpy()
        y_np = y_val.cpu().numpy()
        auc = roc_auc_score(y_np, scores_np)

        tmp = val_df.copy()
        tmp['score'] = scores_np
        tmp = structured_selection(tmp, score_col='score')
        topk = ((tmp['pred'].values == 1) & (y_np == 1)).sum() / max(y_np.sum(), 1)
        return topk, auc

    def train(
        self,
        model,
        X_train, y_train, c_train, p5_train, p_train, s_train,
        X_val, y_val, c_val, p5_val, p_val, s_val,
        train_groups, val_groups, val_df,
    ):
        """Train with early stopping + Top-K snapshot ensemble.

        Instead of returning a single best-val snapshot, we keep the
        top K snapshots by val TopK (tiebroken by AUC). At inference
        we average their predictions. This effectively gives us K-1
        free models with no extra training cost, and damps the noise
        of a single val-best pick on a small val set.

        TopK alone ties frequently on this val set; AUC is the tie
        breaker, in that order.
        """
        cfg = self.cfg
        device = cfg.DEVICE
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        # Min-heap-style ranked list of (val_key, epoch, state_dict).
        top_snapshots = []  # sorted ascending by key; tail = worst kept
        K = cfg.SNAPSHOT_ENSEMBLE_K
        best_epoch = 0
        best_val_key = (-np.inf, -np.inf)
        epochs_without_improvement = 0

        for epoch in range(cfg.EPOCHS):
            model.train()
            total_loss = 0.0
            for idx in train_groups:
                optimizer.zero_grad()
                s, r = model.forward_group(
                    X_train[idx], c_train[idx], p5_train[idx], s_train[idx]
                )
                loss = self._compute_loss(s, r, y_train[idx], p_train[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()
                total_loss += loss.item()

            val_topk, val_auc = self._val_topk_and_auc(
                model, X_val, y_val, c_val, p5_val, s_val, val_df, val_groups
            )
            val_key = (val_topk, val_auc)

            if val_key > best_val_key:
                best_val_key = val_key
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Maintain the top-K snapshot list.
            snap = (val_key, epoch + 1, {k: v.detach().clone()
                                          for k, v in model.state_dict().items()})
            top_snapshots.append(snap)
            top_snapshots.sort(key=lambda t: t[0], reverse=True)
            top_snapshots = top_snapshots[:K]

            if (epoch + 1) % 10 == 0 or epoch < 5:
                print(
                    f"Epoch {epoch+1:03d} | Loss {total_loss:.2f} | "
                    f"Val TopK {val_topk:.4f} AUC {val_auc:.4f} "
                    f"(best TopK {best_val_key[0]:.4f} @ epoch {best_epoch})"
                )

            if epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stop at epoch {epoch+1} "
                    f"(no val gain in {cfg.EARLY_STOPPING_PATIENCE} epochs)"
                )
                break

        return [snap[2] for snap in top_snapshots]


# =========================================================
# EVALUATION
# =========================================================
def predict_scores(model, X, c, p5, s, n_rows, groups, cfg):
    """Forward pass + score blend for every row in a given split."""
    device = cfg.DEVICE
    model.eval()
    scores = torch.zeros(n_rows, device=device)
    with torch.no_grad():
        for idx in groups:
            s_, r_ = model.forward_group(X[idx], c[idx], p5[idx], s[idx])
            blended = (
                cfg.STARTER_BLEND * s_ + cfg.RESERVE_BLEND * r_
            ).squeeze()
            scores[idx] = torch.sigmoid(blended)
    return scores.cpu().numpy()


def evaluate(scores_np, y_test, test_df):
    """Structured-selection eval + per-season breakdown."""
    y_np = y_test.cpu().numpy()
    test_df = test_df.reset_index(drop=True).copy()
    test_df['score'] = scores_np
    test_df = structured_selection(test_df, score_col='score')
    y_pred_global = test_df['pred'].values

    print("\n=== FINAL RESULTS ===")
    print(f"AUC      : {roc_auc_score(y_np, scores_np):.4f}")
    print(f"Accuracy : {accuracy_score(y_np, y_pred_global):.4f}")
    print(f"Precision: {precision_score(y_np, y_pred_global, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_np, y_pred_global, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_np, y_pred_global, zero_division=0):.4f}")

    print("\n=== Per-Season (Conference Split) ===")
    for (season, conf), group in test_df.groupby(['Season Ending Year', 'Conference_East']):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        score_g = group['score'].values
        auc = roc_auc_score(y_g, score_g) if len(np.unique(y_g)) > 1 else np.nan
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
    for season, group in test_df.groupby('Season Ending Year'):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        score_g = group['score'].values
        auc = roc_auc_score(y_g, score_g) if len(np.unique(y_g)) > 1 else np.nan
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


# =========================================================
# MAIN
# =========================================================
def init_weights(m):
    """Kaiming init for ReLU MLPs."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def main():
    cfg = Config()

    pipeline = DataPipeline(cfg)
    df = pipeline.load('source/cleaned/cleaned_data.csv')
    train_df, val_df, test_df = pipeline.split(df)

    (X_train, y_train, c_train, p5_train, p_train, s_train,
     X_val, y_val, c_val, p5_val, p_val, s_val,
     X_test, y_test, c_test, p5_test, p_test, s_test) = pipeline.prepare(
        train_df, val_df, test_df
    )

    train_groups = build_group_indices(train_df, cfg.DEVICE)
    val_groups = build_group_indices(val_df, cfg.DEVICE)
    test_groups = build_group_indices(test_df, cfg.DEVICE)

    # Two-axis ensemble: seeds × per-seed snapshots.
    # Each seed re-seeds NumPy + PyTorch, re-inits with Kaiming, and
    # trains until early stop. The trainer returns the top-K val
    # snapshots; we average test scores across (seed × snapshot)
    # for a total of N_SEEDS * SNAPSHOT_ENSEMBLE_K contributing models.
    test_scores_accum = np.zeros(len(test_df))
    n_contrib = 0

    for seed_idx, seed in enumerate(cfg.SEEDS[: cfg.N_SEEDS]):
        print(f"\n===== Seed {seed_idx + 1}/{cfg.N_SEEDS} (seed={seed}) =====")
        set_seed(seed)

        model = AllStarNN(X_train.shape[1], cfg)
        model.apply(init_weights)
        trainer = Trainer(cfg)

        snapshots = trainer.train(
            model,
            X_train, y_train, c_train, p5_train, p_train, s_train,
            X_val, y_val, c_val, p5_val, p_val, s_val,
            train_groups, val_groups, val_df,
        )

        for state in snapshots:
            model.load_state_dict(state)
            test_scores = predict_scores(
                model, X_test, c_test, p5_test, s_test,
                len(test_df), test_groups, cfg,
            )
            test_scores_accum += test_scores
            n_contrib += 1

    test_scores_final = test_scores_accum / max(n_contrib, 1)

    print(f"\n\n===== ENSEMBLE EVAL (avg over {n_contrib} snapshots) =====")
    evaluate(test_scores_final, y_test, test_df)


if __name__ == "__main__":
    main()
