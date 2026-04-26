import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score
)
import torch
import torch.nn as nn

class Config:
    SEED = 47
    LR = 0.00005
    WEIGHT_DECAY = 0.00001
    EPOCHS = 80
    HIDDEN_DIMS = [256, 128, 64]
    DROPOUT = 0.1
    ATTN_HEADS = 4
    ATTN_DIM = ATTN_HEADS * 32
    TOPK_TEMP = 0.34
    PAIRWISE_WEIGHT = 0.3
    BCE_WEIGHT = 0.4
    GRAD_CLIP = 1.0
    TRAIN_END = 2015
    VAL_END = 2021
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_group_indices(df, device):
    keys = df[['Season Ending Year', 'Conference_East']].values

    groups = {}
    for i, k in enumerate(map(tuple, keys)):
        groups.setdefault(k, []).append(i)

    return [
        torch.tensor(v, dtype=torch.long, device=device)
        for v in groups.values()
    ]

class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()

    def load(self, path):
        df = pd.read_csv(path)
        return self._features(df)

    def _features(self, df):
        df['Conf_ID'] = df['Conference_East'].astype(int)
        df['Pos_ID'] = df['PosGroup_Backcourt'].astype(int)

        min_year = df['Season Ending Year'].min()
        df['Season_ID'] = df['Season Ending Year'] - min_year

        df['Usage'] = df['FGA per game'] + 0.44 * df['FTA per game']
        df['Usage_x_Win'] = df['Usage'] * df['Team Win %']
        df['Impact_on_winning'] = df['PTS per game'] * df['Team Win %']

        group_keys = ['Season Ending Year', 'Conference_East']

        pts_mean = df.groupby(group_keys)['PTS per game'].transform('mean')
        pts_std = df.groupby(group_keys)['PTS per game'].transform('std')

        df['PTS_conf_z'] = (df['PTS per game'] - pts_mean) / (pts_std + 1e-8)

        pts_max = df.groupby(group_keys)['PTS per game'].transform('max')
        df['PTS_conf_share'] = df['PTS per game'] / (pts_max + 1e-8)

        impact_mean = df.groupby(group_keys)['Impact_on_winning'].transform('mean')
        impact_std = df.groupby(group_keys)['Impact_on_winning'].transform('std')

        df['Impact_conf_z'] = (
            (df['Impact_on_winning'] - impact_mean) /
            (impact_std + 1e-8)
        )

        df = df.sort_values(['Player', 'Season Ending Year']).reset_index(drop=True)

        df['AllStar_Last_Year'] = (
            df.groupby('Player')['All Star'].shift(1).fillna(0)
        )

        df['AllStar_last_2'] = (
            df.groupby('Player')['All Star']
            .shift(1)
            .rolling(2)
            .sum()
            .fillna(0)
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
        drop_cols = [
            'All Star', 'Player', 'Season Ending Year',
            'Prev All Stars', 'Conference_East',
            'PosGroup_Backcourt', 'PosGroup_Frontcourt'
        ]
       
        def split_xy(df):
            X = df.drop(columns=drop_cols + ['Conf_ID', 'Pos_ID', 'Season_ID'])
            y = df['All Star']
            return X, y, df['Conf_ID'], df['Pos_ID'], df['Season_ID']

        X_train, y_train, c_train, p_train, s_train = split_xy(train_df)
        X_val, y_val, c_val, p_val, s_val = split_xy(val_df)
        X_test, y_test, c_test, p_test, s_test = split_xy(test_df)

        X_train = X_train.fillna(X_train.median())
        X_val   = X_val.fillna(X_train.median())
        X_test  = X_test.fillna(X_train.median())

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        device = self.cfg.DEVICE

        return (
            torch.tensor(X_train, dtype=torch.float32).to(device),
            torch.tensor(y_train.values, dtype=torch.float32).to(device),
            torch.tensor(c_train.values, dtype=torch.long).to(device),
            torch.tensor(p_train.values, dtype=torch.long).to(device),
            torch.tensor(s_train.values, dtype=torch.long).to(device),

            torch.tensor(X_val, dtype=torch.float32).to(device),
            torch.tensor(y_val.values, dtype=torch.float32).to(device),
            torch.tensor(c_val.values, dtype=torch.long).to(device),
            torch.tensor(p_val.values, dtype=torch.long).to(device),
            torch.tensor(s_val.values, dtype=torch.long).to(device),

            torch.tensor(X_test, dtype=torch.float32).to(device),
            torch.tensor(y_test.values, dtype=torch.float32).to(device),
            torch.tensor(c_test.values, dtype=torch.long).to(device),
            torch.tensor(p_test.values, dtype=torch.long).to(device),
            torch.tensor(s_test.values, dtype=torch.long).to(device),
        )

class AllStarNN(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()

        self.conf_emb = nn.Embedding(2, 4)
        self.pos_emb = nn.Embedding(2, 4)
        self.season_emb = nn.Embedding(50, 8)
        self.ln1 = nn.LayerNorm(cfg.ATTN_DIM)
        self.ln2 = nn.LayerNorm(cfg.ATTN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(cfg.ATTN_DIM, cfg.ATTN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.ATTN_DIM, cfg.ATTN_DIM)
        )

        prev = input_dim + 16

        layers = []
        for h in cfg.HIDDEN_DIMS:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.DROPOUT)]
            prev = h

        self.mlp = nn.Sequential(*layers)
        self.to_attn = nn.Linear(prev, cfg.ATTN_DIM)

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.ATTN_DIM,
            num_heads=cfg.ATTN_HEADS,
            batch_first=True
        )

        self.starter_head = nn.Linear(cfg.ATTN_DIM, 1)
        self.reserve_head = nn.Linear(cfg.ATTN_DIM, 1)

    def forward_group(self, x, conf, pos, season):
        device = next(self.parameters()).device  

        x = x.to(device)

        conf = conf.to(device=device, dtype=torch.long)
        pos = pos.to(device=device, dtype=torch.long)
        season = season.to(device=device, dtype=torch.long)

        h = torch.cat([
            x,
            self.conf_emb(conf),
            self.pos_emb(pos),
            self.season_emb(season)
        ], dim=1)

        h = self.mlp(h)
        h = self.to_attn(h)

        h = h.unsqueeze(0)
        attn_out, _ = self.attn(h, h, h)
        h = self.ln1(h + attn_out)

        ff_out = self.ff(h)
        h = self.ln2(h + ff_out)

        h = h.squeeze(0)

        return self.starter_head(h), self.reserve_head(h)

def soft_selection(scores, k, temp):
    scores = scores.view(-1)
    probs = torch.softmax(scores / temp, dim=0)
    return k * probs  # sums to k

def selection_loss(scores, labels, k, temp):
    scores = scores.view(-1)
    labels = labels.view(-1)

    if labels.sum() == 0:
        return torch.zeros(1, device=scores.device)

    z = soft_selection(scores, k, temp)

    target = k * labels / (labels.sum() + 1e-8)

    return torch.mean((z - target) ** 2)

def pairwise_loss(scores, labels, margin=0.01):
    scores = scores.view(-1)
    labels = labels.view(-1)

    pos = scores[labels == 1]
    neg = scores[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return torch.zeros(1, device=scores.device)

    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return torch.relu(margin - diff).mean()

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss()

    def train(self, model,
              X_train, y_train, c_train, p_train, s_train,
              X_val, y_val, c_val, p_val, s_val,
              train_groups, val_groups):

        device = self.cfg.DEVICE
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.LR,
            weight_decay=self.cfg.WEIGHT_DECAY
        )

        for epoch in range(self.cfg.EPOCHS):
            model.train()
            total_loss = 0

            for idx in train_groups:
                Xg = X_train[idx]
                yg = y_train[idx]
                cg = c_train[idx]
                pg = p_train[idx]
                sg = s_train[idx]

                bc = (pg == 1)
                fc = (pg == 0)

                optimizer.zero_grad()

                s, r = model.forward_group(Xg, cg, pg, sg)

                loss = torch.zeros(1, device=Xg.device)

                if bc.sum() > 0:
                    loss += selection_loss(
                        s[bc], yg[bc], k=2, temp=self.cfg.TOPK_TEMP
                    )
                    loss += self.cfg.PAIRWISE_WEIGHT * pairwise_loss(s[bc], yg[bc])

                if fc.sum() > 0:
                    loss += selection_loss(
                        s[fc], yg[fc], k=3, temp=self.cfg.TOPK_TEMP
                    )
                    loss += self.cfg.PAIRWISE_WEIGHT * pairwise_loss(s[fc], yg[fc])

                loss += selection_loss(
                    r, yg, k=7, temp=self.cfg.TOPK_TEMP
                )
                loss += self.cfg.PAIRWISE_WEIGHT * pairwise_loss(r, yg)

                combined_scores = (s + r).squeeze()
                loss += self.cfg.BCE_WEIGHT * self.bce(combined_scores, yg)

                z_total = soft_selection(
                    combined_scores,
                    k=12,
                    temp=self.cfg.TOPK_TEMP
                )

                target_bc = 2 / 12
                actual_bc = z_total[pg == 1].sum() / 12

                pos_penalty = (actual_bc - target_bc) ** 2

                loss += 0.005 * pos_penalty  
                z_starters = torch.zeros_like(s.squeeze())

                if bc.sum() > 0:
                    z_starters[bc] = soft_selection(
                        s[bc], k=2, temp=self.cfg.TOPK_TEMP
                    )

                if fc.sum() > 0:
                    z_starters[fc] = soft_selection(
                        s[fc], k=3, temp=self.cfg.TOPK_TEMP
                    )

                z_reserves = soft_selection(
                    r.squeeze(), k=7, temp=self.cfg.TOPK_TEMP
                )

                overlap = (z_starters * z_reserves).sum()

                loss += 0.005 * overlap  
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            model.eval()
            scores = torch.zeros_like(y_val)

            with torch.no_grad():
                for idx in val_groups:
                    s_, r_ = model.forward_group(
                        X_val[idx], c_val[idx], p_val[idx], s_val[idx]
                    )
                    scores[idx] = (0.3 * s_ + 0.7 * r_).squeeze()

            auc = roc_auc_score(y_val.cpu(), scores.cpu())

            print(f"Epoch {epoch+1:03d} | Loss {total_loss:.2f} | Val AUC {auc:.4f}")

        return model

def evaluate(model, X_test, y_test, c_test, p_test, s_test, test_df, cfg):
    device = cfg.DEVICE
    model.eval()

    test_df = test_df.reset_index(drop=True)
    scores = torch.zeros(len(test_df), device=device)

    with torch.no_grad():
        groups = build_group_indices(test_df, device)

        for idx in groups:
            s_, r_ = model.forward_group(
                X_test[idx],
                c_test[idx],
                p_test[idx],
                s_test[idx]
            )
            scores[idx] = torch.sigmoid((0.3*s_ + 0.7*r_).squeeze())

    scores_np = scores.cpu().numpy()
    y_np = y_test.cpu().numpy()

    test_df['score'] = scores_np
    test_df['pred'] = 0

    for (season, conf), group in test_df.groupby(['Season Ending Year', 'Conference_East']):
        bc = group[group['PosGroup_Backcourt'] == 1]
        fc = group[group['PosGroup_Backcourt'] == 0]

        starters = pd.concat([
            bc.sort_values('score', ascending=False).head(2),
            fc.sort_values('score', ascending=False).head(3)
        ])

        remaining = group.drop(index=starters.index)
        reserves = remaining.sort_values('score', ascending=False).head(7)

        selected = pd.concat([starters, reserves])
        test_df.loc[selected.index, 'pred'] = 1

    y_pred_global = test_df['pred'].values


    print("\n=== FINAL RESULTS ===")
    print("AUC:", roc_auc_score(y_np, scores_np))
    print("Accuracy:", accuracy_score(y_np, y_pred_global))
    print("Precision:", precision_score(y_np, y_pred_global, zero_division=0))
    print("Recall:", recall_score(y_np, y_pred_global, zero_division=0))
    print("F1:", f1_score(y_np, y_pred_global, zero_division=0))

    print("\n=== Per-Season (Conference Split) ===")

    season_conf_metrics = []

    for (season, conf), group in test_df.groupby(['Season Ending Year', 'Conference_East']):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        score_g = group['score'].values

        auc = roc_auc_score(y_g, score_g)
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

    print("\n=== Per-Season (Overall) ===")

    season_metrics = []

    for season, group in test_df.groupby('Season Ending Year'):
        y_g = group['All Star'].values
        p_g = group['pred'].values
        score_g = group['score'].values

        auc = roc_auc_score(y_g, score_g)
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

    season_metrics = np.array(season_metrics)

    print("\n=== Avg Per-Season Metrics ===")
    print("AUC        :", season_metrics[:, 0].mean())
    print("Accuracy   :", season_metrics[:, 1].mean())
    print("Precision  :", season_metrics[:, 2].mean())
    print("Recall     :", season_metrics[:, 3].mean())
    print("F1         :", season_metrics[:, 4].mean())
    print("TopK Recall:", season_metrics[:, 5].mean())
    print("Top12 Acc  :", season_metrics[:, 6].mean())

def main():
    cfg = Config()
    set_seed(cfg.SEED)

    pipeline = DataPipeline(cfg)
    df = pipeline.load('source\\cleaned\\cleaned_data.csv')

    train_df, val_df, test_df = pipeline.split(df)

    data = pipeline.prepare(train_df, val_df, test_df)

    (X_train, y_train, c_train, p_train, s_train,
     X_val, y_val, c_val, p_val, s_val,
     X_test, y_test, c_test, p_test, s_test) = data

    train_groups = build_group_indices(train_df, cfg.DEVICE)
    val_groups   = build_group_indices(val_df, cfg.DEVICE)

    model = AllStarNN(X_train.shape[1], cfg)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    trainer = Trainer(cfg)

    model = trainer.train(
        model,
        X_train, y_train, c_train, p_train, s_train,
        X_val, y_val, c_val, p_val, s_val,
        train_groups, val_groups
    )
    evaluate(
    model,
    X_test,
    y_test,
    c_test,
    p_test,
    s_test,
    test_df,
    cfg
    )

if __name__ == "__main__":
    main()