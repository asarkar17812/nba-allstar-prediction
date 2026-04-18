import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Hyperparams
# =========================
SEED = 3
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

learning_rate = 0.00006
epochs = 70
patience = 20
auc_tol = 1e-4

# =========================
# LOAD
# =========================
df = pd.read_csv('source\\cleaned\\cleaned_data.csv')

# =========================
# FEATURE ENGINEERING
# =========================
df['Conf_Label'] = np.where(df['Conference_East'] == 1, 'East', 'West')

for col in ['PTS per game', 'AST per game', 'TRB per game']:
    df[f'{col}_rank_conf'] = df.groupby(
        ['Season Ending Year', 'Conf_Label']
    )[col].rank(pct=True)

    df[f'{col}_z_conf'] = df.groupby(
        ['Season Ending Year', 'Conf_Label']
    )[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

df['Team Win % rank'] = df.groupby(
    ['Season Ending Year', 'Conf_Label']
)['Team Win %'].rank(pct=True)

df['Team Rank'] = df.groupby(
    ['Season Ending Year', 'Conf_Label']
)['Team Win %'].rank(ascending=False)

df['Is_Top4_Team'] = (df['Team Rank'] <= 4).astype(int)

df['Usage'] = df['FGA per game'] + 0.44 * df['FTA per game']
df['Impact_on_winning'] = df['PTS per game'] * df['Team Win %']
df['Usage_on_good_team'] = df['Usage'] * df['Team Win %']

df['Availability'] = df['Games'] / df.groupby(
    ['Season Ending Year']
)['Games'].transform('max')

df['Crowding_PTS'] = df['PTS per game'] / df.groupby(
    ['Season Ending Year', 'Conf_Label']
)['PTS per game'].transform('sum')

df['Crowding_Usage'] = df['Usage'] / df.groupby(
    ['Season Ending Year', 'Conf_Label']
)['Usage'].transform('sum')

df['Starter_Indicator'] = df['Games Started'] / (df['Games'] + 1e-6)

df = df.sort_values(['Player', 'Season Ending Year'])

df['AllStar_Weighted_History'] = (
    df.groupby('Player')['All Star']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1)
               .apply(lambda v: np.dot(v, np.linspace(1, 2, len(v))) / len(v)))
).fillna(0)

df['Was_AllStar_Last_Year'] = (
    df.groupby('Player')['All Star'].shift(1).fillna(0)
)

# =========================
# SPLIT
# =========================
train_df = df[df['Season Ending Year'] <= 2015].copy()
test_df  = df[df['Season Ending Year'] > 2015].copy()

drop_cols = ['All Star', 'Player', 'Season Ending Year', 'Prev All Stars', 'Conf_Label']

X_train = train_df.drop(columns=drop_cols)
y_train = train_df['All Star']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['All Star']

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# convert once
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).to(device)

# =========================
# MODEL
# =========================
class AllStarNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.starter_head = nn.Linear(32, 1)
        self.reserve_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.starter_head(h), self.reserve_head(h)

model = AllStarNN(X_train.shape[1]).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-3
)

# =========================
# LISTWISE LOSS
# =========================
def topk_listwise_loss(scores, labels, k):
    scores = scores.view(-1)
    labels = labels.view(-1)

    if labels.sum() == 0:
        return torch.tensor(0.0, device=device)

    scores = scores - scores.max()
    probs = torch.softmax(scores, dim=0)

    pos_probs = probs[labels == 1]
    k_eff = min(k, len(pos_probs))

    if k_eff == 0:
        return torch.tensor(0.0, device=device)

    topk_vals, _ = torch.topk(pos_probs, k_eff)
    return -torch.log(topk_vals.sum() + 1e-8)

# =========================
# TRAINING
# =========================
train_df = train_df.reset_index(drop=True)
groups = train_df.groupby(['Season Ending Year', 'Conference_East']).groups

best_auc = -np.inf
best_state = None
counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for idx in groups.values():
        idx = list(idx)

        X_group = X_train_tensor[idx]
        y_group = y_train_tensor[idx]

        if y_group.sum() == 0:
            continue

        optimizer.zero_grad()

        starter_scores, reserve_scores = model(X_group)

        pos_group = train_df.iloc[idx]

        bc_mask = torch.tensor(pos_group['PosGroup_Backcourt'].values, dtype=torch.bool).to(device)
        fc_mask = torch.tensor(pos_group['PosGroup_Frontcourt'].values, dtype=torch.bool).to(device)

        loss = torch.tensor(0.0, device=device)

        if bc_mask.sum() > 2:
            loss += topk_listwise_loss(starter_scores[bc_mask], y_group[bc_mask], 2)

        if fc_mask.sum() > 3:
            loss += topk_listwise_loss(starter_scores[fc_mask], y_group[fc_mask], 3)

        loss += topk_listwise_loss(reserve_scores, y_group, 7)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # evaluation
    model.eval()
    with torch.no_grad():
        s, r = model(X_test_tensor)
        probs = torch.sigmoid(0.2*s + 0.8*r).cpu().numpy()
        auc = roc_auc_score(y_test, probs)

    print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f} | AUC: {auc:.4f}")

    # stable early stopping
    if auc > best_auc + auc_tol:
        best_auc = auc
        best_state = model.state_dict()
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

# restore best model
model.load_state_dict(best_state)

# =========================
# INFERENCE
# =========================
model.eval()
with torch.no_grad():
    s, r = model(X_test_tensor)
    starter_prob = torch.sigmoid(s).cpu().numpy().flatten()
    reserve_prob = torch.sigmoid(r).cpu().numpy().flatten()

combined_prob = 0.2 * starter_prob + 0.8 * reserve_prob

test_df = test_df.copy()
test_df['starter_prob'] = starter_prob
test_df['reserve_prob'] = reserve_prob
test_df['combined_prob'] = combined_prob
test_df['pred'] = 0

for (season, conf), group in test_df.groupby(['Season Ending Year', 'Conference_East']):
    bc = group[group['PosGroup_Backcourt'] == 1]
    fc = group[group['PosGroup_Frontcourt'] == 1]

    starters = pd.concat([
        bc.sort_values('starter_prob', ascending=False).head(2),
        fc.sort_values('starter_prob', ascending=False).head(3)
    ])

    remaining = group.drop(index=starters.index)
    reserves = remaining.sort_values('reserve_prob', ascending=False).head(7)

    selected = pd.concat([starters, reserves])
    test_df.loc[selected.index, 'pred'] = 1

y_pred = test_df['pred'].values
y_true = test_df['All Star'].values

print("\nAUC:", roc_auc_score(y_true, combined_prob))
print("Top-12 Precision:", precision_score(y_true, y_pred))
print("Top-12 Recall:", recall_score(y_true, y_pred))