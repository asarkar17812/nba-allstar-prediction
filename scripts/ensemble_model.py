import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

import torch
import torch.nn as nn

# =========================
# HYPERPARAMS (FIXED BEST)
# =========================
# Logistic
LOG_C = 0.046416

# SVM
SVM_C = 15.848932
SVM_GAMMA = 0.001

# kNN
KNN_K = 11
KNN_POWER = 3.0
KNN_TEMP = 0.5
EPS = 1e-6

# NN
NN_LR = 0.00006
NN_EPOCHS = 70

# Ensemble weights (tune later if needed)
W_NN = 0.35
W_LOG = 0.25
W_SVM = 0.20
W_KNN = 0.20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

df['Usage'] = df['FGA per game'] + 0.44 * df['FTA per game']
df['Impact_on_winning'] = df['PTS per game'] * df['Team Win %']

df = df.sort_values(['Player', 'Season Ending Year'])

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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# masks
train_east = train_df['Conference_East'] == 1
train_west = ~train_east

test_east = test_df['Conference_East'] == 1
test_west = ~test_east

# =========================
# PROB NORMALIZATION
# =========================
def normalize_probs(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (p - p.min()) / (p.max() - p.min() + 1e-8)

# =========================
# 1. LOGISTIC
# =========================
log_e = LogisticRegression(C=LOG_C, max_iter=1000)
log_w = LogisticRegression(C=LOG_C, max_iter=1000)

log_e.fit(X_train_scaled[train_east], y_train[train_east])
log_w.fit(X_train_scaled[train_west], y_train[train_west])

log_prob = np.zeros(len(test_df))
log_prob[test_east] = log_e.predict_proba(X_test_scaled[test_east])[:, 1]
log_prob[test_west] = log_w.predict_proba(X_test_scaled[test_west])[:, 1]
log_prob = normalize_probs(log_prob)

# =========================
# 2. SVM
# =========================
svm_e = CalibratedClassifierCV(
    SVC(C=SVM_C, kernel='rbf', gamma=SVM_GAMMA),
    method='sigmoid',
    cv=3
)

svm_w = CalibratedClassifierCV(
    SVC(C=SVM_C, kernel='rbf', gamma=SVM_GAMMA),
    method='sigmoid',
    cv=3
)

svm_e.fit(X_train_scaled[train_east], y_train[train_east])
svm_w.fit(X_train_scaled[train_west], y_train[train_west])

svm_prob = np.zeros(len(test_df))
svm_prob[test_east] = svm_e.predict_proba(X_test_scaled[test_east])[:, 1]
svm_prob[test_west] = svm_w.predict_proba(X_test_scaled[test_west])[:, 1]
svm_prob = normalize_probs(svm_prob)

# =========================
# 3. kNN
# =========================
def knn_weights(d):
    d = d / KNN_TEMP
    return 1.0 / (np.power(d + EPS, KNN_POWER))

knn_e = KNeighborsClassifier(
    n_neighbors=KNN_K,
    weights=knn_weights,
    metric='euclidean'
)

knn_w = KNeighborsClassifier(
    n_neighbors=KNN_K,
    weights=knn_weights,
    metric='euclidean'
)

knn_e.fit(X_train_scaled[train_east], y_train[train_east])
knn_w.fit(X_train_scaled[train_west], y_train[train_west])

knn_prob = np.zeros(len(test_df))
knn_prob[test_east] = knn_e.predict_proba(X_test_scaled[test_east])[:, 1]
knn_prob[test_west] = knn_w.predict_proba(X_test_scaled[test_west])[:, 1]
knn_prob = normalize_probs(knn_prob)

# =========================
# 4. NN
# =========================
class NN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        return self.out(self.net(x))

model = NN(X_train_scaled.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=NN_LR)
loss_fn = nn.BCEWithLogitsLoss()

X_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_t = torch.tensor(y_train.values, dtype=torch.float32).to(device)

for _ in range(NN_EPOCHS):
    opt.zero_grad()
    out = model(X_t).squeeze()
    loss = loss_fn(out, y_t)
    loss.backward()
    opt.step()

with torch.no_grad():
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    nn_prob = torch.sigmoid(model(X_test_t)).cpu().numpy()

nn_prob = normalize_probs(nn_prob)

# =========================
# ENSEMBLE
# =========================
final_prob = (
    W_NN * nn_prob +
    W_LOG * log_prob +
    W_SVM * svm_prob +
    W_KNN * knn_prob
)

# =========================
# STRUCTURED SELECTION
# =========================
test_df = test_df.copy()
test_df['prob'] = final_prob
test_df['pred'] = 0

for (season, conf), group in test_df.groupby(['Season Ending Year', 'Conference_East']):

    bc = group[group['PosGroup_Backcourt'] == 1]
    fc = group[group['PosGroup_Frontcourt'] == 1]

    starters = pd.concat([
        bc.sort_values('prob', ascending=False).head(2),
        fc.sort_values('prob', ascending=False).head(3)
    ])

    remaining = group.drop(index=starters.index)
    reserves = remaining.sort_values('prob', ascending=False).head(7)

    selected = pd.concat([starters, reserves])
    test_df.loc[selected.index, 'pred'] = 1

# =========================
# METRICS
# =========================
y_pred = test_df['pred'].values
y_true = y_test.values

print("\nENSEMBLE RESULTS")
print("AUC:", roc_auc_score(y_true, final_prob))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))