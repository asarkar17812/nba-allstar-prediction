from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import pandas as pd
import numpy as np

# =========================
# Hyperparams
# =========================
K_VALUES = range(5,12)
METRIC = 'euclidean'

DIST_POWERS = [1.0, 1.5, 2.0, 3.0]
TEMPS = [0.5]
EPS = 1e-6

# =========================
# LOAD
# =========================
df = pd.read_csv('source\\cleaned\\cleaned_data.csv')

# =========================
# FEATURE ENGINEERING (MATCH NN)
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

# =========================
# CONFERENCE MASKS
# =========================
train_east_idx = train_df['Conference_East'] == 1
train_west_idx = train_df['Conference_East'] == 0

test_east_idx = test_df['Conference_East'] == 1
test_west_idx = test_df['Conference_East'] == 0

# =========================
# RESULTS STORAGE
# =========================
results = []

# =========================
# SWEEP
# =========================
for power in DIST_POWERS:
    for temp in TEMPS:

        def custom_weights(distances):
            d = distances / temp
            return 1.0 / (np.power(d + EPS, power))

        for K in K_VALUES:

            knn_east = KNeighborsClassifier(
                n_neighbors=K,
                weights=custom_weights,
                metric=METRIC
            )

            knn_west = KNeighborsClassifier(
                n_neighbors=K,
                weights=custom_weights,
                metric=METRIC
            )

            knn_east.fit(X_train_scaled[train_east_idx], y_train[train_east_idx])
            knn_west.fit(X_train_scaled[train_west_idx], y_train[train_west_idx])

            # probabilities
            probs = np.zeros(len(test_df))
            probs[test_east_idx] = knn_east.predict_proba(X_test_scaled[test_east_idx])[:, 1]
            probs[test_west_idx] = knn_west.predict_proba(X_test_scaled[test_west_idx])[:, 1]

            temp_df = test_df.copy()
            temp_df['prob'] = probs
            temp_df['pred'] = 0

            # =========================
            # STRUCTURED SELECTION
            # =========================
            for (season, conf), group in temp_df.groupby(['Season Ending Year', 'Conference_East']):

                bc = group[group['PosGroup_Backcourt'] == 1]
                fc = group[group['PosGroup_Frontcourt'] == 1]

                starters = pd.concat([
                    bc.sort_values('prob', ascending=False).head(2),
                    fc.sort_values('prob', ascending=False).head(3)
                ])

                remaining = group.drop(index=starters.index)
                reserves = remaining.sort_values('prob', ascending=False).head(7)

                selected = pd.concat([starters, reserves])
                temp_df.loc[selected.index, 'pred'] = 1

            y_pred = temp_df['pred'].values
            y_true = y_test.values

            auc = roc_auc_score(y_true, probs)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)

            results.append({
                "K": K,
                "power": power,
                "temp": temp,
                "AUC": auc,
                "precision": precision,
                "recall": recall
            })

            print(f"K={K:2d} | p={power:.1f} | t={temp:.1f} | AUC={auc:.4f} | P={precision:.4f} | R={recall:.4f}")

# =========================
# BEST CONFIG
# =========================
results_df = pd.DataFrame(results)

best = results_df.sort_values(
    by=["precision", "recall"],
    ascending=False
).iloc[0]

print("\nBest config:")
print(best)