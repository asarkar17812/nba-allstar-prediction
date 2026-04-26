from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np

# =========================
# Hyperparams
# =========================
C_VALUES = np.logspace(-2, 2, 6)
KERNEL = 'rbf' 
GAMMA_VALUES = ['scale', 0.1, 0.01, 0.001]
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
for C in C_VALUES:
    for gamma in GAMMA_VALUES:

        base_east = SVC(
            C=C,
            kernel=KERNEL,
            gamma=gamma,
            probability=False
        )

        base_west = SVC(
            C=C,
            kernel=KERNEL,
            gamma=gamma,
            probability=False
        )

        model_east = CalibratedClassifierCV(base_east, method='sigmoid', cv=3)
        model_west = CalibratedClassifierCV(base_west, method='sigmoid', cv=3)

        model_east.fit(X_train_scaled[train_east_idx], y_train[train_east_idx])
        model_west.fit(X_train_scaled[train_west_idx], y_train[train_west_idx])

        # probabilities
        probs = np.zeros(len(test_df))
        probs[test_east_idx] = model_east.predict_proba(X_test_scaled[test_east_idx])[:, 1]
        probs[test_west_idx] = model_west.predict_proba(X_test_scaled[test_west_idx])[:, 1]

        temp_df = test_df.copy()
        temp_df['prob'] = probs
        temp_df['pred'] = 0

        # structured selection
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
            "C": C,
            "gamma": gamma,
            "kernel": KERNEL,
            "AUC": auc,
            "precision": precision,
            "recall": recall
        })

        print(f"C={C:.4f} | gamma={gamma} | AUC={auc:.4f} | P={precision:.4f} | R={recall:.4f}")

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
        "C": C,
        "kernel": KERNEL,
        "AUC": auc,
        "precision": precision,
        "recall": recall
    })

    print(f"C={C:.4f} | AUC={auc:.4f} | P={precision:.4f} | R={recall:.4f}")

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