"""Generate the figures used in the README.

Outputs to assets/figures/:

  * model_comparison.png        — bar chart of global metrics across LR/SVM/NN
  * per_season_f1.png           — per-season F1 lines for each model
  * label_correction_impact.png — counts of missing All-Star labels per season
                                  that the diacritic fix recovers
  * tuning_curves.png           — log_reg CV score vs C, plus a per-arch SVM curve

Re-run after any pipeline / model change. Metric values are pasted in
from the most recent training runs — update the dicts below when you
rerun the models.

Run:
    python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ASSETS = Path("assets/figures")
ASSETS.mkdir(parents=True, exist_ok=True)

# Consistent palette across all figures.
COLORS = {
    "Logistic Regression": "#4C72B0",
    "Neural Network":      "#DD8452",
    "SVM":                 "#55A868",
    "k-NN":                "#8172B3",
    "Ensemble":            "#937860",
}

sns.set_theme(style="whitegrid", context="talk")


# ===========================================================
# 1. Global metric comparison bar chart
# ===========================================================
def make_model_comparison():
    """Hard-coded final test metrics. Update these whenever the runs
    change — they're the canonical numbers reported in the README.
    """
    data = pd.DataFrame([
        {"Model": "Logistic Regression", "AUC": 0.9883, "Accuracy": 0.9690,
         "Precision": 0.7917, "Recall": 0.7176, "F1": 0.7527},
        {"Model": "Neural Network",      "AUC": 0.9850, "Accuracy": 0.9678,
         "Precision": 0.7812, "Recall": 0.7075, "F1": 0.7429},
        {"Model": "SVM",                 "AUC": 0.9864, "Accuracy": 0.9678,
         "Precision": 0.7812, "Recall": 0.7075, "F1": 0.7426},
        {"Model": "k-NN",                "AUC": 0.9732, "Accuracy": 0.9567,
         "Precision": 0.6875, "Recall": 0.6226, "F1": 0.6535},
        {"Model": "Ensemble",            "AUC": 0.9881, "Accuracy": 0.9641,
         "Precision": 0.7500, "Recall": 0.6792, "F1": 0.7129},
    ])

    melted = data.melt(id_vars="Model", var_name="Metric", value_name="Value")

    fig, ax = plt.subplots(figsize=(14, 6.5))
    palette = [COLORS[m] for m in data["Model"]]
    sns.barplot(
        data=melted, x="Metric", y="Value", hue="Model",
        ax=ax, palette=palette,
    )
    ax.set_ylim(0.60, 1.04)
    ax.set_title("Test-set metrics (2022–2025) across all five models", fontsize=18, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=5, frameon=False)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(ASSETS / "model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ASSETS / 'model_comparison.png'}")


# ===========================================================
# 2. Per-season F1 lines
# ===========================================================
def make_per_season_f1():
    """Per-season F1 for each model — diagnostic for which years are
    hard. 2023 is consistently the toughest year across all three
    models, suggesting label noise / coach voting weirdness, not model
    quality.
    """
    seasons = [2022, 2023, 2024, 2025]
    per_season = pd.DataFrame({
        "Season": seasons,
        "Logistic Regression": [0.745, 0.706, 0.760, 0.800],
        "Neural Network":      [0.745, 0.667, 0.800, 0.760],
        "SVM":                 [0.706, 0.667, 0.800, 0.800],
        "k-NN":                [0.627, 0.588, 0.720, 0.640],
        "Ensemble":            [0.720, 0.640, 0.768, 0.720],
    })

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Stagger line styles so overlapping lines stay readable.
    styles = {
        "Logistic Regression": dict(linewidth=2.6, linestyle='-',  marker='o', markersize=10),
        "Neural Network":      dict(linewidth=2.6, linestyle='--', marker='s', markersize=9),
        "SVM":                 dict(linewidth=2.6, linestyle=':',  marker='^', markersize=10),
        "k-NN":                dict(linewidth=2.2, linestyle='-',  marker='D', markersize=7, alpha=0.7),
        "Ensemble":            dict(linewidth=2.2, linestyle='-.', marker='X', markersize=8, alpha=0.85),
    }
    for col in ["Logistic Regression", "Neural Network", "SVM", "k-NN", "Ensemble"]:
        ax.plot(per_season["Season"], per_season[col],
                color=COLORS[col], label=col, **styles[col])

    ax.set_xticks(seasons)
    ax.set_xlabel("Season Ending Year")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-season F1: where each model wins and loses", fontsize=16, pad=12)
    ax.set_ylim(0.50, 0.88)
    ax.legend(loc="lower right", frameon=True, ncol=2, fontsize=11)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(ASSETS / "per_season_f1.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ASSETS / 'per_season_f1.png'}")


# ===========================================================
# 3. Label correction: how many positives were missing per season
# ===========================================================
def make_label_correction_chart():
    """The 28 missing All-Star labels, by season. From the diff
    between the authoritative "All Star Seasons" sheet and the main
    'compiled data set' before the fix.
    """
    fixes = {
        1994: 1,   # B.J. Armstrong
        1992: 1,   # Magic Johnson
        2001: 1,   # Dikembe Mutombo
        2007: 1,   # LeBron James (string mismatch)
        2018: 2,   # Goran Dragić, Kristaps Porziņģis
        2005: 3,   # Manu Ginóbili, Vince Carter, Ž. Ilgauskas
        2003: 2,   # Peja Stojaković, Ž. Ilgauskas
        2002: 1,   # Peja Stojaković
        2004: 1,   # Peja Stojaković
        2011: 1,   # Manu Ginóbili
        1995: 1,   # Penny Hardaway
        1996: 1,   # Penny Hardaway
        1997: 1,   # Penny Hardaway
        1998: 1,   # Penny Hardaway
        2019: 2,   # Nikola Jokić, Nikola Vučević
        2020: 1,   # Jokić
        2021: 2,   # Jokić, Vučević
        2022: 1,   # Jokić
        2023: 1,   # Jokić
        2024: 1,   # Jokić
        2025: 1,   # Jokić
    }
    df = pd.DataFrame({
        "Season": sorted(fixes.keys()),
        "Missing labels": [fixes[s] for s in sorted(fixes.keys())],
    })

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(df["Season"], df["Missing labels"],
           color="#C44E52", alpha=0.85, edgecolor="white")
    ax.set_title("All-Star labels recovered from the authoritative sheet",
                 fontsize=16, pad=12)
    ax.set_xlabel("Season Ending Year")
    ax.set_ylabel("# positives reapplied")
    ax.set_yticks([0, 1, 2, 3])

    # Highlight the test window so it's clear how much the fix touches
    # the held-out years.
    ax.axvspan(2021.5, 2025.5, color="#DD8452", alpha=0.15,
               label="Test window (2022–2025)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig(ASSETS / "label_correction_impact.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ASSETS / 'label_correction_impact.png'}")


# ===========================================================
# 4. Hyperparameter tuning curves
# ===========================================================
def make_tuning_curves():
    """Two-panel figure: (left) log_reg CV score vs C with the
    optimum highlighted, (right) SVM linear CV curve. Values are
    pasted in from the most recent training run output.
    """
    log_reg_C = np.logspace(-3, 2, 12)
    log_reg_cv = [1.2316, 1.2215, 1.2343, 1.2407, 1.2492, 1.2428,
                  1.2427, 1.2405, 1.2362, 1.2320, 1.2320, 1.2320]
    best_idx = int(np.argmax(log_reg_cv))

    svm_C = np.array([0.001, 0.00464, 0.02154, 0.1, 0.46416, 2.15443, 10.0])
    svm_cv = [1.1805, 1.1894, 1.1835, 1.1860, 1.1741, 1.1746, 1.2387]
    svm_best_idx = int(np.argmax(svm_cv))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.semilogx(log_reg_C, log_reg_cv, marker="o",
                color=COLORS["Logistic Regression"], linewidth=2.3)
    ax.axvline(log_reg_C[best_idx], color="black",
               linestyle="--", alpha=0.6,
               label=f"best C = {log_reg_C[best_idx]:.3f}")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("CV score = TopK + 0.5 · AUC")
    ax.set_title("Logistic regression", fontsize=14)
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)

    ax = axes[1]
    ax.semilogx(svm_C, svm_cv, marker="o", color=COLORS["SVM"],
                linewidth=2.3, label="Linear")
    ax.axvline(svm_C[svm_best_idx], color="black",
               linestyle="--", alpha=0.6,
               label=f"best C = {svm_C[svm_best_idx]:.1f}")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("CV score")
    ax.set_title("SVM (linear kernel)", fontsize=14)
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)

    fig.suptitle("Hyperparameter sweep — CV objective vs regularisation",
                 fontsize=17, y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "tuning_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ASSETS / 'tuning_curves.png'}")


# ===========================================================
# 5. Pairwise model agreement (Jaccard on selections)
# ===========================================================
def make_model_agreement():
    """Jaccard agreement between the four base models' top-12 selections.

    Reads the per-model selections from
    assets/figures/test_predictions.csv (dumped by ensemble_model.py).
    """
    pred_path = ASSETS / "test_predictions.csv"
    if not pred_path.exists():
        print(f"  skip make_model_agreement — {pred_path} not present "
              f"(run ensemble_model.py first).")
        return

    df = pd.read_csv(pred_path)
    models = [("LR", "pred_lr"), ("SVM", "pred_svm"),
              ("k-NN", "pred_knn"), ("NN", "pred_nn")]

    n = len(models)
    mat = np.zeros((n, n))
    for i, (_, ci) in enumerate(models):
        for j, (_, cj) in enumerate(models):
            si = df[ci].astype(bool).values
            sj = df[cj].astype(bool).values
            inter = (si & sj).sum()
            union = (si | sj).sum()
            mat[i, j] = inter / union if union else 0.0

    labels = [m[0] for m in models]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        mat, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="YlGnBu", vmin=0.5, vmax=1.0,
        cbar_kws={"label": "Jaccard overlap"},
        ax=ax, linewidths=0.8, linecolor='white',
        annot_kws={"fontsize": 13},
    )
    ax.set_title("Model selection agreement\n"
                 "(Jaccard of top-12 picks per group, 2022–2025)",
                 fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "model_agreement.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ASSETS / 'model_agreement.png'}")


# ===========================================================
# 6. Precision-Recall scatter with F1 isolines
# ===========================================================
def make_pr_scatter():
    """Precision–recall positions with F1 isolines as background.

    Visualises why all four parametric models cluster (similar F1
    bands) and where k-NN drops off. The ensemble sits inside the
    cluster rather than on its outer edge — that's the geometric
    version of "blending didn't beat the best single model."
    """
    # SVM and NN are at the same precision/recall point; we plot one
    # marker for them and label both names.
    points = [
        ("Logistic Regression", 0.7917, 0.7176, ( 0.004,  0.004)),
        ("SVM / NN",            0.7812, 0.7075, ( 0.005, -0.014)),
        ("k-NN",                0.6875, 0.6226, ( 0.005,  0.005)),
        ("Ensemble",            0.7500, 0.6792, ( 0.005, -0.014)),
    ]
    point_colors = {
        "Logistic Regression": COLORS["Logistic Regression"],
        "SVM / NN":            "#6FB078",   # blend of SVM green + NN orange
        "k-NN":                COLORS["k-NN"],
        "Ensemble":            COLORS["Ensemble"],
    }

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # F1 isolines as contour background. We plot one contour line at
    # each level and manually annotate it so the labels are guaranteed
    # to appear inside the plotting window.
    p_grid, r_grid = np.meshgrid(np.linspace(0.5, 1.0, 300),
                                 np.linspace(0.5, 1.0, 300))
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_grid = 2 * p_grid * r_grid / (p_grid + r_grid + 1e-12)
    levels = [0.60, 0.65, 0.70, 0.75, 0.80]
    ax.contour(p_grid, r_grid, f1_grid, levels=levels,
               colors='grey', linestyles='--', alpha=0.55, linewidths=1.4)

    # Manual contour labels placed where the line crosses a fixed x.
    # F1 = 2*p*r/(p+r) → solve for r given p: r = p*F1 / (2p - F1)
    for f1 in levels:
        p_loc = 0.77
        denom = 2 * p_loc - f1
        if denom <= 0:
            continue
        r_loc = p_loc * f1 / denom
        if 0.59 < r_loc < 0.76:
            ax.text(p_loc, r_loc, f"F1={f1:.2f}", fontsize=9, color='grey',
                    rotation=-40, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.85))

    for name, prec, rec, (dx, dy) in points:
        ax.scatter(prec, rec, s=210, color=point_colors[name],
                   edgecolors='white', linewidths=2.0, zorder=5)
        ax.annotate(name, (prec, rec), xytext=(prec + dx, rec + dy),
                    fontsize=11, fontweight='bold', zorder=6)

    ax.set_xlim(0.65, 0.83)
    ax.set_ylim(0.59, 0.76)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision vs Recall on the 2022–2025 test set\n"
                 "(dashed lines are constant-F1 contours)",
                 fontsize=14, pad=12)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(ASSETS / "pr_scatter.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ASSETS / 'pr_scatter.png'}")


# ===========================================================
# 7. All-Stars caught by each model (overlap bar chart)
# ===========================================================
def make_catch_chart():
    """How many All-Stars each model caught vs. the ceiling.

    Reads per-model selections from assets/figures/test_predictions.csv.
    Shows two segments per model: All-Stars caught by *all four* models
    (the easy picks the field agrees on) vs caught by this model
    *uniquely or in disagreement*. Plus the always-missed bar that
    nobody catches — that's the irreducible noise from coach voting.
    """
    pred_path = ASSETS / "test_predictions.csv"
    if not pred_path.exists():
        print(f"  skip make_catch_chart — {pred_path} not present.")
        return

    df = pd.read_csv(pred_path)
    truth = (df['All Star'] == 1).values
    cols = {'LR': 'pred_lr', 'SVM': 'pred_svm',
            'k-NN': 'pred_knn', 'NN': 'pred_nn'}
    sels = {name: df[col].astype(bool).values for name, col in cols.items()}

    all_four = np.logical_and.reduce(list(sels.values()))
    caught_by_all = (all_four & truth).sum()
    total_pos = truth.sum()

    none_caught = ~np.logical_or.reduce(list(sels.values()))
    missed_by_all = (none_caught & truth).sum()

    names = list(sels.keys())
    extra = []
    for n in names:
        caught = (sels[n] & truth).sum()
        extra.append(caught - caught_by_all)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    width = 0.65

    ax.bar(x, [caught_by_all] * len(names), width,
           color='#888', label=f"Caught by all four models ({caught_by_all})",
           edgecolor='white')
    ax.bar(x, extra, width, bottom=[caught_by_all] * len(names),
           color=[COLORS[{'LR': 'Logistic Regression', 'SVM': 'SVM',
                          'k-NN': 'k-NN', 'NN': 'Neural Network'}[n]]
                  for n in names],
           edgecolor='white')

    # Reference line for the always-reachable ceiling.
    ceiling = total_pos - missed_by_all
    ax.axhline(ceiling, color='#C44E52', linestyle='--', linewidth=2,
               label=f"Reachable ceiling: {ceiling}/{total_pos} "
                     f"({total_pos - ceiling} caught by no model)")
    ax.axhline(total_pos, color='black', linestyle=':', alpha=0.6,
               label=f"Total true All-Stars: {total_pos}")

    # Annotate each bar with its total and the per-model "extra" count.
    for i, n in enumerate(names):
        total = caught_by_all + extra[i]
        ax.text(i, total + 0.7, f"{total}", ha='center',
                fontsize=12, fontweight='bold')
        if extra[i] > 0:
            ax.text(i, caught_by_all + extra[i] / 2,
                    f"+{extra[i]}", ha='center', color='white',
                    fontsize=11, fontweight='bold')
        # Consensus label only on the first bar to avoid repetition.
        if i == 0:
            ax.text(i, caught_by_all / 2, f"{caught_by_all}",
                    ha='center', color='white',
                    fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, total_pos + 8)
    ax.set_ylabel("All-Stars correctly selected")
    ax.set_title(f"All-Stars caught by each model (test set, n={total_pos})",
                 fontsize=14, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, frameon=False, fontsize=10)
    ax.grid(True, axis='y', alpha=0.4)

    fig.tight_layout()
    fig.savefig(ASSETS / "catches_per_model.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ASSETS / 'catches_per_model.png'}")


if __name__ == "__main__":
    make_model_comparison()
    make_per_season_f1()
    make_label_correction_chart()
    make_tuning_curves()
    make_model_agreement()
    make_pr_scatter()
    make_catch_chart()
