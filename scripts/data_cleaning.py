"""
NBA All-Star Prediction — data cleaning and preprocessing

Prepares a stable, fully numeric dataset from raw historical NBA data.

The pipeline is built around four invariants:
1. Franchise identity is normalized across relocations / renamings so that
   grouping by (Season, Conference) is meaningful.
2. Structural missingness (e.g. 3P% when 3PA = 0) is resolved before any
   statistical imputation so that distance-based imputation is not poisoned
   by deterministic zeros.
3. Imputation, normalization, and rank-derived features are computed
   strictly within a single season, so that information from future
   seasons cannot leak into past ones.
4. The output is fully numeric, one row per (Player, Season), with all
   categoricals one-hot encoded.

Sources:
| Team_Records |
| https://www.kaggle.com/datasets/boonpalipatana/nba-season-records-from-every-year |
| All Seasons (96–'23) — basketball-reference fill-ins |
| https://www.kaggle.com/datasets/justinas/nba-players-data |
| Team Abbreviations |
| https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores |
| Seasons 1950–2017 + height/weight |
| https://www.kaggle.com/datasets/drgilermo/nba-players-stats/data |
| Misc fills |
| https://www.sports-reference.com/stathead/basketball/player-season-finder.cgi |
| Historic All-Star list |
| https://en.wikipedia.org/wiki/List_of_NBA_All-Stars |

Input:
    source/uncleaned/NBA ALL STAR DATA.xlsx

Output:
    source/cleaned/cleaned_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. LOAD
# =========================================================
SOURCE_XLSX = "source/uncleaned/NBA ALL STAR DATA.xlsx"

df_team_abbr = pd.read_excel(SOURCE_XLSX, sheet_name=7)
df_data = pd.read_excel(SOURCE_XLSX, sheet_name=1)

df_data = df_data.drop(columns=["Unnamed: 0"])
df_data.columns = df_data.columns.str.strip()
df_data["Team"] = df_data["Team"].str.strip()
df_data = df_data.dropna(subset=["Team"])


# =========================================================
# 1b. AUTHORITATIVE ALL-STAR LABELS
# =========================================================
# The compiled-data sheet is missing All-Star labels for ~28 players,
# mostly international names that lost their diacritics during ingest
# (e.g. "Nikola Jokić" → "Nikola Jokic" misses the join). The
# "All Star Seasons" sheet is the authoritative source, taken straight
# from Wikipedia's All-Star history.
#
# We rebuild the (player, season) → All-Star mapping from that sheet,
# normalising names by stripping diacritics so the join matches the
# ASCII names in the main data. The corrected labels are reapplied
# before any modelling features are derived, so career-form lags
# (AllStar_Last_Year etc.) all see the right ground truth.
import unicodedata

def _strip_accents(s):
    if pd.isna(s):
        return s
    s = str(s)
    normalized = unicodedata.normalize("NFKD", s)
    return "".join(c for c in normalized if not unicodedata.combining(c))

_as_raw = pd.read_excel(SOURCE_XLSX, sheet_name="All Star Seasons")
# Header row is on iloc 0; the actual columns we want are at positions 4, 5, 7
_as = _as_raw.iloc[1:, [4, 5, 7]].copy()
_as.columns = ["Player", "Year", "IsAllStar"]
_as = _as.dropna(subset=["Player", "Year"])
_as["Year"] = pd.to_numeric(_as["Year"], errors="coerce")
_as = _as.dropna(subset=["Year"])
_as["Year"] = _as["Year"].astype(int)
_as["IsAllStar"] = pd.to_numeric(_as["IsAllStar"], errors="coerce").fillna(0).astype(int)
_as["PlayerKey"] = _as["Player"].apply(_strip_accents)

authoritative_allstars = set(
    (p, y) for p, y in zip(
        _as.loc[_as["IsAllStar"] == 1, "PlayerKey"],
        _as.loc[_as["IsAllStar"] == 1, "Year"],
    )
)

# Reapply labels. We accept that the join can also flip a 0 to a 1 for
# players we previously missed (this is the whole point), but we never
# flip a 1 to a 0 — the authoritative sheet is a superset of the truth
# we already had.
_main_key = df_data["Player"].apply(_strip_accents)
_authoritative_match = pd.Series(
    [(p, y) in authoritative_allstars
     for p, y in zip(_main_key, df_data["Season Ending Year"].astype("Int64"))],
    index=df_data.index,
)
df_data["All Star"] = ((df_data["All Star"] == 1) | _authoritative_match).astype(int)


# =========================================================
# 2. FRANCHISE NORMALIZATION
# =========================================================
# Map every historical team name / abbreviation to a single canonical code.
# This keeps grouping by (Season, Conference) consistent across relocations
# (Seattle -> OKC, New Jersey -> BKN, Vancouver -> MEM, etc.).
team_to_code = {
    "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "New York Knicks": "NYK",
    "Philadelphia 76ers": "PHI", "Toronto Raptors": "TOR",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Detroit Pistons": "DET",
    "Indiana Pacers": "IND", "Milwaukee Bucks": "MIL",
    "Atlanta Hawks": "ATL", "Charlotte Hornets": "CHA", "Miami Heat": "MIA",
    "Orlando Magic": "ORL", "Washington Wizards": "WAS",

    "Denver Nuggets": "DEN", "Minnesota Timberwolves": "MIN",
    "Oklahoma City Thunder": "OKC", "Portland Trail Blazers": "POR",
    "Utah Jazz": "UTA", "Golden State Warriors": "GSW",
    "LA Clippers": "LAC", "LA Lakers": "LAL",
    "Los Angeles Lakers": "LAL", "Los Angeles Clippers": "LAC",
    "Phoenix Suns": "PHX", "Sacramento Kings": "SAC",
    "Dallas Mavericks": "DAL", "Houston Rockets": "HOU",
    "Memphis Grizzlies": "MEM", "New Orleans Pelicans": "NOP",
    "San Antonio Spurs": "SAS",

    "New Jersey Nets": "BKN",
    "Charlotte Bobcats": "CHA",
    "New Orleans Hornets": "NOP",
    "Seattle SuperSonics": "OKC",
    "Vancouver Grizzlies": "MEM",
    "Washington Bullets": "WAS",
}

abbrev_map = {
    "BOS": "BOS", "NYK": "NYK", "PHI": "PHI", "TOR": "TOR", "CHI": "CHI",
    "CLE": "CLE", "DET": "DET", "IND": "IND", "MIL": "MIL",
    "ATL": "ATL", "CHA": "CHA", "MIA": "MIA", "ORL": "ORL", "WAS": "WAS",
    "WSB": "WAS",

    "DEN": "DEN", "MIN": "MIN", "OKC": "OKC", "POR": "POR", "UTA": "UTA",
    "GSW": "GSW", "LAC": "LAC", "LAL": "LAL",
    "PHX": "PHX", "PHO": "PHX",
    "SAC": "SAC", "KCK": "SAC",
    "DAL": "DAL", "HOU": "HOU", "MEM": "MEM", "VAN": "MEM",
    "NOP": "NOP", "NOH": "NOP", "SAS": "SAS",

    "SEA": "OKC", "SDC": "LAC", "NJN": "BKN", "BRK": "BKN",
    "CHO": "CHA", "CHH": "CHA",
}

df_data["Team_Code"] = df_data["Team"].map(team_to_code)
df_data["Team_Code"] = df_data["Team_Code"].fillna(df_data["Team"])
df_data["Team_Code"] = df_data["Team_Code"].replace(abbrev_map)


# =========================================================
# 3. CONFERENCE ASSIGNMENT
# =========================================================
# All-Star selection is conference-constrained, so conference is the second
# half of every group key downstream. Deriving it deterministically from
# the franchise code avoids the gaps you get if you trust the source labels.
east_codes = {
    "BOS", "BKN", "NYK", "PHI", "TOR", "CHI", "CLE", "DET", "IND", "MIL",
    "ATL", "CHA", "MIA", "ORL", "WAS",
}

west_codes = {
    "DEN", "MIN", "OKC", "POR", "UTA", "GSW", "LAC", "LAL",
    "PHX", "SAC", "DAL", "HOU", "MEM", "NOP", "SAS",
}

def map_conference(code):
    if code in east_codes:
        return "East"
    if code in west_codes:
        return "West"
    return np.nan

df_data["Conference"] = df_data["Team_Code"].apply(map_conference)

missing = df_data[df_data["Conference"].isna()]["Team"].unique()
if len(missing) > 0:
    raise ValueError(f"Unmapped teams: {missing}")


# =========================================================
# 4. STRUCTURAL MISSINGNESS
# =========================================================
# A NaN in 3P% with 0 3PA isn't missing data — it's a zero denominator.
# Resolving these before imputation prevents KNN from inventing values
# for what is actually a known constant.
df_data.loc[df_data["3PA per game"] == 0, "3P%"] = 0
df_data.loc[df_data["FTA per game"] == 0, "FT%"] = 0


# =========================================================
# 5. HEIGHT NORMALIZATION
# =========================================================
def height_to_inches(h):
    try:
        if pd.isna(h):
            return np.nan
        feet, inches = str(h).replace("'", "-").split("-")
        return int(feet) * 12 + int(inches)
    except Exception:
        return np.nan

df_data["Height"] = df_data["Height"].apply(height_to_inches)


# =========================================================
# 6. DROP CORE-MISSING ROWS
# =========================================================
# Drop only when the core signal columns are missing. Everything else is
# kept and resolved through imputation so we don't bleed data.
df_data = df_data.dropna(
    subset=["Games", "Minutes per game", "PTS per game", "All Star"]
)

df_data["Prev All Stars"] = df_data["Prev All Stars"].fillna(0)
df_data["Games Started"] = df_data["Games Started"].fillna(0)
df_data["# Team Games"] = df_data["# Team Games"].fillna(
    df_data["# Team Games"].median()
)


# =========================================================
# 7. POSITION GROUPING
# =========================================================
# All-Star ballots are split into backcourt (2 starters) and frontcourt
# (3 starters), so we add that grouping as a separate categorical.
def map_pos_group(pos):
    if pos in ["PG", "SG"]:
        return "Backcourt"
    if pos in ["SF", "PF", "C"]:
        return "Frontcourt"
    return "Other"

df_data["PosGroup"] = df_data["Pos"].apply(map_pos_group)


# =========================================================
# 8. AVAILABILITY FILTER
# =========================================================
# Players with sub-replacement availability add noise to season-wise
# scaling and imputation without being plausible All-Star candidates.
# We keep them only if they were actually selected (we still need their
# labels), and drop the rest.
df_data = df_data[
    ((df_data["Games"] >= 20) & (df_data["Minutes per game"] >= 10)) |
    (df_data["All Star"] == 1)
].copy()


# =========================================================
# 8b. TRADE-SEASON DEDUPLICATION
# =========================================================
# Mid-season trades produce multiple rows for one (Player, Season), one
# per team. Downstream this is corrosive:
#   * lagged "previous-season" features become ambiguous,
#   * structured selection can double-count a player within a group,
#   * the row order of the duplicates is non-deterministic, so model
#     training is silently non-reproducible.
# Keep the row with the most games played — i.e. the team the player
# spent the most of the season on. The All-Star label is identical
# across the duplicate rows, so no positives are lost.
df_data = (
    df_data
    .sort_values(["Player", "Season Ending Year", "Games"],
                 ascending=[True, True, False])
    .drop_duplicates(subset=["Player", "Season Ending Year"], keep="first")
    .reset_index(drop=True)
)


# =========================================================
# 9. SEASON-WISE KNN IMPUTATION
# =========================================================
# Different eras play at different paces and scoring levels. Imputing
# globally drags older players toward the modern mean (and vice versa),
# which directly distorts the relative-performance signal the model needs.
# Standardizing within a season, imputing in standardized space, then
# inverting keeps imputation aware of era while remaining leak-free.
impute_cols = [
    'Age', 'Games', 'Minutes per game',
    'FGA per game', '2PA per game', '3PA per game', 'FTA per game',
    'ORB per game', 'DRB per game', 'TRB per game',
    'AST per game', 'STL per game', 'BLK per game',
    'TOV per game', 'PF per game', 'PTS per game',
    'FG%', '2P%', '3P%', 'FT%', 'eFG%',
    'Team Win %', 'Height', 'Weight',
]

processed = []
for season, group in df_data.groupby("Season Ending Year"):
    group = group.copy()

    for col in impute_cols:
        if group[col].isna().all():
            group[col] = 0

    scaler = StandardScaler()
    scaled = scaler.fit_transform(group[impute_cols])

    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(scaled)

    group[impute_cols] = scaler.inverse_transform(imputed)
    processed.append(group)

df_data = pd.concat(processed, axis=0)


# =========================================================
# 10. DERIVED EFFICIENCY FEATURES (LEAK-FREE)
# =========================================================
# These are deterministic transforms of within-row stats — no cross-row
# information, so no risk of season-to-season leakage.

# Approximate true-shooting percentage. Captures volume-adjusted scoring
# efficiency that raw FG% misses (a 50% 3PT shooter is wildly more
# valuable than a 50% 2PT shooter).
ts_denom = 2 * (df_data["FGA per game"] + 0.44 * df_data["FTA per game"])
df_data["TS%"] = np.where(
    ts_denom > 1e-6,
    df_data["PTS per game"] / ts_denom,
    0.0,
)

# Composite box-score "stuffer" index: simple weighted sum that mirrors
# the popular per-game volume composites (NBA Efficiency / GameScore).
# Useful as a single dense signal alongside the raw stats.
df_data["BoxLoad"] = (
    df_data["PTS per game"]
    + df_data["TRB per game"]
    + df_data["AST per game"]
    + df_data["STL per game"]
    + df_data["BLK per game"]
    - df_data["TOV per game"]
)

# Age curve indicator. All-Stars cluster heavily in the 24–30 prime band.
df_data["PrimeAge"] = (
    (df_data["Age"] >= 24) & (df_data["Age"] <= 30)
).astype(int)

# Games-played fraction. Availability matters: a 70-game superstar will
# almost always beat a 40-game one for an All-Star slot.
df_data["GamesFrac"] = df_data["Games"] / df_data["# Team Games"].clip(lower=1)


# =========================================================
# 11. CATEGORICAL ENCODING
# =========================================================
df_data = pd.get_dummies(df_data, columns=["Pos", "PosGroup", "Conference"])
df_data = df_data.drop(columns=["Team", "Team_Code"])


# =========================================================
# 12. SEASON-WISE STAT NORMALIZATION
# =========================================================
# Raw per-game volume stats are not comparable across eras (1996 plodded;
# 2019 ran a sprint). Z-scoring within each season removes the era-level
# shift while preserving relative standing — exactly what All-Star
# selection actually depends on.
stat_cols = [
    'Minutes per game', 'FGA per game', '2PA per game', '3PA per game',
    'FTA per game', 'ORB per game', 'DRB per game', 'TRB per game',
    'AST per game', 'STL per game', 'BLK per game', 'TOV per game',
    'PF per game', 'PTS per game', 'BoxLoad',
]

df_data[stat_cols] = df_data.groupby(
    "Season Ending Year"
)[stat_cols].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x
)


# =========================================================
# 13. FINAL FALLBACK IMPUTATION
# =========================================================
# Edge cases left over from season groups that were entirely NaN for a
# column. Median is a safer default than mean for the long-tailed
# distributions we have.
for col in df_data.columns:
    if df_data[col].isna().sum() > 0:
        if col == "Prev All Stars":
            df_data[col] = df_data[col].fillna(0)
        else:
            df_data[col] = df_data[col].fillna(df_data[col].median())

assert df_data.isna().sum().sum() == 0


# =========================================================
# 14. LOCKOUT REMOVAL
# =========================================================
# The 1999 All-Star Game was cancelled because of the 1998–99 NBA lockout,
# so there are no positive labels that season. Including it would just
# add noise to the negative class.
df_data = df_data[df_data["Season Ending Year"] != 1999]


# =========================================================
# 15. WRITE
# =========================================================
print("Final shape:", df_data.shape)
print(df_data.describe())

df_data.to_csv("source/cleaned/cleaned_data.csv", index=False)
