import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
"""
NBA All-Star Prediction — data cleaning and preprocessing

This script prepares a dataset from raw historical NBA data
from the following sources: 

| Team_Records |
| https://www.kaggle.com/datasets/boonpalipatana/nba-season-records-from-every-year |

| All Seasons(96-'23): General data to fill in missing entries (basketballrefernece.com) |
| https://www.kaggle.com/datasets/justinas/nba-players-data | 

| Team Abbreviations |
| https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores?select=TeamHistories.csv |

| Seasons from 1950-2017 + Player height/weight data |
| https://www.kaggle.com/datasets/drgilermo/nba-players-stats/data?select=player_data.csv |

| General data to fill missing entries | 
| https://www.sports-reference.com/stathead/basketball/player-season-finder.cgi? |

| Historic list of NBA All Stars |
| https://en.wikipedia.org/wiki/List_of_NBA_All-Stars |

The pipeline is designed to:
- remove inconsistencies across eras (team names, pace, etc.)
- preserve relative performance within each season
- avoid leakage when imputing or normalizing
- produce a stable, fully numeric dataset for modeling

Input: 
    source/uncleaned/NBA ALL STAR DATA.xlsx

Output:
    source/cleaned/cleaned_data.csv
"""

# We begin by loading both the compliled raw player data and the team reference sheet.
# Only the player data is used downstream, but the structure is preserved
# in case team-level joins are needed later
df_team_abbr = pd.read_excel(
    "source/uncleaned/NBA ALL STAR DATA.xlsx", sheet_name=7
)
df_data = pd.read_excel(
    "source/uncleaned/NBA ALL STAR DATA.xlsx", sheet_name=1
)

# Basic cleanup removes formatting artifacts from Excel and ensures that
# team identifiers are consistent before normalization
df_data = df_data.drop(columns=["Unnamed: 0"])
df_data.columns = df_data.columns.str.strip()
df_data["Team"] = df_data["Team"].str.strip()
df_data = df_data.dropna(subset=["Team"])

# Team identities change over time (relocations, renaming, abbreviations, etc.)
# We map everything to a canonical franchise code so that historical data
# is comparable and grouping operations are useable
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
    "Washington Bullets": "WAS"
}

abbrev_map = {
    "BOS":"BOS","NYK":"NYK","PHI":"PHI","TOR":"TOR","CHI":"CHI",
    "CLE":"CLE","DET":"DET","IND":"IND","MIL":"MIL",
    "ATL":"ATL","CHA":"CHA","MIA":"MIA","ORL":"ORL","WAS":"WAS",
    "WSB":"WAS",

    "DEN":"DEN","MIN":"MIN","OKC":"OKC","POR":"POR","UTA":"UTA",
    "GSW":"GSW","LAC":"LAC","LAL":"LAL",
    "PHX":"PHX","PHO":"PHX",
    "SAC":"SAC","KCK":"SAC",
    "DAL":"DAL","HOU":"HOU","MEM":"MEM","VAN":"MEM",
    "NOP":"NOP","NOH":"NOP","SAS":"SAS",

    "SEA":"OKC","SDC":"LAC","NJN":"BKN","BRK":"BKN",
    "CHO":"CHA","CHH":"CHA"
}

df_data["Team_Code"] = df_data["Team"].map(team_to_code)
df_data["Team_Code"] = df_data["Team_Code"].fillna(df_data["Team"])
df_data["Team_Code"] = df_data["Team_Code"].replace(abbrev_map)

# Conference is not always explicitly encoded in historical data, but it's
# essential for modeling because All-Star selection is conference-constrained
# We assign it deterministically from the normalized team codes
east_codes = {
    "BOS","BKN","NYK","PHI","TOR","CHI","CLE","DET","IND","MIL",
    "ATL","CHA","MIA","ORL","WAS"
}

west_codes = {
    "DEN","MIN","OKC","POR","UTA","GSW","LAC","LAL",
    "PHX","SAC","DAL","HOU","MEM","NOP","SAS"
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

df_data = df_data.drop(columns=["Team_Code"])

# Some missing values are structural rather than random. ie,
# if a player attempts zero three-pointers, their percentage should be 0,
# not NaN. Fixing these early prevents distortion and errors during imputation
df_data.loc[df_data["3PA per game"] == 0, "3P%"] = 0
df_data.loc[df_data["FTA per game"] == 0, "FT%"] = 0

# Height is stored as a string (e.g., "6-7"). Converting to inches makes
# it usable as a numeric feature and consistent across rows and over time
def height_to_inches(h):
    try:
        if pd.isna(h):
            return np.nan
        feet, inches = str(h).replace("'", "-").split("-")
        return int(feet) * 12 + int(inches)
    except:
        return np.nan

df_data["Height"] = df_data["Height"].apply(height_to_inches)

# We only drop rows missing core signal variables. Everything else is
# preserved and handled through imputation to avoid unnecessary data loss
df_data = df_data.dropna(
    subset=["Games", "Minutes per game", "PTS per game", "All Star"]
)

df_data["Prev All Stars"] = df_data["Prev All Stars"].fillna(0)
df_data["Games Started"] = df_data["Games Started"].fillna(0)
df_data["# Team Games"] = df_data["# Team Games"].fillna(
    df_data["# Team Games"].median()
)

# Position is grouped into backcourt vs frontcourt to match how All-Star
# rosters are actually constructed (starter constraints depend on this)
def map_pos_group(pos):
    if pos in ["PG", "SG"]:
        return "Backcourt"
    if pos in ["SF", "PF", "C"]:
        return "Frontcourt"
    return "Other"

df_data["PosGroup"] = df_data["Pos"].apply(map_pos_group)

# Very low-minute or low-availability players introduce noise without
# contributing useful signal. We filter them out unless they were selected
df_data = df_data[
    ((df_data["Games"] >= 20) & (df_data["Minutes per game"] >= 10)) |
    (df_data["All Star"] == 1)
]

# Missing values are imputed within each season to avoid leakage across time
# Scaling before kNN ensures that distance-based imputation is meaningful
impute_cols = [
    'Age','Games','Minutes per game',
    'FGA per game','2PA per game','3PA per game','FTA per game',
    'ORB per game','DRB per game','TRB per game',
    'AST per game','STL per game','BLK per game',
    'TOV per game','PF per game','PTS per game',
    'FG%','2P%','3P%','FT%','eFG%',
    'Team Win %','Height','Weight'
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

# At this point, categorical variables are converted into numeric form
# This keeps the pipeline simple and compatible with all downstream models
df_data = df_data.drop(columns=["Team"])
df_data = pd.get_dummies(df_data, columns=["Pos", "PosGroup", "Conference"])

# Raw statistics vary significantly across eras (pace, scoring inflation, etc.)
# Normalizing within each season preserves relative performance while
# removing global shifts over time
stat_cols = [
    'Minutes per game','FGA per game','2PA per game','3PA per game',
    'FTA per game','ORB per game','DRB per game','TRB per game',
    'AST per game','STL per game','BLK per game','TOV per game',
    'PF per game','PTS per game'
]

df_data[stat_cols] = df_data.groupby(
    "Season Ending Year"
)[stat_cols].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x
)

# Final pass ensures no missing values remain. Median imputation is used
# as a fallback for any edge cases not previously handled 
for col in df_data.columns:
    if df_data[col].isna().sum() > 0:
        if col == "Prev All Stars":
            df_data[col] = df_data[col].fillna(0)
        else:
            df_data[col] = df_data[col].fillna(df_data[col].median())

assert df_data.isna().sum().sum() == 0

# The 1999 season is removed due to the lockout that prevented All-Star 
# selection that year
df_data = df_data[df_data["Season Ending Year"] != 1999]

# Save the cleaned dataset 
print("Final shape:", df_data.shape)
print(df_data.describe())

df_data.to_csv("source\\cleaned\\cleaned_data.csv", index=False)