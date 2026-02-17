import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---- CONFIGURATION ----
FILE_PATH = "Database_Distance Analysis.xlsx"
DATA_SHEET = "Edited"        # The sheet containing variant data
META_SHEET = "Feature_data"  # The sheet containing metadata rules

def to_binary(series):
    """Maps common yes/no strings to 1/0."""
    s = series.astype(str).str.strip().str.lower()
    return s.map({'yes': 1, 'no': 0, 'available': 1, 'not available': 0}).fillna(0)

def parse_hp_range(x):
    """Converts '61-70' to 65.5 for numeric comparison."""
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("–", "-")
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    nums = re.findall(r"\d+", s)
    return float(nums[0]) if nums else np.nan

def parse_gears(x):
    """Converts '12x4' to 48 (total gear combinations)."""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower().replace("×", "x")
    m = re.match(r"^(\d+)\s*x\s*(\d+)$", s)
    if m:
        return float(m.group(1)) * float(m.group(2))
    nums = re.findall(r"\d+", s)
    return float(nums[0]) if nums else np.nan

# ---- LOAD DATA ----
df_raw = pd.read_excel(FILE_PATH, sheet_name=DATA_SHEET)
meta = pd.read_excel(FILE_PATH, sheet_name=META_SHEET)

# Filter: Only keep features where 'included_in_analysis' is 'yes'
meta_inc = meta[meta['included_in_analysis'].astype(str).str.lower() == "yes"].copy()
# Ensure feature names match the raw data column headers
valid_cols = [c for c in meta_inc['feature_name'].tolist() if c in df_raw.columns]
df = df_raw[valid_cols].copy()

# ---- PROCESS BY DATA TYPE ----
final_df = pd.DataFrame(index=df.index)

for _, row in meta_inc.iterrows():
    col = row['feature_name']
    if col not in df.columns: continue
    
    d_type = str(row['data_types']).lower()
    norm = str(row['normalization_method']).lower()
    ideal = str(row['ideal_direction']).lower()

    # 1. Metric: Normalization
    if d_type == 'metric':
        vals = pd.to_numeric(df[col], errors='coerce').values.reshape(-1, 1)
        if norm == 'min-max':
            scaled = MinMaxScaler().fit_transform(vals).ravel()
            if ideal == "lower is better": # e.g., Minimum turning circle
                scaled = 1 - scaled
        else:
            scaled = StandardScaler().fit_transform(vals).ravel()
        final_df[col] = scaled

    # 2. Ordinal: Custom Parsing
    elif d_type == 'ordinal':
        if 'hp' in col.lower():
            final_df[col] = df[col].apply(parse_hp_range)
        elif 'gear' in col.lower():
            final_df[col] = df[col].apply(parse_gears)
        else:
            final_df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Binary: 1/0
    elif d_type == 'binary':
        final_df[col] = to_binary(df[col])

    # 4. Nominal: One-Hot Encoding
    elif d_type == 'nominal':
        dummies = pd.get_dummies(df[col], prefix=col)
        final_df = pd.concat([final_df, dummies], axis=1)

# Final cleanup: Handle NaNs by filling with column average
final_df = final_df.fillna(final_df.mean())

# ---- SAVE OUTPUT ----
final_df.to_csv("preprocessed_output.csv", index=False)
print(f"Task 2 Complete. Preprocessed data saved with {final_df.shape[1]} features.")