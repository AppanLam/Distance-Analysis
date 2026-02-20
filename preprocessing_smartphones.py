import argparse
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---- CONFIGURATION ----
FILE_PATH = "Database_Distance Analysis.xlsx"
DATA_SHEET = "Edited"        # The sheet containing product data
META_SHEET = "Feature_data"  # The sheet containing your new metadata rules

# =========================
#      UNIVERSAL PARSERS
# =========================

def to_binary(series):
    """Maps common yes/no or supported/not strings to 1/0."""
    s = series.astype(str).str.strip().str.lower()
    return s.map({
        'yes': 1, 'no': 0, 
        'available': 1, 'not available': 0,
        'memory card supported': 1, 'memory card not supported': 0
    }).fillna(0)

def universal_numeric_cleaner(x):
    """Strips currency (₹), commas, and units to find a clean number."""
    if pd.isna(x): return np.nan
    # Removes ₹ and commas so price becomes a float (e.g., 54999.0)
    clean_s = re.sub(r'[^\d.]', '', str(x).replace(',', ''))
    try:
        return float(clean_s)
    except:
        return np.nan

def parse_range(x):
    """Generic range parser: Converts '61-70' to 65.5."""
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("–", "-")
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    nums = re.findall(r"\d+", s)
    return float(nums[0]) if nums else np.nan

def parse_multi_numbers(x):
    """Generic multiplier: Handles '12x4' (tractors) or '1440 x 3216' (phones)."""
    if pd.isna(x): return np.nan
    s = str(x).lower().replace('×', 'x')
    if 'x' in s:
        # Multiplies numbers found around the 'x'
        nums = re.findall(r"(\d+(?:\.\d+)?)", s)
        if len(nums) >= 2:
            return float(nums[-2]) * float(nums[-1])
    return universal_numeric_cleaner(x)

# ---- INPUT LOGIC ----
parser = argparse.ArgumentParser()
parser.add_argument("--excel", type=str, default=FILE_PATH)
args, _ = parser.parse_known_args()
FILE_PATH = args.excel

# ---- LOAD DATA ----
df_raw = pd.read_excel(FILE_PATH, sheet_name=DATA_SHEET)
meta = pd.read_excel(FILE_PATH, sheet_name=META_SHEET)

meta_inc = meta[meta['included_in_analysis'].astype(str).str.lower() == "yes"].copy()
valid_cols = [c for c in meta_inc['feature_name'].tolist() if c in df_raw.columns]
df = df_raw[valid_cols].copy()

# ---- UNIVERSAL PROCESSOR ----
final_df = pd.DataFrame(index=df.index)

for _, row in meta_inc.iterrows():
    col = row['feature_name']
    if col not in df.columns: continue
    
    d_type = str(row['data_types']).lower()
    norm = str(row['normalization_method']).lower()
    ideal = str(row['ideal_direction']).lower()

    # 1. Metric: Handles clean numbers and currency like ₹
    if d_type == 'metric':
        clean_vals = df[col].apply(universal_numeric_cleaner)
        vals = pd.to_numeric(clean_vals, errors='coerce').values.reshape(-1, 1)
        if norm == 'min-max':
            scaled = MinMaxScaler().fit_transform(vals).ravel()
            if ideal == "lower is better":
                scaled = 1 - scaled
        else:
            scaled = StandardScaler().fit_transform(vals).ravel()
        final_df[col] = scaled

    # 2. Ordinal: Pattern-based detection
    elif d_type == 'ordinal':
        sample = str(df[col].dropna().iloc[0]).lower()
        if '-' in sample:
            final_df[col] = df[col].apply(parse_range)
        elif 'x' in sample or '×' in sample:
            final_df[col] = df[col].apply(parse_multi_numbers)
        else:
            final_df[col] = df[col].apply(universal_numeric_cleaner)

    # 3. Binary: Yes/No mapping
    elif d_type == 'binary':
        final_df[col] = to_binary(df[col])

    # 4. Nominal: Category expansion
    elif d_type in ['nominal', 'normal']:
        dummies = pd.get_dummies(df[col], prefix=col)
        final_df = pd.concat([final_df, dummies], axis=1)

# Cleanup and save
final_df = final_df.fillna(final_df.mean())
final_df.to_csv("preprocessed_output.csv", index=False)
print(f"Task 3 Complete. Output saved for: {FILE_PATH}")