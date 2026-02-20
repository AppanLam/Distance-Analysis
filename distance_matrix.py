import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# =========================
#           CONFIG
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--excel", type=str, default="Database_Distance Analysis.xlsx")
args, _ = parser.parse_known_args()

FILE_PATH = args.excel
DATA_SHEET = "Edited"
META_SHEET = "Feature_data"
# Ensure output filename is compatible with distance script expectations
OUTPUT_CSV = "preprocessed_output.csv" 

# =========================
#     UNIVERSAL PARSERS
# =========================

def to_binary(series):
    """Maps yes/no or available/not available strings to 1/0."""
    s = series.astype(str).str.strip().lower()
    return s.map({
        'yes': 1, 'no': 0, 
        'available': 1, 'not available': 0,
        'memory card supported': 1, 'memory card not supported': 0
    }).fillna(0)

def universal_numeric_cleaner(x):
    """
    Universal cleaner for Task 3: Handles currency (₹), commas, 
    and resolution patterns (e.g., '1440 x 3216') for any dataset.
    """
    if pd.isna(x): return np.nan
    
    # 1. Handle resolution/gear patterns like '12x4' or '1440 x 3216'
    if isinstance(x, str) and ('x' in x.lower() or '×' in x):
        parts = re.findall(r"\d+", x.replace(' ', ''))
        if len(parts) >= 2:
            return float(parts[0]) * float(parts[1])
            
    # 2. Strip currency symbols, units, and commas (e.g., ₹54,999 -> 54999.0)
    clean_s = re.sub(r'[^\d.]', '', str(x).replace(',', ''))
    try:
        return float(clean_s)
    except:
        return np.nan

def main():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Missing {FILE_PATH}")

    # Load data and metadata rules
    df = pd.read_excel(FILE_PATH, sheet_name=DATA_SHEET)
    meta = pd.read_excel(FILE_PATH, sheet_name=META_SHEET)

    processed_df = pd.DataFrame()

    # Apply universal logic based on Feature_data rules
    for _, row in meta.iterrows():
        col_name = row['Feature']
        col_type = row['Type']
        
        if col_name not in df.columns:
            continue
            
        if col_type == 'Numeric':
            processed_df[col_name] = df[col_name].apply(universal_numeric_cleaner)
        elif col_type == 'Binary':
            processed_df[col_name] = to_binary(df[col_name])
        elif col_type == 'Nominal':
            # One-Hot Encoding for compatibility with distance matrix
            dummies = pd.get_dummies(df[col_name], prefix=col_name)
            processed_df = pd.concat([processed_df, dummies], axis=1)

    # Handle missing values to prevent distance matrix errors
    processed_df = processed_df.fillna(processed_df.mean())

    # Save to the filename expected by the teammate's script
    processed_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Task 3 Complete: Universal output saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()