import argparse
import os
import re
import numpy as np
import pandas as pd

# =========================
#           CONFIG
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--excel", type=str, default="Database_Distance Analysis.xlsx")
args, _ = parser.parse_known_args()

FILE_PATH = args.excel
DATA_SHEET = "Edited"
META_SHEET = "Feature_data"
# Synced name for teammate's distance script
OUTPUT_CSV = "preprocessed_output.csv" 

# =========================
#     UNIVERSAL PARSERS
# =========================

def to_binary(series):
    """Maps common categorical truth values to 1/0."""
    s = series.astype(str).str.strip().lower()
    return s.map({
        'yes': 1, 'no': 0, 
        'available': 1, 'not available': 0,
        'memory card supported': 1, 'memory card not supported': 0
    }).fillna(0)

def universal_numeric_cleaner(x):
    """Handles currency (₹), commas, and resolutions (1440 x 3216)."""
    if pd.isna(x): return np.nan
    
    # 1. Handle resolution/multiplication patterns
    if isinstance(x, str) and ('x' in x.lower() or '×' in x):
        parts = re.findall(r"\d+", x.replace(' ', ''))
        if len(parts) >= 2:
            return float(parts[0]) * float(parts[1])
            
    # 2. Strip currency symbols and non-numeric characters
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

    # Prioritize 'model' for phones and 'product' for tractors (long strings)
    id_col = None
    if 'model' in df.columns:
        id_col = 'model'
    elif 'product' in df.columns:
        id_col = 'product'
    elif 'brand' in df.columns:
        id_col = 'brand'
    
    if id_col:
        processed_df[id_col] = df[id_col] # Keep the unique label
    # ---------------------------------------------------

    # Apply universal logic based on Feature_data rules
    for _, row in meta.iterrows():
        col_name = row['Feature']
        col_type = row['Type']
        
        if col_name not in df.columns or col_name == id_col:
            continue
            
        if col_type == 'Numeric':
            processed_df[col_name] = df[col_name].apply(universal_numeric_cleaner)
        elif col_type == 'Binary':
            processed_df[col_name] = to_binary(df[col_name])
        elif col_type == 'Nominal':
            # One-Hot Encoding for categorical data compatibility
            dummies = pd.get_dummies(df[col_name], prefix=col_name)
            processed_df = pd.concat([processed_df, dummies], axis=1)

    # Fill missing values to prevent distance matrix errors
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())

    # Final output synced with teammate's script
    processed_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Task 3 Complete. Universal output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()