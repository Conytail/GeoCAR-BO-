
import pandas as pd
import os

file_path = 'D:/Desktop/论文/BioFoundry/dbgoodman-tcsl-lenti-e418b59/tables/supp_raw_data.xlsx'

try:
    # Read all sheet names
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    print(f"Sheet names in {os.path.basename(file_path)}: {sheet_names}")

    # Inspect first few rows of each sheet to find Sequence info
    # Inspect specific sheet
    sheet = 'descriptions'
    if sheet in sheet_names:
        print(f"\n--- Sheet: {sheet} ---")
        df = pd.read_excel(file_path, sheet_name=sheet)
        print(df.to_string())
    else:
        print(f"Sheet {sheet} not found.")

except Exception as e:
    print(f"Error reading Excel file: {e}")
