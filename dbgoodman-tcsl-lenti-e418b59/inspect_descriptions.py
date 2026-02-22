
import pandas as pd

file_path = 'd:/下载/dbgoodman-tcsl-lenti-e418b59/tables/supp_raw_data.xlsx'
try:
    df = pd.read_excel(file_path, sheet_name='descriptions')
    print(df.head().to_string())
    print(df.columns.tolist())
except Exception as e:
    print(e)
