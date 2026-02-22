
import pandas as pd
import os

# Configuration
INPUT_FILE = 'd:/下载/dbgoodman-tcsl-lenti-e418b59/tables/fig2a_ctv.csv'
OUTPUT_FILE = 'd:/下载/dbgoodman-tcsl-lenti-e418b59/training_data.csv'

def process_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print(f"Reading data from {INPUT_FILE}...")
    try:
        # Skip the comment line at the top
        df = pd.read_csv(INPUT_FILE, skiprows=1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("Columns found:", df.columns.tolist())

    # Required columns
    # X: Domain Name (Composition)
    # Y: DESeq2 RLog CAR Score (Functional Score)
    # Metadata: Keep context
    
    required_cols = ['Domain Name', 'DESeq2 RLog CAR Score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return

    # Filter columns to keep
    # We'll keep mostly everything for the "tidy" format, but ensure X and Y are present
    keep_cols = [
        'Domain Name', 'DESeq2 RLog CAR Score', 
        'Assay Name', 'Donor Name', 'Experimental Timepoint', 
        'T Cell Type', 'K562 Condition', 'Sorted Cell Group', 'Read Count'
    ]
    
    # Select available columns from the keep list
    actual_keep_cols = [col for col in keep_cols if col in df.columns]
    clean_df = df[actual_keep_cols].copy()

    # Drop rows with missing values in critical columns
    before_drop = len(clean_df)
    clean_df = clean_df.dropna(subset=['Domain Name', 'DESeq2 RLog CAR Score'])
    after_drop = len(clean_df)
    
    if before_drop != after_drop:
        print(f"Dropped {before_drop - after_drop} rows with missing X/Y values.")

    # Rename columns for clarity if needed, or keep as is. 
    # For ML training, simple names are often better.
    clean_df = clean_df.rename(columns={
        'Domain Name': 'domain_composition',
        'DESeq2 RLog CAR Score': 'score'
    })

    # Save to CSV
    print(f"Saving {len(clean_df)} rows to {OUTPUT_FILE}...")
    clean_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- Top 5 Rows of Training Data ---")
    print(clean_df.head().to_string())
    print("\n--- Data Info ---")
    print(clean_df.info())

if __name__ == "__main__":
    process_data()
