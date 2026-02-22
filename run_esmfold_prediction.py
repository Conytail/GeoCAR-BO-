import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import pandas as pd
import os
import zipfile
import shutil
from tqdm import tqdm

# Paths
INPUT_CSV = "D:/Desktop/论文/BioFoundry/ready_for_structure_prediction.csv"
OUTPUT_DIR = "D:/Desktop/论文/BioFoundry/pdb_dataset"
SCORE_FILE = "D:/Desktop/论文/BioFoundry/structure_prediction_plddt.csv"
ZIP_FILE = "D:/Desktop/论文/BioFoundry/predicted_pdbs.zip"

def main():
    # 0. Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 1. Load Dataset
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total sequences to fold: {len(df)}")

    # 2. Load ESMFold
    print("Loading ESMFold model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    
    if device == "cpu":
        print("WARNING: Running on CPU will be extremely slow for 2000+ sequences.")

    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        model = model.to(device)
        model.eval() # Set to evaluation mode
        
        # Enable FP16 for speed on GPU if available
        if device == "cuda":
            model.half()
            print("Model loaded in half percision (FP16).")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Predict Loop
    results = []
    
    # Use tqdm for progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Folding Proteins"):
        seq = row['AA_Sequence']
        name = row['ID']
        
        # Sanitize filename (windows compatibility)
        safe_name = name.replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_")
        pdb_path = os.path.join(OUTPUT_DIR, f"{safe_name}.pdb")
        
        # Skip if already exists (resume capability)
        # Skip if already exists (resume capability)
        if os.path.exists(pdb_path):
            print(f"Skipping {name}, already exists.")
            continue

        try:
            with torch.no_grad():
                inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                
                # Get PDB String
                pdb_str = model.output_to_pdb(outputs)[0]
                
                # Calculate Avg pLDDT
                # outputs.plddt is shape (batch, seq_len)
                # pLDDT is usually 0-100 scale in output tensor? Or 0-1?
                # Using model.output_to_pdb does scaling usually? 
                # Let's inspect the raw tensor. ESMFold usually outputs 0-100 float32.
                plddt_tensor = outputs.plddt
                avg_plddt = plddt_tensor[0].mean().item()
                
            # Save PDB
            with open(pdb_path, "w") as f:
                f.write(pdb_str)
            
            results.append({
                "ID": name,
                "Avg_pLDDT": avg_plddt
            })
            
        except Exception as e:
            print(f"Failed to fold {name}: {e}")

    # 4. Save Scores
    score_df = pd.DataFrame(results)
    score_df.to_csv(SCORE_FILE, index=False)
    print(f"Scores saved to {SCORE_FILE}")
    print(f"Average pLDDT across dataset: {score_df['Avg_pLDDT'].mean():.2f}")

    # 5. Zip Files
    print(f"Zipping {OUTPUT_DIR} to {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip with relative path
                zipf.write(file_path, os.path.basename(file_path))
    
    print("Done!")

if __name__ == "__main__":
    main()
