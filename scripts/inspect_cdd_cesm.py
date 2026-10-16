import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define Local Paths
base_dir = "/Users/deepakbolleddu/UOW/research/CDD-CESM/PKG - CDD-CESM/CDD-CESM"
subtracted_dir = os.path.join(base_dir, "Subtracted images of CDD-CESM")
low_energy_dir = os.path.join(base_dir, "Low energy images of CDD-CESM")

print("==================================================")
print("       CDD-CESM DATASET INSPECTION & SPLIT        ")
print("==================================================")

# 2. Confirm Access to Images
subtracted_files = glob.glob(os.path.join(subtracted_dir, "**", "*.jpg"), recursive=True) + \
                   glob.glob(os.path.join(subtracted_dir, "**", "*.jpeg"), recursive=True)
low_energy_files = glob.glob(os.path.join(low_energy_dir, "**", "*.jpg"), recursive=True) + \
                   glob.glob(os.path.join(low_energy_dir, "**", "*.jpeg"), recursive=True)

print(f"[INFO] Subtracted images directory accessible: {len(subtracted_files)} files found.")
print(f"[INFO] Low energy images directory accessible: {len(low_energy_files)} files found.")

# 3. Locate Metadata Files (Excel / CSV)
metadata_files = glob.glob(os.path.join(base_dir, "**", "*.xlsx"), recursive=True) + \
                 glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

print("\n[INFO] Discovered metadata/annotation files:")
for f in metadata_files:
    print(f"   -> {f}")

# 4. Load and Inspect Metadata (Update filename if your Excel/CSV path differs)
# Note: CDD-CESM typically includes an Excel file for manual annotations or clinical metrics.
excel_candidates = [f for f in metadata_files if f.endswith('.xlsx')]
if excel_candidates:
    metadata_path = excel_candidates[0]
    print(f"\n[INFO] Loading metadata file: {metadata_path}")
    
    # Inspect sheet names if it's an Excel file
    xl = pd.ExcelFile(metadata_path)
    print(f"   Sheet names available: {xl.sheet_names}")
    
    # Load the primary annotation sheet (usually the first sheet or specific name)
    df = pd.read_excel(metadata_path, sheet_name=0)
    print(f"   Dataset shape (Rows, Columns): {df.shape}")
    print(f"   Columns available: {list(df.columns)[:8]} ...")
    
    # Basic Sanity / Missing Check
    print("\n[CHECK] Missing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # 5. Patient-Level Split (Train: 70%, Validation: 15%, Test: 15%)
    # Assuming columns like 'Patient ID' (or similar) exist in the dataframe. 
    # Let's dynamically check for patient identifier and label columns.
    patient_col = next((c for c in df.columns if 'patient' in c.lower() or 'id' in c.lower()), None)
    label_col = next((c for c in df.columns if 'pathology' in c.lower() or 'class' in c.lower() or 'diagnosis' in c.lower() or 'birads' in c.lower()), None)
    
    if patient_col:
        print(f"\n[INFO] Found patient identifier column: '{patient_col}'")
        unique_patients = df[patient_col].unique()
        print(f"   Total Unique Patients: {len(unique_patients)}")
        
        # Split patients into Train (70%) and Temp (30%)
        train_patients, temp_patients = train_test_split(unique_patients, test_size=0.30, random_state=42)
        # Split Temp into Validation (15%) and Test (15%)
        val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)
        
        # Assign split tags back to the dataframe
        df['Split'] = 'Train'
        df.loc[df[patient_col].isin(val_patients), 'Split'] = 'Validation'
        df.loc[df[patient_col].isin(test_patients), 'Split'] = 'Test'
        
        print("\n--- Summary per Split ---")
        print(df['Split'].value_counts())
        
        if label_col:
            print(f"\n--- Class Distribution per Split (using '{label_col}') ---")
            print(pd.crosstab(df['Split'], df[label_col]))
            
    else:
        print("\n[WARNING] Could not automatically detect a patient ID column. Please inspect the column names printed above.")

else:
    print("\n[WARNING] No Excel metadata file automatically isolated. Please check the files listed under base_dir.")
    
import json

# Group unique patients by their assigned split and convert numpy int64 to standard Python int
splits_dict = {
    "train": [int(x) for x in df[df['Split'] == 'Train']['Patient_ID'].unique()],
    "val": [int(x) for x in df[df['Split'] == 'Validation']['Patient_ID'].unique()],
    "test": [int(x) for x in df[df['Split'] == 'Test']['Patient_ID'].unique()]
}

# Save to JSON file
json_path = "/Users/deepakbolleddu/UOW/research/CDD-CESM/splits.json"
with open(json_path, "w") as f:
    json.dump(splits_dict, f, indent=4)

print(f"[INFO] Split JSON exported successfully to {json_path}")