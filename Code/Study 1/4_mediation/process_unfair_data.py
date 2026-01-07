import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def process_unfair_data():
    """
    Process demo.xlsx and filter unfair condition data
    Unfair condition: amount_of_allocation between 10-14 (inclusive)
    Save as unfair_data_for_mediation.xlsx
    
    DATA FILE NOTE:
    This script uses demo.xlsx, which contains a subset of the full dataset
    (5 participants per group: human, gpt3.5, o3, V3, R1).
    
    To use the full dataset:
    1. Access the full dataset at: [INSERT LINK HERE]
    2. Download the file named "Study1_experimental_data.xlsx"
    3. Replace "demo.xlsx" with "Study1_experimental_data.xlsx" in the read_excel() call below
    
    Note: When switching to full data, computation time may increase significantly.
    """
    print("=== Processing Unfair Data ===")
    print("Reading demo.xlsx...")
    
    # Read original data - Using demo data - see DATA FILE NOTE above for full data access
    input_file = 'demo.xlsx'
    df = pd.read_excel(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if amount_of_allocation column exists
    if 'amount_of_allocation' not in df.columns:
        print("Error: 'amount_of_allocation' column not found in the data.")
        print("Available columns:", df.columns.tolist())
        return
    
    # Display allocation value distribution
    print(f"\nAllocation value distribution:")
    print(df['amount_of_allocation'].value_counts().sort_index())
    
    # Filter unfair conditions: amount_of_allocation between 10-14 (inclusive)
    print(f"\nFiltering unfair conditions (allocation = 10-14)...")
    df_unfair = df[(df['amount_of_allocation'] >= 10) & (df['amount_of_allocation'] <= 14)].copy()
    
    print(f"Filtered data shape: {df_unfair.shape}")
    print(f"Number of rows filtered: {len(df_unfair)}")
    
    # Display filtered allocation distribution
    if len(df_unfair) > 0:
        print(f"\nFiltered allocation distribution:")
        print(df_unfair['amount_of_allocation'].value_counts().sort_index())
        
        # Display sample size by group
        if 'group' in df_unfair.columns:
            print(f"\nSample size by group:")
            print(df_unfair['group'].value_counts())
    else:
        print("Warning: No data found matching the unfair condition (allocation 10-14)")
        return
    
    # Save to Excel file
    output_file = 'unfair_data_for_mediation.xlsx'
    print(f"\nSaving filtered data to {output_file}...")
    df_unfair.to_excel(output_file, index=False)
    
    print(f"✓ Successfully saved {len(df_unfair)} rows to {output_file}")
    print(f"\n=== Processing Complete ===")

if __name__ == "__main__":
    process_unfair_data()

