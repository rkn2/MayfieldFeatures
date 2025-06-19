import pandas as pd
import numpy as np
import re
import os


def reverse_one_hot_encoding(input_csv_path, output_csv_path):
    """
    Reads a CSV, identifies one-hot encoded columns based on prefixes,
    consolidates them into single categorical columns, and saves a new CSV.

    Args:
        input_csv_path (str): Path to the original CSV file.
        output_csv_path (str): Path to save the new, cleaned CSV file.
    """
    print(f"--- Starting to reverse one-hot encoding for: {input_csv_path} ---")

    try:
        # FIX: Added encoding='latin-1' to handle special characters in the file.
        df = pd.read_csv(input_csv_path, low_memory=False, encoding='latin-1')
        print(f"Successfully loaded data. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv_path}'")
        return

    # Clean up column names by stripping leading/trailing whitespace
    df.columns = df.columns.str.strip()

    # --- Define the prefixes of the one-hot encoded columns ---
    # The key is the new categorical column name we want to create.
    # The value is the prefix used for the one-hot encoded columns.
    encoding_groups = {
        'construction_material_h': 'const_material_h_',
        'construction_material_v': 'const_material_v_',
        'property_type': 'prop_',
        'owner_type': 'owner_'
    }

    df_copy = df.copy()
    columns_to_drop = []

    for new_col_name, prefix in encoding_groups.items():
        print(f"\nProcessing group: '{new_col_name}' with prefix '{prefix}'...")

        # Find all columns that belong to this group
        encoded_cols = [col for col in df_copy.columns if col.startswith(prefix)]

        if not encoded_cols:
            print(f"  No columns found with prefix '{prefix}'. Skipping.")
            continue

        print(f"  Found {len(encoded_cols)} related columns.")
        columns_to_drop.extend(encoded_cols)

        # Ensure the data in these columns is numeric (0 or 1)
        df_copy[encoded_cols] = df_copy[encoded_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Reverse the one-hot encoding
        # idxmax(axis=1) finds the first column name with the maximum value (which will be 1)
        def get_category(row):
            if row.sum() == 0:
                return 'unknown'  # Handle cases where no flag is set
            return row.idxmax()

        # Apply the function and clean up the resulting category name
        df_copy[new_col_name] = df_copy[encoded_cols].apply(get_category, axis=1)
        df_copy[new_col_name] = df_copy[new_col_name].str.replace(prefix, '', regex=False).str.strip()

        print(f"  Created new categorical column: '{new_col_name}'")

    # Drop the original one-hot encoded columns
    print(f"\nDropping {len(columns_to_drop)} original one-hot encoded columns...")
    df_copy.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Save the new dataframe
    try:
        df_copy.to_csv(output_csv_path, index=False)
        print(f"\n--- Successfully saved cleaned data to: {output_csv_path} ---")
        print(f"New shape: {df_copy.shape}")
    except Exception as e:
        print(f"\nError saving the new CSV file: {e}")


if __name__ == '__main__':
    # Define your input and output file paths here
    INPUT_FILE = 'QuadState_Tornado_DataInputv2.csv'
    OUTPUT_FILE = 'QuadState_Tornado_DataInput_Categorical.csv'

    # Check if the input file exists in the current directory
    if not os.path.exists(INPUT_FILE):
        print(f"FATAL ERROR: The input file '{INPUT_FILE}' was not found in this directory.")
        print("Please make sure the script is in the same folder as your data.")
    else:
        reverse_one_hot_encoding(INPUT_FILE, OUTPUT_FILE)

