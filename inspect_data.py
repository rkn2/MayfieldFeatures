import pandas as pd
import numpy as np

def inspect():
    print("--- NASHVILLE ---")
    df_n = pd.read_csv('updatedData/Nashville_Tornado_DataInput_Final_111425(in).csv')
    print("Columns:", df_n.columns.tolist())
    print("Shape:", df_n.shape)
    print("Row 0:", df_n.iloc[0].tolist()[:15])

    print("\n--- QUADSTATE ---")
    # Read without header
    df_q = pd.read_csv('updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv', header=None)
    print("Shape:", df_q.shape)
    print("Row 0:", df_q.iloc[0].tolist()[:15])
    print("Row 1:", df_q.iloc[1].tolist()[:15])
    
    # Try to verify if row 0 could be headers?
    # If row 0 has string 'tornado_ef' or similar
    row0 = df_q.iloc[0].astype(str).tolist()
    if 'tornado' in [x.lower() for x in row0]:
        print("Row 0 might be header.")
    else:
        print("Row 0 does not look like header.")

if __name__ == "__main__":
    inspect()
