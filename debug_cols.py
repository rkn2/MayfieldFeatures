import pandas as pd

# Load data
print("Loading data...")
df_nash = pd.read_excel('Nashville_Tornado_DataInput_Final_110725.xlsx')
df_qs = pd.read_csv('QuadState_Tornado_DataInputv2.csv', encoding='latin1')

def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
    return df

df_nash = normalize_cols(df_nash)
df_qs = normalize_cols(df_qs)

print("Nashville Columns containing 'ef':")
print([c for c in df_nash.columns if 'ef' in c])

print("\nQuadState Columns containing 'ef':")
print([c for c in df_qs.columns if 'ef' in c])
