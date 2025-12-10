import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
print("Loading data...")
df_nash = pd.read_csv('updatedData/Nashville_Tornado_DataInput_Final_111425(in).csv')
df_qs = pd.read_csv('updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv', encoding='latin1')

def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
    return df

df_nash = normalize_cols(df_nash)
df_qs = normalize_cols(df_qs)
df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)

# Map target
def map_target(val):
    if pd.isna(val): return np.nan
    if val == 0: return 0 # Undamaged
    if val == 1: return 1 # Low
    if val >= 2: return 2 # Significant
    return np.nan

df['target'] = df['degree_of_damage_u'].apply(map_target)
df = df.dropna(subset=['target'])

# Features (Hazard-Neutral)
feature_cols = [
    'mwfrs_u_wall', 'occupany_u', 'foundation_type_u', 
    'wall_thickness', 'wall_length_front', 'wall_cladding_u', 
    'wall_fenestration_per_w', 'roof_slope_u', 'construction_type_u',
    'building_urban_setting', 'retrofit_type_u', 'structural_wall_system_u',
    'parapet_height_m', 'year_built_u'
]
# Filter to columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].copy()
y = df['target']

# Identify categorical indices for SMOTENC
cat_cols = X.select_dtypes(include=['object']).columns
cat_indices = [X.columns.get_loc(c) for c in cat_cols]

# Preprocessing
# We need to handle NaNs before SMOTENC if we don't use a pipeline that handles it inside
# But SMOTENC handles categorical data, but not NaNs usually.
# Let's use a basic pipeline: Impute -> Encode -> (SMOTE) -> Model

# Simplified preprocessing for ablation
# 1. Impute Numeric
num_cols = X.select_dtypes(include=['number']).columns
imputer_num = SimpleImputer(strategy='median')
X[num_cols] = imputer_num.fit_transform(X[num_cols])

# 2. Impute Categorical & Encode
# Ensure all categorical columns are treated as strings to avoid mixed type errors
X[cat_cols] = X[cat_cols].astype(str)

imputer_cat = SimpleImputer(strategy='constant', fill_value='missing')
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])

# Define Scorer (Macro F1 and Low Damage F1)
def low_damage_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1], average=None)[0]

scoring = {
    'macro_f1': 'f1_macro',
    'low_f1': make_scorer(low_damage_f1)
}

# CV Setup
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

# --- Experiment 1: With SMOTENC ---
print("\nRunning WITH SMOTENC...")
f1_macro_smote = []
f1_low_smote = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Apply SMOTENC on training only
    try:
        smote = SMOTENC(categorical_features=cat_indices, k_neighbors=3, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except ValueError: # Fallback if class too small
        X_train_res, y_train_res = X_train, y_train
        
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_res, y_train_res)
    
    y_pred = model.predict(X_test)
    f1_macro_smote.append(f1_score(y_test, y_pred, average='macro'))
    f1_low_smote.append(f1_score(y_test, y_pred, labels=[1], average=None)[0] if 1 in y_test.values else 0)

print(f"With SMOTENC - Macro F1: {np.mean(f1_macro_smote):.3f}")
print(f"With SMOTENC - Low Damage F1: {np.mean(f1_low_smote):.3f}")

# --- Experiment 2: Without SMOTENC ---
print("\nRunning WITHOUT SMOTENC...")
f1_macro_no = []
f1_low_no = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # No SMOTE, just class weights
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    f1_macro_no.append(f1_score(y_test, y_pred, average='macro'))
    f1_low_no.append(f1_score(y_test, y_pred, labels=[1], average=None)[0] if 1 in y_test.values else 0)

print(f"Without SMOTENC - Macro F1: {np.mean(f1_macro_no):.3f}")
print(f"Without SMOTENC - Low Damage F1: {np.mean(f1_low_no):.3f}")
