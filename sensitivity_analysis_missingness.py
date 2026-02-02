import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTENC
import replicate_analysis_damage_target as main_analysis

# Config
RANDOM_STATE = 42
OUTPUT_DIR = 'tornado_vulnerability_outputs_damage_target'

def run_sensitivity_analysis():
    print("Loading data...")
    df = main_analysis.load_and_preprocess_data()
    df = main_analysis.engineer_features(df)
    
    # Get feature sets
    numeric_cols, cat_cols, haz_cols = main_analysis.get_feature_sets()
    
    # We define relevant features (Hazard-Neutral + Distance for context if we want, but let's stick to the paper's main model)
    # The paper uses distance in the main model.
    feature_set = numeric_cols + cat_cols # Hazard-Neutral (Wait, main paper uses distance? Yes, main_distance.tex uses distance.)
    # Let's ensure we use the same feature set as the "Best Model" in the paper.
    # In replicate_analysis_damage_target.py, 'Hazard-Neutral' EXCLUDES hazard columns (no ef_numeric), but includes 'distance_km' in numeric_cols?
    # Let's check get_feature_sets in replicate_analysis_damage_target.py.
    # It puts 'distance_km' in numeric_features. So 'Hazard-Neutral' includes distance.
    
    # Prepare X, y
    X = df[feature_set].copy()
    y = df['target']
    
    # Target Encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Identify indices for SMOTE
    cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    
    # Define Strategies
    strategies = {
        'Baseline (Median/Missing)': {
            'numeric': SimpleImputer(strategy='median'),
            'categorical': SimpleImputer(strategy='constant', fill_value='missing') # We actually handle this in preprocessing usually by str conversion
        },
        'MICE (Iterative)': {
            'numeric': IterativeImputer(max_iter=10, random_state=RANDOM_STATE),
            'categorical': 'mode' # MICE for categorical is complex. We will approximate by encoding then imputing or using Mode.
            # actually IterativeImputer works on numbers. So we will Ordinal Encode FIRST, preserving NaNs, then MICE everything.
        }
    }
    
    results = []
    
    print("Running Sensitivity Analysis...")
    
    # We need to manually handle the pipeline to allow for MICE on encoded categoricals
    
    # --- Strategy 1: Baseline ---
    print("\n--- Strategy: Baseline (Median Imputation) ---")
    # This mimics the main script: SimpleImputer(median) for num, "Missing" category for cat
    
    # 1. Preprocess Categoricals (NaN -> 'missing' -> Int)
    X_base = X.copy()
    for c in cat_cols:
        if c in X_base.columns:
            X_base[c] = X_base[c].fillna('missing').astype(str)
            
    # 2. Encode to Int
    enc_base = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_base[cat_cols] = enc_base.fit_transform(X_base[cat_cols])
    
    # Remove commas from numeric columns if they are strings
    for col in numeric_cols:
        if col in X_base.columns and X_base[col].dtype == object:
            X_base[col] = X_base[col].astype(str).str.replace(',', '').str.replace(' ', '')

    # 3. Impute Numerics (Median)
    num_subset = [c for c in numeric_cols if c in X_base.columns]
    imp_base = SimpleImputer(strategy='median')
    X_base[num_subset] = imp_base.fit_transform(X_base[num_subset])
    
    # 4. Train & Permutation Importance
    ranks_base = get_feature_importance(X_base, y_enc, cat_indices, "Baseline")
    
    # --- Strategy 2: MICE (Iterative) ---
    print("\n--- Strategy: MICE (Iterative Imputation) ---")
    # For MICE to work on categoricals, we Ordinal Encode them but KEEP NaNs.
    # Then IterativeImputer fills the NaNs (as floats). We round them back to integers?
    # Or we just leave them as continuous capabilities in the RF.
    
    X_mice = X.copy()
    
    # 1. Encode Categoricals (Preserve NaN)
    # OrdinalEncoder doesn't handle NaNs by passing them through (it raises error or encodes them).
    # We need to map categories to ints manually or use masking.
    
    cat_encoders = {}
    for c in cat_cols:
        if c in X_mice.columns:
            # We treat 'unknown' strings as NaN for MICE to impute them?
            # Or do we treat 'unknown' as a valid class?
            # The USER concern is "missingness". 
            # If we explicitly have np.nan, we want to impute.
            # In load_data, 'unknown' strings were converted to np.nan.
            # So X_mice[c] has np.nan.
            
            # Mask selection
            valid_mask = ~X_mice[c].isna()
            le_cat = LabelEncoder()
            # Force string to avoid mixed type errors
            X_mice.loc[valid_mask, c] = le_cat.fit_transform(X_mice.loc[valid_mask, c].astype(str))
            cat_encoders[c] = le_cat
            
    # Remove commas
    for col in numeric_cols:
        if col in X_mice.columns and X_mice[col].dtype == object:
             X_mice[col] = X_mice[col].astype(str).str.replace(',', '').str.replace(' ', '')
             
    # Convert to float for MICE
    X_mice = X_mice.astype(float)
    
    # 2. Iterative Imputer on WHOLE matrix
    # This uses correlations between all features (num and cat) to fill missing values
    mice = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
    X_mice_imputed = mice.fit_transform(X_mice)
    X_mice = pd.DataFrame(X_mice_imputed, columns=X.columns)
    
    # 3. Round categorical columns to nearest integer (to map back to discrete levels)
    # Although RF handles floats fine, rounding preserves the "categorical" nature
    for c in cat_cols:
        if c in X_mice.columns:
            X_mice[c] = X_mice[c].round()
            
    # 4. Train & Permutation Importance
    ranks_mice = get_feature_importance(X_mice, y_enc, cat_indices, "MICE")
    
    # --- Strategy 3: Missing Indicator ---
    print("\n--- Strategy: Missing Indicator ---")
    # Explicitly add _is_missing columns for features with frequent missingness
    X_ind = X.copy()
    
    high_missing_cols = ['wall_thickness', 'foundation_type_u'] # As noted by user
    for c in high_missing_cols:
        if c in X_ind.columns:
            X_ind[f'{c}_is_missing'] = X_ind[c].isna().astype(int)
            
    # Then do Baseline processing for the rest
    for c in cat_cols:
        if c in X_ind.columns:
            X_ind[c] = X_ind[c].fillna('missing').astype(str)
            
    # Remove commas for Numerics
    for col in numeric_cols:
        if col in X_ind.columns and X_ind[col].dtype == object:
             X_ind[col] = X_ind[col].astype(str).str.replace(',', '').str.replace(' ', '')
             
    enc_ind = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_ind[cat_cols] = enc_ind.fit_transform(X_ind[cat_cols])
    
    num_subset_ind = [c for c in numeric_cols if c in X_ind.columns]
    imp_ind = SimpleImputer(strategy='median')
    X_ind[num_subset_ind] = imp_ind.fit_transform(X_ind[num_subset_ind])
    
    # Update cat indices for SMOTE (columns might have shifted or added? appended is safe)
    # Actually we added columns. SMOTE needs indices of ALL categoricals.
    # The new binary indicators are technically categorical too.
    cat_cols_ind = cat_cols + [f'{c}_is_missing' for c in high_missing_cols if c in X.columns]
    cat_indices_ind = [X_ind.columns.get_loc(c) for c in cat_cols_ind if c in X_ind.columns]
    
    ranks_ind = get_feature_importance(X_ind, y_enc, cat_indices_ind, "Indicator")

    # --- Compare ---
    compare_ranks(ranks_base, ranks_mice, ranks_ind)

def get_feature_importance(X, y, cat_indices, strategy_name):
    # 5-fold CV with SMOTENC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    importances = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # SMOTE
        try:
             # handle empty cat_indices?
            if not cat_indices:
                 from imblearn.over_sampling import SMOTE
                 sampler = SMOTE(random_state=RANDOM_STATE)
            else:
                sampler = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        except:
            X_res, y_res = X_train, y_train
            
        # RF
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_res, y_res)
        
        # Permutation Importance
        baseline_acc = accuracy_score(y_val, rf.predict(X_val))
        
        fold_imps = {}
        for feat in X.columns:
            save = X_val[feat].copy()
            X_val[feat] = np.random.permutation(X_val[feat])
            perm_acc = accuracy_score(y_val, rf.predict(X_val))
            X_val[feat] = save
            fold_imps[feat] = baseline_acc - perm_acc
            
        importances.append(fold_imps)
        
    # Average across folds
    avg_imp = pd.DataFrame(importances).mean().sort_values(ascending=False)
    return avg_imp

def compare_ranks(base, mice, ind):
    print("\n--- Feature Importance Ranking Comparison ---")
    
    # Combine into DataFrame
    df_ranks = pd.DataFrame({
        'Baseline_Score': base,
        'MICE_Score': mice,
        'Indicator_Score': ind
    })
    
    # Add Rank columns
    df_ranks['Baseline_Rank'] = df_ranks['Baseline_Score'].rank(ascending=False)
    df_ranks['MICE_Rank'] = df_ranks['MICE_Score'].rank(ascending=False)
    df_ranks['Indicator_Rank'] = df_ranks['Indicator_Score'].rank(ascending=False)
    
    # Focus on key features mentioned by User
    key_features = ['distance_km', 'wall_thickness', 'foundation_type_u', 'roof_substrate_type_u']
    
    print("\nScore Comparison (Decrease in Accuracy):")
    print(df_ranks.loc[key_features])
    
    print("\nRank Comparison:")
    print(df_ranks.loc[key_features, ['Baseline_Rank', 'MICE_Rank', 'Indicator_Rank']])
    
    # Save
    df_ranks.to_csv(f'{OUTPUT_DIR}/sensitivity_missingness_comparison.csv')
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Filter to top 10 features from Baseline
    top_feats = df_ranks.sort_values('Baseline_Score', ascending=False).head(10).index
    
    plot_data = df_ranks.loc[top_feats, ['Baseline_Score', 'MICE_Score', 'Indicator_Score']]
    plot_data.plot(kind='bar', figsize=(14, 7), width=0.8)
    plt.title("Sensitivity Analysis: Feature Importance under Missingness Strategies")
    plt.ylabel("Importance (Decrease in Accuracy)")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sensitivity_missingness_plot.png')
    print(f"Plot saved to {OUTPUT_DIR}/sensitivity_missingness_plot.png")

if __name__ == "__main__":
    run_sensitivity_analysis()
