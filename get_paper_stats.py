import pandas as pd
import numpy as np
from replicate_analysis import load_and_preprocess_data, engineer_features, get_models
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def get_stats():
    # 1. Counts
    df = load_and_preprocess_data()
    # Historic counts? 
    # Use 'historic registered' (Nashville) and 'located_in_historic_district' or 'national_register_listing_year'?
    # Nashville: 'located_in_historic_district'
    # QuadState: 'located_in_historic_district'
    
    # Normalize check
    cols = df.columns.tolist()
    hist_col = 'located_in_historic_district'
    if hist_col in df.columns:
        n_historic = df[df[hist_col].astype(str).str.lower().isin(['yes', 'y', 'true', '1'])].shape[0]
    else:
        n_historic = 0
        
    print(f"Total Buildings: {len(df)}")
    print(f"Historic Buildings: {n_historic}")
    print(f"Non-Historic: {len(df) - n_historic}")
    
    df = engineer_features(df)
    
    # Damage Classes
    print("\nDamage Class Counts:")
    print(df['target'].value_counts())
    print(df['target'].value_counts(normalize=True) * 100)

    # 2. Performance Metrics (RF Hazard Inclusive)
    print("\nRunning Model Evaluation for Classification Report...")
    # Features
    numeric_features = [
        'number_stories', 'year_built_u', 'building_area_m2', 'buidling_height_m', 
        'wall_length_side', 'wall_length_front', 
        'wall_thickness', 'parapet_height_m', 'overhang_length_u', 'random_noise',
        'wall_fenestration_per_n', 'wall_fenestration_per_s', 
        'wall_fenestration_per_e', 'wall_fenestration_per_w'
    ]
    categorical_features = [
        'archetype', 'occupany_u', 'building_urban_setting', 'building_position_on_street', 
        'roof_shape_u', 'roof_slope_u', 'construction_type_u', 'mwfrs_u_wall', 
        'mwfrs_u_roof', 'structural_wall_system_u', 'foundation_type_u', 
        'wall_substrate_u', 'wall_cladding_u', 'roof_system_u', 'roof_substrate_type_u', 
        'roof_cover_u', 'retrofit_present_u', 'retrofit_type_u'
    ]
    hazard_features = ['ef_numeric', 'distance_km']
    
    features = numeric_features + categorical_features + hazard_features
    # Filter
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y_raw = df['target']
    
    # Encode Target
    # Undamaged -> 0, Low -> 1, Significant -> 2 (alphabetical: Low, Significant, Undamaged -> 0, 1, 2?)
    # Wait, 'Low', 'Significant', 'Undamaged' sorted is 'Low', 'Significant', 'Undamaged'
    # replicate_analysis uses LabelEncoder.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Classes: {le.classes_}") 
    # Likely: 0='Low', 1='Significant', 2='Undamaged'
    # Need to map to textual classes for report
    
    # Preprocess
    num_subset = [c for c in features if c in numeric_features or c in hazard_features]
    cat_subset = [c for c in features if c in categorical_features]
    
    # Quick Pipeline mimic
    X_proc = X.copy()
    for col in num_subset:
        X_proc[col] = pd.to_numeric(X_proc[col], errors='coerce')
    
    X_proc[num_subset] = SimpleImputer(strategy='median').fit_transform(X_proc[num_subset])
    
    X_proc[cat_subset] = X_proc[cat_subset].astype(str)
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_proc[cat_subset] = ord_enc.fit_transform(X_proc[cat_subset])
    
    cat_idxs = [X_proc.columns.get_loc(c) for c in cat_subset]
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    reports = []
    
    for train_idx, val_idx in cv.split(X_proc, y):
        X_train, X_val = X_proc.iloc[train_idx], X_proc.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        try:
            smote = SMOTENC(categorical_features=cat_idxs, random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        except:
            X_res, y_res = X_train, y_train
            
        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', min_samples_leaf=2, random_state=42)
        clf.fit(X_res, y_res)
        
        y_pred = clf.predict(X_val)
        
        reports.append(classification_report(y_val, y_pred, output_dict=True, zero_division=0))

    # Average Report
    avg_report = {}
    keys = list(reports[0].keys()) # '0', '1', '2', 'macro avg', ...
    
    mapped_keys = {
        str(i): name for i, name in enumerate(le.classes_)
    }
    
    print("\n--- Averaged Classification Report ---")
    for k in keys:
        if k == 'accuracy': continue
        
        metri = reports[0][k].keys() # precision, recall, f1-score, support
        print(f"Class: {mapped_keys.get(k, k)}")
        for m in metri:
            vals = [r[k][m] for r in reports]
            print(f"  {m}: {np.mean(vals):.3f}")

    # 3. Read Stats CSV for p-values if needed
    try:
        stats_df = pd.read_csv('tornado_vulnerability_outputs/statistical_equivalence.csv')
        print("\n--- Statistical Tests ---")
        print(stats_df)
    except:
        print("Stats CSV not found.")

if __name__ == "__main__":
    get_stats()
