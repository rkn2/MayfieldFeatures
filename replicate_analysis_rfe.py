import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTENC
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
RANDOM_STATE = 42
OUTPUT_DIR = 'tornado_vulnerability_outputs_rfe'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def point_line_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0: return haversine_km(py, px, y1, x1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = np.clip(t, 0, 1)
    return haversine_km(py, px, y1 + t * dy, x1 + t * dx)

def load_and_preprocess():
    print("Loading and preparing data...")
    df_nash = pd.read_csv('updatedData/Nashville_Tornado_DataInput_Final_111425(in).csv')
    df_qs = pd.read_csv('updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv', encoding='latin1')
    
    def normalize_cols(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
        return df

    df_nash = normalize_cols(df_nash)
    df_qs = normalize_cols(df_qs)
    df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)
    
    def clean_unknowns(val):
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ['un', 'unknown', 'n/a', 'na']: return np.nan
        return val

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_unknowns)
        
    # Target
    def map_target(val):
        if pd.isna(val): return np.nan
        if val == 0: return 0 # Undamaged
        if val == 1: return 1 # Low
        if val >= 2: return 2 # Significant
        return np.nan
    df['target'] = df['degree_of_damage_u'].apply(map_target)
    df = df.dropna(subset=['target'])
    
    # Distance
    df['distance_km'] = df.apply(
        lambda row: point_line_segment_distance(
            row['longitude'], row['latitude'],
            row['tornado_start_long'], row['tornado_start_lat'],
            row['tornado_end_long'], row['tornado_end_lat']
        ), axis=1
    )
    
    # Engineered
    engineering_cols = [
        'buidling_height_m', 'wall_length_front', 'wall_length_side', 
        'wall_thickness', 'building_area_m2', 'parapet_height_m',
        'wall_fenestration_per_n', 'wall_fenestration_per_s',
        'wall_fenestration_per_e', 'wall_fenestration_per_w'
    ]
    for c in engineering_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(',', '').str.replace(' ', '')
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['min_dimension'] = df[['wall_length_front', 'wall_length_side']].min(axis=1)
    df['aspect_ratio'] = df['buidling_height_m'] / df['min_dimension']
    df['wall_slenderness'] = (df['buidling_height_m'] * 1000) / df['wall_thickness']
    df['perimeter'] = 2 * (df['wall_length_front'] + df['wall_length_side'])
    df['total_wall_area'] = df['perimeter'] * df['buidling_height_m']
    df['roof_wall_ratio'] = df['building_area_m2'] / df['total_wall_area']
    df['parapet_slenderness'] = (df['parapet_height_m'] * 1000) / df['wall_thickness']
    
    fen_cols = ['wall_fenestration_per_n', 'wall_fenestration_per_s', 'wall_fenestration_per_e', 'wall_fenestration_per_w']
    df['mean_fenestration'] = df[fen_cols].mean(axis=1)
    df['max_dimension'] = df[['wall_length_front', 'wall_length_side']].max(axis=1)
    df['plan_aspect_ratio'] = df['max_dimension'] / df['min_dimension']
    
    for f in ['aspect_ratio', 'wall_slenderness', 'roof_wall_ratio', 'parapet_slenderness', 'mean_fenestration', 'plan_aspect_ratio']:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
        
    return df

def run_rfecv(df):
    num_cols = [
        'number_stories', 'year_built_u', 'building_area_m2', 'buidling_height_m', 
        'wall_length_side', 'wall_length_front', 'wall_thickness', 'parapet_height_m', 
        'overhang_length_u', 'distance_km', 'aspect_ratio', 'wall_slenderness', 
        'roof_wall_ratio', 'parapet_slenderness', 'mean_fenestration', 'plan_aspect_ratio'
    ]
    cat_cols = [
        'archetype', 'occupany_u', 'building_urban_setting', 'building_position_on_street', 
        'roof_shape_u', 'roof_slope_u', 'construction_type_u', 'mwfrs_u_wall', 
        'mwfrs_u_roof', 'structural_wall_system_u', 'foundation_type_u', 
        'wall_substrate_u', 'wall_cladding_u', 'roof_system_u', 'roof_substrate_type_u', 
        'roof_cover_u', 'retrofit_present_u', 'retrofit_type_u'
    ]
    
    features = num_cols + cat_cols
    X = df[features]
    y = df['target']
    
    # Preprocess
    X_processed = X.copy()
    num_imputer = SimpleImputer(strategy='median')
    X_processed[num_cols] = num_imputer.fit_transform(X[num_cols].apply(pd.to_numeric, errors='coerce'))
    
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_processed[cat_cols] = cat_encoder.fit_transform(X[cat_cols].astype(str))
    
    # Stratified split for SMOTE context
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(skf.split(X_processed, y))
    X_train, y_train = X_processed.iloc[train_idx], y.iloc[train_idx]
    
    # SMOTE to balance for RFE
    cat_idxs = [X_processed.columns.get_loc(c) for c in cat_cols]
    smote = SMOTENC(categorical_features=cat_idxs, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    print("Running RFECV with RandomForest...")
    # Using RF as base estimator
    estimator = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
    
    # RFECV: step=1 (remove one feature at a time), cv=3
    selector = RFECV(estimator, step=1, cv=3, scoring='f1_macro', n_jobs=-1)
    selector.fit(X_res, y_res)
    
    print(f"Optimal number of features: {selector.n_features_}")
    
    ranking = pd.DataFrame({
        'Feature': features,
        'Ranking': selector.ranking_,
        'Support': selector.support_
    }).sort_values('Ranking')
    
    ranking.to_csv(f'{OUTPUT_DIR}/rfe_ranking.csv', index=False)
    print(ranking)
    return ranking

if __name__ == "__main__":
    df = load_and_preprocess()
    run_rfecv(df)
