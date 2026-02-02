import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5  # Total 25 iterations
OUTPUT_DIR = 'tornado_vulnerability_outputs_engineered'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Data Loading & Preprocessing ---
def load_and_preprocess_data():
    print("Loading data...")
    df_nash = pd.read_csv('updatedData/Nashville_Tornado_DataInput_Final_111425(in).csv')
    df_qs = pd.read_csv('updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv', encoding='latin1')
    
    # Normalize columns
    def normalize_cols(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
        return df

    df_nash = normalize_cols(df_nash)
    df_qs = normalize_cols(df_qs)
    
    # Add source identifier
    df_nash['dataset_source'] = 'Nashville'
    df_qs['dataset_source'] = 'QuadState'
    
    # Combine
    df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)
    
    # Clean unknowns
    def clean_unknowns(val):
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ['un', 'unknown', 'n/a', 'na']:
                return np.nan
        return val

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_unknowns)
    
    print(f"Combined shape: {df.shape}")
    return df

# --- 2. Feature Engineering ---
def engineer_features(df):
    print("Engineering features...")
    
    # Target Mapping
    def map_target(val):
        if pd.isna(val): return np.nan
        if val == 0: return 'Undamaged'
        if val == 1: return 'Low'
        if val >= 2: return 'Significant'
        return np.nan

    target_col = 'degree_of_damage_u'
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found.")
        return None
        
    df['target'] = df[target_col].apply(map_target)
    df = df.dropna(subset=['target'])
    
    # Hazard Features (EF & Distance)
    def parse_ef(val):
        if pd.isna(val): return np.nan
        s = str(val).strip().lower()
        if s == 'subef': return -1
        s = s.replace('ef', '')
        try: return int(float(s))
        except: return np.nan

    df['ef_numeric'] = df['tornado_ef'].apply(parse_ef)
    
    # Distance Calculation
    required_coords = ['tornado_start_lat', 'tornado_start_long', 'tornado_end_lat', 'tornado_end_long', 'latitude', 'longitude']
    if all(c in df.columns for c in required_coords):
        print("Calculating distance_km...")
        
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

        df['distance_km'] = df.apply(
            lambda row: point_line_segment_distance(
                row['longitude'], row['latitude'],
                row['tornado_start_long'], row['tornado_start_lat'],
                row['tornado_end_long'], row['tornado_end_lat']
            ), axis=1
        )
    
    # --- NEW ENGINEERED FEATURES ---
    print("Calculating engineered geometric features...")
    
    # Convert relevant columns to numeric first to handle strings/NaNs safely
    numeric_engineering_cols = [
        'buidling_height_m', 'wall_length_front', 'wall_length_side', 
        'wall_thickness', 'building_area_m2', 'parapet_height_m',
        'wall_fenestration_per_n', 'wall_fenestration_per_s',
        'wall_fenestration_per_e', 'wall_fenestration_per_w'
    ]
    for c in numeric_engineering_cols:
        if c in df.columns:
             # Remove commas if string
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(',', '').str.replace(' ', '')
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # 1. Aspect Ratio (Global Stability)
    # Height / min(width, length)
    df['min_dimension'] = df[['wall_length_front', 'wall_length_side']].min(axis=1)
    df['aspect_ratio'] = df['buidling_height_m'] / df['min_dimension']
    
    # 2. Wall Slenderness (Height (mm) / Thickness (mm))
    # Height is in meters, Thickness is in mm (usually, let's verify context - assumed mm based on "380mm" in text)
    df['wall_slenderness'] = (df['buidling_height_m'] * 1000) / df['wall_thickness']
    
    # 3. Roof-to-Wall Ratio (Uplift Area / Wall Area)
    # Wall Area approx = Perimeter * Height
    df['perimeter'] = 2 * (df['wall_length_front'] + df['wall_length_side'])
    df['total_wall_area'] = df['perimeter'] * df['buidling_height_m']
    df['roof_wall_ratio'] = df['building_area_m2'] / df['total_wall_area'] # building_area is footprint ~ roof area

    # 4. Parapet Slenderness
    # Parapet Height (m) -> mm / Thickness
    df['parapet_slenderness'] = (df['parapet_height_m'] * 1000) / df['wall_thickness']
    
    # 5. Mean Fenestration
    fenestration_cols = ['wall_fenestration_per_n', 'wall_fenestration_per_s', 
                         'wall_fenestration_per_e', 'wall_fenestration_per_w']
    # Use existing columns
    valid_fen = [c for c in fenestration_cols if c in df.columns]
    if valid_fen:
        df['mean_fenestration'] = df[valid_fen].mean(axis=1)
    else:
        df['mean_fenestration'] = np.nan
        
    # 6. Plan Aspect Ratio (Elongation)
    # max(L, W) / min(L, W)
    df['max_dimension'] = df[['wall_length_front', 'wall_length_side']].max(axis=1)
    df['plan_aspect_ratio'] = df['max_dimension'] / df['min_dimension']
    
    # Handle infinite ratios (div by zero)
    eng_features = ['aspect_ratio', 'wall_slenderness', 'roof_wall_ratio', 
                    'parapet_slenderness', 'mean_fenestration', 'plan_aspect_ratio']
    
    for f in eng_features:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
        
    # Guardrail
    np.random.seed(RANDOM_STATE)
    df['random_noise'] = np.random.randn(len(df))
    
    return df

def get_feature_sets():
    numeric_features = [
        'number_stories', 'year_built_u', 'building_area_m2', 'buidling_height_m', 
        'wall_length_side', 'wall_length_front', 
        'wall_thickness', 'parapet_height_m', 'overhang_length_u', 'random_noise', 'distance_km',
        # Engineered Features
        'aspect_ratio', 'wall_slenderness', 'roof_wall_ratio', 
        'parapet_slenderness', 'mean_fenestration', 'plan_aspect_ratio'
    ]
    
    categorical_features = [
        'archetype', 'occupany_u', 'building_urban_setting', 'building_position_on_street', 
        'roof_shape_u', 'roof_slope_u', 'construction_type_u', 'mwfrs_u_wall', 
        'mwfrs_u_roof', 'structural_wall_system_u', 'foundation_type_u', 
        'wall_substrate_u', 'wall_cladding_u', 'roof_system_u', 'roof_substrate_type_u', 
        'roof_cover_u', 'retrofit_present_u', 'retrofit_type_u'
    ]
    
    hazard_features = ['ef_numeric']
    
    numeric_features = [f for f in numeric_features if 'damage' not in f.lower()]
    categorical_features = [f for f in categorical_features if 'damage' not in f.lower()]
    hazard_features = [f for f in hazard_features if 'damage' not in f.lower()]
    
    return numeric_features, categorical_features, hazard_features

# --- 3. Modeling & Evaluation ---
def get_models():
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', min_samples_leaf=2, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, 
                                     colsample_bytree=0.8, objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE)
    }
    return models

def run_analysis(df, numeric_cols, cat_cols, hazard_cols):
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    hazard_cols = [c for c in hazard_cols if c in df.columns]
    
    print(f"Numeric: {len(numeric_cols)}, Categorical: {len(cat_cols)}, Hazard: {len(hazard_cols)}")
    
    settings = {
        'Hazard-Neutral-Engineered': numeric_cols + cat_cols,
    }
    
    results = []
    perm_results = []
    
    for setting_name, features in settings.items():
        print(f"\nRunning {setting_name}...")
        X = df[features]
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df['target']), index=df.index)
        
        cat_indices = [i for i, col in enumerate(features) if col in cat_cols]
        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        X_processed = X.copy()
        num_subset = list(set([c for c in features if c in numeric_cols or c in hazard_cols]))
        
        for col in num_subset:
            X_processed[col] = pd.to_numeric(X[col], errors='coerce')
        
        valid_num_subset = [c for c in num_subset if not X_processed[c].isna().all()]
        invalid_cols = list(set(num_subset) - set(valid_num_subset))
        if invalid_cols:
             X_processed = X_processed.drop(columns=invalid_cols)
             features = [f for f in features if f not in invalid_cols]
             num_subset = valid_num_subset
            
        transformed_data = numeric_transformer.fit_transform(X_processed[num_subset])
        X_processed[num_subset] = transformed_data
        
        cat_subset = [c for c in features if c in cat_cols]
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_processed[cat_subset] = enc.fit_transform(X[cat_subset].astype(str))
        
        cat_idxs = [X_processed.columns.get_loc(c) for c in cat_subset]
        
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
        models = get_models()
        
        for model_name, model in models.items():
            print(f"  Model: {model_name}")
            for i, (train_idx, val_idx) in enumerate(cv.split(X_processed, y)):
                X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    smote = SMOTENC(categorical_features=cat_idxs, random_state=RANDOM_STATE)
                    X_res, y_res = smote.fit_resample(X_train, y_train)
                except:
                    X_res, y_res = X_train, y_train
                
                clf = clone(model)
                clf.fit(X_res, y_res)
                y_pred = clf.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='macro')
                acc_baseline = accuracy_score(y_val, y_pred)
                
                results.append({
                    'Setting': setting_name, 'Model': model_name, 'Fold': i,
                    'MacroF1': f1, 'Accuracy': acc_baseline
                })
                
                # Permutation Importance (Subset of iterations to save time if needed, but requested 25)
                # We will do perm importance for every fold as requested.
                for feat in X_val.columns:
                    original_feat = X_val[feat].values.copy()
                    X_val[feat] = np.random.permutation(original_feat)
                    y_pred_perm = clf.predict(X_val)
                    acc_perm = accuracy_score(y_val, y_pred_perm)
                    X_val[feat] = original_feat
                    delta_acc = acc_baseline - acc_perm
                    
                    perm_results.append({
                        'Setting': setting_name, 'Model': model_name, 'Feature': feat,
                        'Fold': i, 'Decrease_in_Accuracy': delta_acc
                    })

    return pd.DataFrame(results), pd.DataFrame(perm_results)

def plot_permutation_importance(perm_df, output_dir):
    print("\nGenerating Plots...")
    for setting in perm_df['Setting'].unique():
        subset = perm_df[perm_df['Setting'] == setting]
        if subset.empty: continue
        
        # Filter > Noise
        important_features = set()
        models = subset['Model'].unique()
        for model in models:
            model_data = subset[subset['Model'] == model]
            mean_imp = model_data.groupby('Feature')['Decrease_in_Accuracy'].mean()
            if 'random_noise' in mean_imp:
                noise_thresh = mean_imp['random_noise']
                feats_above_noise = mean_imp[(mean_imp > noise_thresh) & (mean_imp > 0)].index.tolist()
                important_features.update(feats_above_noise)
        
        if 'random_noise' in important_features: important_features.remove('random_noise')
        subset_filtered = subset[subset['Feature'].isin(important_features)]
        
        if subset_filtered.empty: continue
        
        avg_imp = subset_filtered.groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False)
        sorted_features = avg_imp.index.tolist()
        
        plt.figure(figsize=(12, 10))
        sns.boxplot(data=subset_filtered, y='Feature', x='Decrease_in_Accuracy', order=sorted_features, hue='Model', orient='h')
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f'Engineered Features Analysis - {setting}')
        plt.xlabel('Decrease in Accuracy')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/delta_accuracy_engineered.png')
        plt.close()

if __name__ == "__main__":
    df = load_and_preprocess_data()
    df = engineer_features(df)
    if df is not None:
        num_cols, cat_cols, haz_cols = get_feature_sets()
        results_df, perm_df = run_analysis(df, num_cols, cat_cols, haz_cols)
        results_df.to_csv(f'{OUTPUT_DIR}/model_performance.csv', index=False)
        perm_df.to_csv(f'{OUTPUT_DIR}/permutation_importance.csv', index=False)
        plot_permutation_importance(perm_df, OUTPUT_DIR)
        print("Done.")
