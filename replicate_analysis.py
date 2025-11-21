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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5  # Increased for robust testing
OUTPUT_DIR = 'tornado_vulnerability_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Data Loading & Preprocessing ---
def load_and_preprocess_data():
    print("Loading data...")
    df_nash = pd.read_excel('Nashville_Tornado_DataInput_Final_110725.xlsx')
    df_qs = pd.read_csv('QuadState_Tornado_DataInputv2.csv', encoding='latin1')
    
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
    # Note: In a real scenario, we'd align columns carefully. Here we assume key columns exist.
    # We'll focus on the columns mentioned in the report.
    common_cols = list(set(df_nash.columns) & set(df_qs.columns))
    df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)
    
    # --- Data Cleaning: Standardize Unknowns ---
    # User identified that Nashville uses "unknown"/"un" while QuadState uses blanks (NaN).
    # We convert all explicit "unknown" strings to np.nan so they are treated identically.
    # This prevents the model from seeing "unknown" and "nan" as two different categories.
    def clean_unknowns(val):
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ['un', 'unknown', 'n/a', 'na']:
                return np.nan
        return val

    # Apply to all columns (object type)
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
    
    # Hazard Features
    # 1. EF_numeric
    def parse_ef(val):
        if pd.isna(val): return np.nan
        s = str(val).strip().lower()
        if s == 'subef': return -1
        # Handle 'EF1', '1', 1
        s = s.replace('ef', '')
        try:
            return int(float(s))
        except:
            return np.nan

    df['ef_numeric'] = df['tornado_ef'].apply(parse_ef)
    
    # 2. distance_km (Point to Segment)
    # We need tornado_start_lat, tornado_start_long, tornado_end_lat, tornado_end_long
    # and latitude, longitude (building)
    
    required_coords = ['tornado_start_lat', 'tornado_start_long', 'tornado_end_lat', 'tornado_end_long', 'latitude', 'longitude']
    if all(c in df.columns for c in required_coords):
        print("Calculating distance_km...")
        
        def point_line_segment_distance(px, py, x1, y1, x2, y2):
            # Planar approximation
            # Convert lat/lon to km
            # Mean lat for scaling
            mean_lat = np.mean([y1, y2, py])
            lat_scale = 111.0
            lon_scale = 111.0 * np.cos(np.radians(mean_lat))
            
            # Scale coordinates to km
            px_km, py_km = px * lon_scale, py * lat_scale
            x1_km, y1_km = x1 * lon_scale, y1 * lat_scale
            x2_km, y2_km = x2 * lon_scale, y2 * lat_scale
            
            # Vector math for point to segment distance
            # Segment vector
            dx = x2_km - x1_km
            dy = y2_km - y1_km
            
            if dx == 0 and dy == 0:
                return np.sqrt((px_km - x1_km)**2 + (py_km - y1_km)**2)
            
            # Project point onto line (parameter t)
            t = ((px_km - x1_km) * dx + (py_km - y1_km) * dy) / (dx*dx + dy*dy)
            
            # Clamp t to segment [0, 1]
            t = np.clip(t, 0, 1)
            
            # Closest point
            closest_x = x1_km + t * dx
            closest_y = y1_km + t * dy
            
            dist = np.sqrt((px_km - closest_x)**2 + (py_km - closest_y)**2)
            return dist

        # Apply to DataFrame
        # Vectorized approach would be faster but apply is easier to write/read for now
        df['distance_km'] = df.apply(
            lambda row: point_line_segment_distance(
                row['longitude'], row['latitude'],
                row['tornado_start_long'], row['tornado_start_lat'],
                row['tornado_end_long'], row['tornado_end_lat']
            ), axis=1
        )
    else:
        print("Warning: Coordinate columns missing. Cannot calculate distance_km.")

    # Guardrail
    np.random.seed(RANDOM_STATE)
    df['random_noise'] = np.random.randn(len(df))
    
    return df

def get_feature_sets():
    # Define lists based on report
    numeric_features = [
        'number_stories', 'year_built_u', 'building_area_m2', 'buidling_height_m', 
        'wall_length_side', 'wall_length_front', 
        'wall_thickness', 'parapet_height_m', 'overhang_length_u', 'random_noise',
        # Added Fenestration Features (Percentage of Openings)
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
    
    return numeric_features, categorical_features, hazard_features

# --- 3. Modeling & Evaluation ---
def get_models():
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', min_samples_leaf=2, random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
        'RidgeClassifier': RidgeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
        'LinearSVC': LinearSVC(class_weight='balanced', random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, 
                                     colsample_bytree=0.8, objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE)
    }
    return models

def run_analysis(df, numeric_cols, cat_cols, hazard_cols):
    
    # Filter columns that actually exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    hazard_cols = [c for c in hazard_cols if c in df.columns]
    
    print(f"Numeric: {len(numeric_cols)}, Categorical: {len(cat_cols)}, Hazard: {len(hazard_cols)}")
    
    settings = {
        'Hazard-Neutral': numeric_cols + cat_cols,
        'Hazard-Inclusive': numeric_cols + cat_cols + hazard_cols
    }
    
    results = []
    perm_results = []
    
    for setting_name, features in settings.items():
        print(f"\nRunning {setting_name}...")
        X = df[features]
        
        # Encode target for XGBoost
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df['target']), index=df.index)
        print(f"Target classes: {le.classes_}")
        
        # Identify categorical indices for SMOTENC
        # Note: SMOTENC needs indices of categorical features
        # We need to know which columns in X are categorical
        # Since X is a subset, we recalculate indices
        cat_indices = [i for i, col in enumerate(features) if col in cat_cols]
        
        # Preprocessing Pipeline
        # Numeric: Impute
        # Categorical: Ordinal Encode (for SMOTENC)
        
        # We need a custom setup because SMOTENC takes the whole array
        # So we impute/encode FIRST, then SMOTE, then Classifier
        
        # Define Preprocessor
        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, [c for c in features if c in numeric_cols or c in hazard_cols]), # Hazard are mostly numeric/ordinal
                ('cat', categorical_transformer, [c for c in features if c in cat_cols])
            ]
        )
        
        # But wait, ColumnTransformer reorders columns. We need to track cat_indices for SMOTENC after transformation?
        # Actually, if we put cat last, we know the indices.
        # Let's simplify: Transform everything to numeric-compatible format first.
        
        X_processed = X.copy()
        
        # Impute Numerics
        num_subset = list(set([c for c in features if c in numeric_cols or c in hazard_cols]))
        # Coerce to numeric, turning errors ('un', etc) into NaN
        for col in num_subset:
            X_processed[col] = pd.to_numeric(X[col], errors='coerce')
            
        # Remove columns that are all NaN from num_subset to avoid SimpleImputer dropping them
        valid_num_subset = [c for c in num_subset if not X_processed[c].isna().all()]
        
        # Drop the invalid numeric columns from X_processed
        invalid_cols = list(set(num_subset) - set(valid_num_subset))
        if invalid_cols:
            print(f"Dropping all-NaN columns: {invalid_cols}")
            X_processed = X_processed.drop(columns=invalid_cols)
            # Update feature lists
            features = [f for f in features if f not in invalid_cols]
            num_subset = valid_num_subset
            
        transformed_data = numeric_transformer.fit_transform(X_processed[num_subset])
        X_processed[num_subset] = transformed_data
        
        # Encode Categoricals
        cat_subset = [c for c in features if c in cat_cols]
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_processed[cat_subset] = enc.fit_transform(X[cat_subset].astype(str))
        
        # Get indices for SMOTENC
        cat_idxs = [X_processed.columns.get_loc(c) for c in cat_subset]
        
    # Update CV to RepeatedStratifiedKFold
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
        
        models = get_models()
        
        for model_name, model in models.items():
            print(f"  Model: {model_name}")
            
            for i, (train_idx, val_idx) in enumerate(cv.split(X_processed, y)):
                X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # SMOTENC
                try:
                    smote = SMOTENC(categorical_features=cat_idxs, random_state=RANDOM_STATE)
                    X_res, y_res = smote.fit_resample(X_train, y_train)
                except Exception as e:
                    X_res, y_res = X_train, y_train
                
                # Fit
                clf = clone(model)
                clf.fit(X_res, y_res)
                
                # Predict Baseline
                y_pred = clf.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='macro')
                acc_baseline = accuracy_score(y_val, y_pred)
                
                results.append({
                    'Setting': setting_name,
                    'Model': model_name,
                    'Fold': i,
                    'MacroF1': f1,
                    'Accuracy': acc_baseline
                })
                
                # Permutation Importance (Delta Accuracy)
                # We want (Permuted - Baseline). 
                # If feature is important, Permuted < Baseline, so Delta is Negative.
                for feat in X_val.columns:
                    # Save original column
                    original_feat = X_val[feat].values.copy()
                    
                    # Permute
                    X_val[feat] = np.random.permutation(original_feat)
                    
                    # Predict
                    y_pred_perm = clf.predict(X_val)
                    acc_perm = accuracy_score(y_val, y_pred_perm)
                    
                    # Restore
                    X_val[feat] = original_feat
                    
                    delta_acc = acc_perm - acc_baseline
                    
                    perm_results.append({
                        'Setting': setting_name,
                        'Model': model_name,
                        'Feature': feat,
                        'Fold': i,
                        'Delta_Accuracy': delta_acc
                    })

    return pd.DataFrame(results), pd.DataFrame(perm_results)

def plot_permutation_importance(perm_df, stats_df, output_dir):
    print("\nGenerating Permutation Importance Plots (Equivalent Models Only)...")
    
    for setting in perm_df['Setting'].unique():
        subset = perm_df[perm_df['Setting'] == setting]
        
        # Filter for equivalent models
        stats_subset = stats_df[stats_df['Setting'] == setting]
        if not stats_subset.empty:
            best_model = stats_subset['Best_Model'].iloc[0]
            # Models with p > 0.05 are statistically equivalent (fail to reject null)
            equiv_models = stats_subset[stats_subset['p_value'] > 0.05]['Model'].tolist()
            keep_models = [best_model] + equiv_models
            
            print(f"  [{setting}] Keeping models: {keep_models}")
            subset = subset[subset['Model'].isin(keep_models)]
            
        if subset.empty:
            continue
        
        # Calculate average importance for sorting
        avg_imp = subset.groupby('Feature')['Delta_Accuracy'].mean().sort_values()
        sorted_features = avg_imp.index.tolist()
        
        # Plot Overview (Boxplot)
        plt.figure(figsize=(12, 10))
        ax = sns.boxplot(data=subset, y='Feature', x='Delta_Accuracy', order=sorted_features, hue='Model', orient='h')
        plt.axvline(0, color='k', linestyle='--')
        
        # Highlight random_noise label in red
        for label in ax.get_yticklabels():
            if label.get_text() == 'random_noise':
                label.set_color('red')
                label.set_fontweight('bold')
                
        plt.title(f'Permutation Importance (Delta Accuracy) - {setting}')
        plt.xlabel('Delta Accuracy (Permuted - Baseline)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/delta_accuracy_{setting}.png')
        plt.close()
        
        # Grouped Plots by Model (Barplot with Highlight)
        for model in subset['Model'].unique():
            model_subset = subset[subset['Model'] == model]
            avg_imp_model = model_subset.groupby('Feature')['Delta_Accuracy'].mean().sort_values()
            sorted_features_model = avg_imp_model.index.tolist()
            
            # Create colors: Red for random_noise, Blue/Grey for others
            colors = ['red' if x == 'random_noise' else 'skyblue' for x in sorted_features_model]
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=model_subset, y='Feature', x='Delta_Accuracy', order=sorted_features_model, palette=colors, errorbar='ci')
            plt.axvline(0, color='k', linestyle='--')
            
            plt.title(f'Permutation Importance - {setting} - {model}')
            plt.xlabel('Delta Accuracy (Permuted - Baseline)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/perm_imp_{setting}_{model}.png')
            plt.close()

# --- 4. Statistical Tests ---
def perform_stat_tests(results_df):
    print("\nPerforming Statistical Tests...")
    stats_results = []
    
    # Aggregate fold scores for statistical testing
    # Since we have repeated CV, we have N_SPLITS * N_REPEATS scores per model
    
    for setting in results_df['Setting'].unique():
        subset = results_df[results_df['Setting'] == setting]
        
        # Calculate mean MacroF1 per model
        model_means = subset.groupby('Model')['MacroF1'].mean()
        best_model_name = model_means.idxmax()
        print(f"Best Model for {setting}: {best_model_name}")
        
        best_scores = subset[subset['Model'] == best_model_name]['MacroF1'].values
        
        for model in subset['Model'].unique():
            if model == best_model_name:
                continue
                
            scores = subset[subset['Model'] == model]['MacroF1'].values
            
            # Paired Wilcoxon
            # Ensure aligned folds (assuming order is preserved in results_df)
            # The results are appended in order of CV splits, so they should align if models are iterated outer loop
            # Wait, my loops: Setting -> Model -> CV.
            # So 'subset' has: Model A (Fold 1..N), Model B (Fold 1..N).
            # Yes, they align by index 0..N.
            
            stat, p_val = wilcoxon(best_scores - scores, alternative='two-sided')
            
            stats_results.append({
                'Setting': setting,
                'Model': model,
                'Best_Model': best_model_name,
                'p_value': p_val,
                'Diff_Mean': np.mean(best_scores) - np.mean(scores)
            })
            
    return pd.DataFrame(stats_results)

# --- Main ---
if __name__ == "__main__":
    df = load_and_preprocess_data()
    df = engineer_features(df)
    
    if df is not None:
        num_cols, cat_cols, haz_cols = get_feature_sets()
        results_df, perm_df = run_analysis(df, num_cols, cat_cols, haz_cols)
        
        print("\n--- Results Summary ---")
        summary = results_df.groupby(['Setting', 'Model'])['MacroF1'].agg(['mean', 'std']).reset_index()
        print(summary)
        
        results_df.to_csv(f'{OUTPUT_DIR}/model_performance_cv.csv', index=False)
        perm_df.to_csv(f'{OUTPUT_DIR}/permutation_importance.csv', index=False)
        
        stats_df = perform_stat_tests(results_df)
        print("\n--- Statistical Tests (vs Best) ---")
        print(stats_df)
        stats_df.to_csv(f'{OUTPUT_DIR}/statistical_equivalence.csv', index=False)
        
        plot_permutation_importance(perm_df, stats_df, OUTPUT_DIR)
        
        print(f"\nAnalysis complete. Outputs saved to {OUTPUT_DIR}")
