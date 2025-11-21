import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import StratifiedKFold

# Configuration
RANDOM_STATE = 42
OUTPUT_DIR = 'tornado_vulnerability_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data and preparing for SHAP analysis...")

# Load the processed data from the main analysis
df_nash = pd.read_excel('Nashville_Tornado_DataInput_Final_110725.xlsx')
df_qs = pd.read_csv('QuadState_Tornado_DataInputv2.csv', encoding='latin1')

# Normalize columns
def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
    return df

df_nash = normalize_cols(df_nash)
df_qs = normalize_cols(df_qs)

# Combine
df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)

# Engineer features (same as main analysis)
def map_target(val):
    if pd.isna(val): return np.nan
    if val == 0: return 'Undamaged'
    if val == 1: return 'Low'
    if val >= 2: return 'Significant'
    return np.nan

df['target'] = df['degree_of_damage_u'].apply(map_target)
df = df.dropna(subset=['target'])

# EF numeric
def parse_ef(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s == 'subef': return -1
    s = s.replace('ef', '')
    try:
        return int(float(s))
    except:
        return np.nan

df['ef_numeric'] = df['tornado_ef'].apply(parse_ef)

# Distance km (simplified version for SHAP - using actual calculation)
required_coords = ['tornado_start_lat', 'tornado_start_long', 'tornado_end_lat', 'tornado_end_long', 'latitude', 'longitude']
if all(c in df.columns for c in required_coords):
    def point_line_segment_distance(px, py, x1, y1, x2, y2):
        mean_lat = np.mean([y1, y2, py])
        lat_scale = 111.0
        lon_scale = 111.0 * np.cos(np.radians(mean_lat))
        
        px_km, py_km = px * lon_scale, py * lat_scale
        x1_km, y1_km = x1 * lon_scale, y1 * lat_scale
        x2_km, y2_km = x2 * lon_scale, y2 * lat_scale
        
        dx = x2_km - x1_km
        dy = y2_km - y1_km
        
        if dx == 0 and dy == 0:
            return np.sqrt((px_km - x1_km)**2 + (py_km - y1_km)**2)
        
        t = ((px_km - x1_km) * dx + (py_km - y1_km) * dy) / (dx*dx + dy*dy)
        t = np.clip(t, 0, 1)
        
        closest_x = x1_km + t * dx
        closest_y = y1_km + t * dy
        
        dist = np.sqrt((px_km - closest_x)**2 + (py_km - closest_y)**2)
        return dist

    df['distance_km'] = df.apply(
        lambda row: point_line_segment_distance(
            row['longitude'], row['latitude'],
            row['tornado_start_long'], row['tornado_start_lat'],
            row['tornado_end_long'], row['tornado_end_lat']
        ), axis=1
    )

# Define feature sets
numeric_features = [
    'number_stories', 'year_built_u', 'building_area_m2', 'buidling_height_m', 
    'first_floor_elevation_m', 'wall_length_side', 'wall_length_front', 
    'wall_thickness', 'parapet_height_m', 'overhang_length_u'
]

categorical_features = [
    'archetype', 'occupany_u', 'building_urban_setting', 'building_position_on_street', 
    'roof_shape_u', 'roof_slope_u', 'construction_type_u', 'mwfrs_u_wall', 
    'mwfrs_u_roof', 'structural_wall_system_u', 'foundation_type_u', 
    'wall_substrate_u', 'wall_cladding_u', 'roof_system_u', 'roof_substrate_type_u', 
    'roof_cover_u', 'retrofit_present_u', 'retrofit_type_u'
]

hazard_features = ['ef_numeric', 'distance_km']

# Use Hazard-Inclusive setting
all_features = numeric_features + categorical_features + hazard_features

# Filter to existing columns
all_features = [f for f in all_features if f in df.columns]

# Prepare data
X = df[all_features].copy()
le = LabelEncoder()
y = le.fit_transform(df['target'])

# Handle missing values and encode categoricals
for col in X.columns:
    if col in numeric_features or col in hazard_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    else:
        # One-hot encode categoricals for SHAP
        X[col] = X[col].astype(str).fillna('missing')

# One-hot encode all categoricals
X_encoded = pd.get_dummies(X, columns=[c for c in categorical_features if c in X.columns], drop_first=True)

# Drop any remaining NaN columns
X_encoded = X_encoded.dropna(axis=1, how='all')

print(f"Final feature shape: {X_encoded.shape}")
print(f"Target classes: {le.classes_}")

def run_shap_cv(model_class, model_name, **model_params):
    print(f"\nComputing SHAP values for {model_name} using 5-Fold Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Arrays to store results
    if model_name == 'XGBoost':
        # XGBoost output shape depends on objective, for multi:softprob it is (n_samples, n_classes, n_features) usually
        # But shap.TreeExplainer for XGBoost returns (n_samples, n_features, n_classes)
        all_shap_values = np.zeros((X_encoded.shape[0], X_encoded.shape[1], len(le.classes_)))
    else:
        all_shap_values = np.zeros((X_encoded.shape[0], X_encoded.shape[1], len(le.classes_)))
        
    test_indices_all = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_encoded, y)):
        print(f"Processing Fold {fold+1}/5...")
        
        X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model on this fold
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Compute SHAP values for the TEST set only
        explainer = shap.TreeExplainer(model)
        shap_values_fold = explainer.shap_values(X_test)
        
        # Store results
        if isinstance(shap_values_fold, list):
            # Convert to numpy array [n_samples, n_features, n_classes]
            shap_values_fold_np = np.stack(shap_values_fold, axis=-1)
        else:
            shap_values_fold_np = shap_values_fold
            
        all_shap_values[test_idx] = shap_values_fold_np
        test_indices_all.extend(test_idx)
        
    return all_shap_values

# --- Run SHAP for Random Forest ---
rf_params = {
    'n_estimators': 200, 
    'class_weight': 'balanced_subsample', 
    'min_samples_leaf': 2, 
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
shap_values_rf = run_shap_cv(RandomForestClassifier, 'Random Forest', **rf_params)

# --- Run SHAP for XGBoost ---
xgb_params = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': len(le.classes_),
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'mlogloss'
}
shap_values_xgb = run_shap_cv(XGBClassifier, 'XGBoost', **xgb_params)

# --- Save Comparison Results ---
print("\nSaving SHAP comparison statistics...")

def get_mean_abs_shap(shap_vals, class_idx):
    return np.abs(shap_vals[:, :, class_idx]).mean(axis=0)

shap_summary = pd.DataFrame({
    'Feature': X_encoded.columns,
    'RF_Mean_Abs_SHAP_Class2': get_mean_abs_shap(shap_values_rf, 2),
    'XGB_Mean_Abs_SHAP_Class2': get_mean_abs_shap(shap_values_xgb, 2)
})

# Calculate rank
shap_summary['RF_Rank'] = shap_summary['RF_Mean_Abs_SHAP_Class2'].rank(ascending=False)
shap_summary['XGB_Rank'] = shap_summary['XGB_Mean_Abs_SHAP_Class2'].rank(ascending=False)

shap_summary = shap_summary.sort_values('RF_Mean_Abs_SHAP_Class2', ascending=False)
shap_summary.to_csv(f'{OUTPUT_DIR}/shap_model_comparison.csv', index=False)

print("\n=== Top 10 Features Comparison (Class 2: Significant Damage) ===")
print(shap_summary.head(10)[['Feature', 'RF_Rank', 'XGB_Rank', 'RF_Mean_Abs_SHAP_Class2', 'XGB_Mean_Abs_SHAP_Class2']])

# Calculate Spearman correlation between rankings
spearman_corr = shap_summary['RF_Rank'].corr(shap_summary['XGB_Rank'], method='spearman')
print(f"\nSpearman Correlation between RF and XGBoost SHAP rankings: {spearman_corr:.4f}")

# --- Generate Plots for Random Forest (Primary Analysis) ---
# We use RF for the main plots as per the paper, but now we have validation
print("\nGenerating SHAP plots for Random Forest...")

# Re-fit a final model just for the 'base_values' (expected value) for the plot object
rf_final = RandomForestClassifier(**rf_params)
rf_final.fit(X_encoded, y)
explainer_final = shap.TreeExplainer(rf_final)

shap_values_obj = shap.Explanation(
    values=shap_values_rf,
    base_values=explainer_final.expected_value, 
    data=X_encoded.values,
    feature_names=X_encoded.columns
)

# Beeswarm for Class 2
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values_obj[:, :, 2], max_display=20, show=False)
plt.title('SHAP Summary: Features Driving Significant Damage (Class 2)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_beeswarm_class2.png', dpi=300, bbox_inches='tight')
plt.close()

# Beeswarm for Class 0
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values_obj[:, :, 0], max_display=20, show=False)
plt.title('SHAP Summary: Features Driving Survival (Class 0)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_beeswarm_class0.png', dpi=300, bbox_inches='tight')
plt.close()

# Dependence Plot
mean_abs_shap = np.abs(shap_values_rf[:, :, 2]).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-2:][::-1]
top_features = [X_encoded.columns[i] for i in top_features_idx]

if len(top_features) >= 2:
    plt.figure(figsize=(10, 8))
    shap.dependence_plot(
        top_features[0],
        shap_values_rf[:, :, 2],
        X_encoded,
        interaction_index=top_features[1],
        show=False
    )
    plt.title(f'SHAP Dependence: {top_features[0]} (colored by {top_features[1]})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shap_dependence_top2_class2.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nSHAP analysis complete. Outputs saved to {OUTPUT_DIR}/")
