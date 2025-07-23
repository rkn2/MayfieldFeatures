# config.py

import os
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import mord

# --- GENERAL ---
RANDOM_STATE = 42
PROBLEM_TYPE = 'regression'  # << CHANGE THIS to 'classification' to run the original analysis

# --- PATHS ---
DATA_DIR = 'processed_ml_data'
BASE_RESULTS_DIR = 'clustering_performance_results'
SHAP_RESULTS_DIR = 'shap_results_top_performers'
SHAP_DEPENDENCE_PLOTS_DIR = 'shap_dependence_plots'
REPORT_DIR = 'reports'
INPUT_CSV_PATH = 'QuadState_Tornado_DataInput_Categorical.csv'
CLEANED_CSV_PATH = 'cleaned_data_categorical_latlong.csv'
PIPELINE_LOG_PATH = 'pipeline.log'
REPORT_FILENAME = 'pipeline_visual_report.pdf'
DETAILED_RESULTS_CSV = os.path.join(BASE_RESULTS_DIR, 'clustering_performance_detailed_results.csv')
BEST_ESTIMATORS_PATH = os.path.join(BASE_RESULTS_DIR, 'best_estimators_per_combo.pkl')
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
PREPROCESSOR_PATH = os.path.join(DATA_DIR, 'preprocessor.pkl')
SHAP_VALUES_PATH = os.path.join(SHAP_RESULTS_DIR, 'all_shap_values.pkl')
SHAP_TEST_SAMPLES_PATH = os.path.join(SHAP_RESULTS_DIR, 'all_test_samples.pkl')

# --- DATA CLEANING ---
TARGET_COLUMN_FOR_NAN_DROP = 'roof_structure_damage_u_per' #'degree_of_damage_u'
LOW_VARIATION_THRESHOLD = 1
KEYWORDS_TO_DROP = ['photos', 'details', 'prop_', '_unc']
SPECIFIC_COLUMNS_TO_DROP = [
    'completed_by', 'damage_status', 'ref# (DELETE LATER)', 'complete_address',
    'building_name_listing', 'building_name_current', 'notes', 'tornado_name',
    'tornado_EF', 'tornado_start_lat', 'tornado_start_long', 'tornado_end_lat',
    'tornado_end_long', 'national_register_listing_year', 'town',
    'located_in_historic_district', 'hazards_present_u',
    'latitude', 'longitude'
]
COLUMNS_FOR_VALUE_REPLACEMENT = {
    'wall_thickness': {'un': '', 'not_applicable': 0},
    'overhang_length_u': {'un': '', 'not_applicable': 0},
    'parapet_height_m': {'un': '', 'not_applicable': 0}
}

# --- PREPROCESSING ---
TARGET_COLUMN = TARGET_COLUMN_FOR_NAN_DROP
TEST_SIZE = 0.2
BALANCING_METHOD = 'SMOTE'

# Set to True to apply a transformation (e.g., log) to the regression target
APPLY_TARGET_TRANSFORMATION = True
# The method to use. 'log1p' is recommended as it handles zero values.
TARGET_TRANSFORMATION_METHOD = 'log1p'

# --- CLASSIFICATION SPECIFIC SETTINGS ---
CLASSIFICATION_SETTINGS = {
    'REDUCE_CLASSES_STRATEGY': 'B',
    'CLASS_MAPPINGS': {
        'A': {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3},
        'B': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2}
    }
}


# --- New settings for Recursive Feature Elimination with Cross-Validation ---
PERFORM_RFECV = True              # Set to True to run RFECV, False to skip
RFECV_MIN_FEATURES = 15           # The minimum number of features to consider
RFECV_CV_FOLDS = 5                # Number of folds for cross-validation
RFECV_SCORING_METRIC = 'f1_macro' if PROBLEM_TYPE == 'classification' else 'r2'
RFECV_STEP = 1                    # How many features to remove at each step
RFECV_USE_1SE_RULE = True # SET TO True TO FIND THE SIMPLEST MODEL WITHIN 1 S.E. OF THE BEST

KEYWORDS_TO_REMOVE_FROM_X = [
    'damage', 'status_u', 'exist', 'demolish', 'failure', 'after'
]

# --- MODELING & EVALUATION ---
CLUSTERING_THRESHOLDS_TO_TEST = [None]
CLUSTERING_LINKAGE_METHOD = 'average'
N_SPLITS_CV = 5
GRIDSEARCH_SCORING_METRIC = 'f1_macro'
PRIMARY_METRIC_COLUMN = 'Test F1 Macro' # ADD this line
PERFORMANCE_THRESHOLD_FOR_PLOT = 0.65
PERMUTATION_SCORING_AVERAGE = 'macro' # ADD this line
N_PERMUTATION_REPEATS = 100  # Added this line
P_VALUE_THRESHOLD = 0.05    # Added this line

METRICS = {
    'classification': {
        'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted', 'f1_macro': 'f1_macro',
        'precision_weighted': 'precision_weighted', 'recall_weighted': 'recall_weighted'
    },
    'regression': {
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
}

METRICS_TO_EVALUATE = METRICS[PROBLEM_TYPE]

GRIDSEARCH_SCORING_METRIC = 'f1_macro' if PROBLEM_TYPE == 'classification' else 'r2'
PRIMARY_METRIC_COLUMN = 'Test F1 Macro' if PROBLEM_TYPE == 'classification' else 'Test R2'
PERMUTATION_SCORING_METRIC = 'f1_macro' if PROBLEM_TYPE == 'classification' else 'r2'

MODELS = {
    'classification': {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
        "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
        "Ordinal Logistic (AT)": mord.LogisticAT(),
        "Ordinal Ridge": mord.OrdinalRidge(),
        "Ordinal LAD": mord.LAD()
    },
    'regression': {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost Regressor": xgb.XGBRegressor(random_state=RANDOM_STATE),
        "LightGBM Regressor": lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1)
    }
}
MODELS_TO_BENCHMARK = MODELS[PROBLEM_TYPE]

PARAM_GRIDS_ALL = {
    'classification': {
        "Logistic Regression": {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]},
        "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 8], 'min_samples_leaf': [10, 15]},
        # ... other classification grids
    },
    'regression': {
        "Ridge": {'alpha': [0.1, 1.0, 10.0, 100.0]},
        "Lasso": {'alpha': [0.01, 0.1, 1.0, 10.0]},
        "Decision Tree Regressor": {'max_depth': [3, 5, 8], 'min_samples_leaf': [10, 20]},
        "Random Forest Regressor": {'n_estimators': [100, 150], 'max_depth': [6, 8], 'min_samples_leaf': [5, 10]},
        # ... other regression grids
    }
}
PARAM_GRIDS = PARAM_GRIDS_ALL[PROBLEM_TYPE]

# --- VISUALIZATION ---
# Define a consistent color scheme for all plots
VISUALIZATION = {
    'main_palette': 'viridis',  # A good choice for sequential data (e.g., bar charts)
    'diverging_palette': 'coolwarm', # Good for heatmaps or SHAP plots where values diverge from a center
    'plot_style': 'seaborn-v0_8-white' # A clean, professional plot style
}

# --- SHAP INTERROGATION ---
# List of features you want to generate dependence plots for.
# Use the actual column names from the processed data.
FEATURES_FOR_DEPENDENCE_PLOTS = [
    'num__building_area_m2',
    'num__year_built_u',
    'num__buidling_height_m',
    'num__wall_length_front'
]