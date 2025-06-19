import pandas as pd
import numpy as np
import os
import joblib
import warnings
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import the configuration file

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
# Set the random seed for reproducibility
np.random.seed(config.RANDOM_STATE)


# --- Logging Configuration Setup ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    """Sets up logging to append to the main pipeline log file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # 'a' for append
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )


# Configure logging
setup_logging()

# --- Dynamic Importer for Balancing Methods ---
sampler_class = None
if config.BALANCING_METHOD:
    try:
        if config.BALANCING_METHOD == 'SMOTE':
            from imblearn.over_sampling import SMOTE

            sampler_class = SMOTE
        # Add other balancers like RandomOverSampler here if needed
        logging.info(f"Selected balancing method: {config.BALANCING_METHOD}")
    except ImportError:
        logging.error(f"Error: 'imbalanced-learn' not found for BALANCING_METHOD '{config.BALANCING_METHOD}'.")
        logging.error("Please install it: pip install imbalanced-learn")
        sys.exit()


def filter_features(df, keywords_to_remove):
    """Removes columns from the dataframe based on a list of keywords."""
    cols_to_drop = {col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords_to_remove)}
    logging.info(f"Filtering features. Removing columns based on keywords: {sorted(list(cols_to_drop))}")
    return df.drop(columns=list(cols_to_drop), errors='ignore')


# --- Main Preprocessing Script ---
def main():
    logging.info(f"--- Starting Script: 2_dataPreprocessing.py ---")

    # 1. Load Data
    logging.info(f"\nStep 1: Loading cleaned data from '{config.CLEANED_CSV_PATH}'...")
    try:
        df = pd.read_csv(config.CLEANED_CSV_PATH, low_memory=False)
        logging.info(f"  Successfully loaded. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"FATAL: Cleaned data file not found at {config.CLEANED_CSV_PATH}")
        sys.exit(1)

    # 2. Separate Target and Features
    logging.info(f"\nStep 2: Separating target ('{config.TARGET_COLUMN}') and features...")
    y = pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce').fillna(0).astype(int)
    X = df.drop(columns=[config.TARGET_COLUMN])

    # 3. Apply Class Reduction
    logging.info("\nStep 3: Applying class reduction...")
    if config.REDUCE_CLASSES_STRATEGY in config.CLASS_MAPPINGS:
        y = y.map(config.CLASS_MAPPINGS[config.REDUCE_CLASSES_STRATEGY])
        logging.info(f"  Applied class reduction strategy '{config.REDUCE_CLASSES_STRATEGY}'.")

    # 4. Filter Feature Set
    X = filter_features(X, config.KEYWORDS_TO_REMOVE_FROM_X)

    # 5. Split Data
    logging.info("\nStep 5: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE,
                                                        random_state=config.RANDOM_STATE, stratify=y)

    # 6. Preprocess Features
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough')

    X_train_processed = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out(),
                                     index=X_train.index)
    X_test_processed = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out(),
                                    index=X_test.index)

    # Step 7: Balance the Training Data (This now happens BEFORE feature selection)
    X_train_for_selection = X_train_processed
    y_train_for_selection = y_train

    if config.BALANCING_METHOD and sampler_class:
        logging.info(f"\nStep 7: Applying balancing method '{config.BALANCING_METHOD}' to the training data...")
        sampler = sampler_class(random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed, y_train)

        # The data to be used for the next step (feature selection) is now the balanced set
        X_train_for_selection = pd.DataFrame(X_train_resampled, columns=X_train_processed.columns)
        y_train_for_selection = y_train_resampled
        logging.info(f"  Shape of training data after balancing: {X_train_for_selection.shape}")
    else:
        logging.info("\nStep 7: Skipping data balancing as per config settings.")

    # Step 8: Perform Feature Selection on the (now balanced) training data
    # Initialize final datasets. They will be overwritten if RFECV is performed.
    X_train_final = X_train_for_selection
    y_train_final = y_train_for_selection
    X_test_final = X_test_processed # Start with the original processed test set

    if config.PERFORM_RFECV:
        logging.info(f"\nStep 8: Performing RFECV on the balanced data to find the optimal number of features...")
        logging.info(f"  - CV Folds: {config.RFECV_CV_FOLDS}, Scoring Metric: '{config.RFECV_SCORING_METRIC}'")

        estimator = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
        cv_strategy = StratifiedKFold(n_splits=config.RFECV_CV_FOLDS)

        rfecv_selector = RFECV(
            estimator=estimator,
            step=config.RFECV_STEP,
            cv=cv_strategy,
            scoring=config.RFECV_SCORING_METRIC,
            min_features_to_select=config.RFECV_MIN_FEATURES,
            n_jobs=-1
        )

        logging.info("  Fitting RFECV selector on balanced data... This may take some time.")
        # --- CRITICAL CHANGE: Fit on the balanced data ---
        rfecv_selector.fit(X_train_for_selection, y_train_for_selection)
        logging.info("  RFECV fitting complete.")

        optimal_feature_count = rfecv_selector.n_features_
        logging.info(f"  RFECV identified {optimal_feature_count} as the optimal number of features.")

        # --- NEW: Apply the "One Standard Error" Rule if enabled ---
        if config.RFECV_USE_1SE_RULE:
            logging.info("  Applying the 'One Standard Error' rule to find a simpler model.")

            cv_scores = rfecv_selector.cv_results_['mean_test_score']
            cv_scores_std = rfecv_selector.cv_results_['std_test_score']

            best_score_idx = np.argmax(cv_scores)
            best_score = cv_scores[best_score_idx]
            best_score_std = cv_scores_std[best_score_idx]

            performance_threshold = best_score - best_score_std
            logging.info(f"  Performance threshold (best score - 1 SE): {performance_threshold:.4f}")

            candidate_indices = np.where(cv_scores >= performance_threshold)[0]
            best_simpler_model_idx = candidate_indices[0]

            optimal_feature_count = best_simpler_model_idx + config.RFECV_MIN_FEATURES
            logging.info(
                f"  Found a simpler model with {optimal_feature_count} features that performs within one SE of the best model.")

        # --- PLOTTING RFECV RESULTS ---
        logging.info("  Generating RFECV performance plot...")
        plt.figure(figsize=(12, 8))
        plt.style.use(config.VISUALIZATION['plot_style'])
        x_axis_data = range(config.RFECV_MIN_FEATURES,
                            len(rfecv_selector.cv_results_['mean_test_score']) + config.RFECV_MIN_FEATURES)
        plt.plot(x_axis_data, rfecv_selector.cv_results_['mean_test_score'])
        plt.xlabel("Number of Features Selected")
        plt.ylabel(f"Cross-Validated Score ({config.RFECV_SCORING_METRIC.capitalize()})")
        plt.title("RFECV Performance vs. Number of Features")
        plt.axvline(x=optimal_feature_count, color='r', linestyle='--',
                    label=f'Optimal Features: {optimal_feature_count}')
        plt.legend()
        plt.grid(True)
        os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)
        output_plot_path = os.path.join(config.BASE_RESULTS_DIR, "rfecv_performance_plot.png")
        plt.savefig(output_plot_path)
        plt.close()
        logging.info(f"  Plot saved to: {output_plot_path}")

        # --- FINAL FEATURE SELECTION & APPLICATION ---
        logging.info(f"  Applying final selection of {optimal_feature_count} features...")
        final_selector = RFE(estimator=estimator, n_features_to_select=optimal_feature_count)
        final_selector.fit(X_train_for_selection, y_train_for_selection)

        selected_features_mask = final_selector.get_support()
        selected_features = X_train_for_selection.columns[selected_features_mask].tolist()

        random_feature_name = 'num__random_feature'
        if random_feature_name in X_train_for_selection.columns and random_feature_name not in selected_features:
            logging.info(f"  Forcing inclusion of '{random_feature_name}' for model sanity check.")
            selected_features.append(random_feature_name)

        # Overwrite the final datasets with the feature-selected versions
        X_train_final = X_train_for_selection[selected_features]
        X_test_final = X_test_processed[selected_features]  # Apply same features to original test set

        logging.info(f"  Final shape of X_train after selection: {X_train_final.shape}")
        logging.info(f"  Final shape of X_test after selection: {X_test_final.shape}")
    else:
        logging.info("\nStep 8: Skipping RFECV as per config settings.")

        # Step 9: Save the final data artifacts
    logging.info("\nStep 9: Saving final processed data...")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    joblib.dump(X_train_final, config.TRAIN_X_PATH)
    joblib.dump(y_train_final, config.TRAIN_Y_PATH)
    joblib.dump(X_test_final, config.TEST_X_PATH)
    joblib.dump(y_test, config.Y_TEST_PATH)  # y_test is never altered
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    logging.info("  All processed data artifacts have been saved successfully.")

    # Step 10: Log and Visualize Final Distributions
    logging.info("\nStep 10: Logging and visualizing data distributions...")

    logging.info(f"\nOriginal Data Distribution:\n{df[config.TARGET_COLUMN].value_counts().to_string()}")
    logging.info(f"\nTraining Data Distribution (Before Balancing):\n{y_train.value_counts().to_string()}")
    logging.info(
        f"\nFinal Training Data Distribution (After Balancing & Selection):\n{pd.Series(y_train_final).value_counts().to_string()}")
    logging.info(f"\nFinal Test Data Distribution:\n{y_test.value_counts().to_string()}")

    # Generate visualization
    plt.style.use(config.VISUALIZATION['plot_style'])
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    palette = config.VISUALIZATION['main_palette']

    sns.countplot(x=df[config.TARGET_COLUMN], ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title('Original Data Distribution')

    sns.countplot(x=y_train, ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title('Training Data (Before Balancing)')

    sns.countplot(x=y_train_final, ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title(f'Training Data (After {config.BALANCING_METHOD or "No"} Balancing)')

    sns.countplot(x=y_test, ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title('Test Data Distribution')

    plt.tight_layout()
    plt.savefig('data_distribution_summary.png')
    logging.info("  Saved data distribution summary plot to 'data_distribution_summary.png'.")

    logging.info(f"--- Finished Script: 2_dataPreprocessing.py ---")


if __name__ == '__main__':
    main()