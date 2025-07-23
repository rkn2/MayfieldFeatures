import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import logging
import sys
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import config

from mlxtend.evaluate import mcnemar_table, mcnemar

from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_score, recall_score, accuracy_score, mean_squared_error, r2_score,
    mean_absolute_error
)
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV


def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)],
                        force=True)


setup_logging()


def load_data(file_path, description="data"):
    logging.info(f"Loading {description} from {file_path}...")
    try:
        return joblib.load(file_path)
    except Exception as e:
        logging.error(f"Error loading {description}: {e}", exc_info=True)
        sys.exit(1)


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    if distance_thresh is None or pd.isna(distance_thresh):
        return original_df.columns.tolist()

    feature_names = original_df.columns.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_df = associations(original_df, nom_nom_assoc='cramer', compute_only=True)['corr'].fillna(0)

    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    condensed_dist_mat = squareform(distance_mat, checks=False)
    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    cluster_labels = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')

    selected_features = []
    for i in range(1, np.max(cluster_labels) + 1):
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        if not cluster_indices: continue
        if len(cluster_indices) == 1:
            selected_features.append(feature_names[cluster_indices[0]])
        else:
            sum_abs_assoc = np.abs(assoc_df.iloc[cluster_indices, cluster_indices].values).sum(axis=1)
            representative_index = np.argmax(sum_abs_assoc)
            selected_features.append(feature_names[cluster_indices[representative_index]])

    return sorted(list(set(selected_features)))


def plot_regression_results(y_true, y_pred, model_key, results_dir):
    """
    Generates and saves a scatter plot for actual vs. predicted values for regression models.
    A good model will have points clustered closely around the diagonal red line.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predicted vs. Actual')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2, label='Ideal Fit (y=x)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs. Predicted Values for {model_key}')
    plt.legend()
    plt.grid(True)
    plot_filename = f"actual_vs_predicted_{model_key.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()
    logging.info(f"  Saved actual vs. predicted plot for {model_key} to {plot_filename}")


def main():
    logging.info(f"--- Starting Script: 4_modeling.py ({config.PROBLEM_TYPE.capitalize()}) ---")
    os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)

    X_train = load_data(config.TRAIN_X_PATH, "training features")
    y_train = load_data(config.TRAIN_Y_PATH, "training target")
    X_test = load_data(config.TEST_X_PATH, "test features")
    y_test = load_data(config.Y_TEST_PATH, "test target")

    y_train_ravel = y_train.to_numpy().ravel()
    y_test_ravel = y_test.to_numpy().ravel()

    all_results = []
    best_estimators = {}
    all_predictions = {}

    for threshold in config.CLUSTERING_THRESHOLDS_TO_TEST:
        feature_set_label = f"Clustered (Thresh={threshold})" if threshold is not None else "Original Features"
        logging.info(f"\n===== PROCESSING FEATURE SET: {feature_set_label} =====")

        selected_features = get_selected_features_by_clustering(X_train, threshold, config.CLUSTERING_LINKAGE_METHOD)
        X_train_fs = X_train[selected_features]
        X_test_fs = X_test.reindex(columns=X_train_fs.columns, fill_value=0)

        for model_name, model_template in config.MODELS_TO_BENCHMARK.items():
            logging.info(f"  --- Benchmarking Model: {model_name} ---")
            param_grid = config.PARAM_GRIDS.get(model_name, {})

            if config.PROBLEM_TYPE == 'classification':
                cv_strategy = StratifiedKFold(n_splits=config.N_SPLITS_CV, shuffle=True,
                                              random_state=config.RANDOM_STATE)
            else:  # Regression
                cv_strategy = KFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_STATE)

            grid_search = GridSearchCV(estimator=model_template, param_grid=param_grid,
                                       scoring=config.GRIDSEARCH_SCORING_METRIC,
                                       cv=cv_strategy, n_jobs=-1, error_score='raise')

            try:
                grid_search.fit(X_train_fs, y_train_ravel)
                best_estimator = grid_search.best_estimator_
                combo_key = f"{model_name}_{feature_set_label}"
                best_estimators[combo_key] = best_estimator

                result_row = {
                    "Model": model_name, "Feature Set Name": feature_set_label,
                    "Number of Features": len(selected_features), "Threshold Value": threshold,
                    "Best Params": str(grid_search.best_params_)
                }

                y_pred_test = best_estimator.predict(X_test_fs)
                all_predictions[combo_key] = y_pred_test

                # Calculate and store metrics based on problem type
                if config.PROBLEM_TYPE == 'classification':
                    for metric_name, scorer_name in config.METRICS_TO_EVALUATE.items():
                        average_method = scorer_name.split('_')[-1] if '_' in scorer_name else 'binary'
                        if 'f1' in scorer_name:
                            score = f1_score(y_test_ravel, y_pred_test, average=average_method, zero_division=0)
                        elif 'precision' in scorer_name:
                            score = precision_score(y_test_ravel, y_pred_test, average=average_method, zero_division=0)
                        elif 'recall' in scorer_name:
                            score = recall_score(y_test_ravel, y_pred_test, average=average_method, zero_division=0)
                        else:  # accuracy
                            score = accuracy_score(y_test_ravel, y_pred_test)
                        result_row[f"Test {metric_name.replace('_', ' ').title()}"] = score
                else:  # Regression
                    for metric_name in config.METRICS_TO_EVALUATE:
                        if metric_name == 'r2':
                            score = r2_score(y_test_ravel, y_pred_test)
                        elif metric_name == 'neg_mean_squared_error':
                            score = mean_squared_error(y_test_ravel, y_pred_test)
                        elif metric_name == 'neg_mean_absolute_error':
                            score = mean_absolute_error(y_test_ravel, y_pred_test)
                        result_row[f"Test {metric_name.replace('_', ' ').title()}"] = score

                all_results.append(result_row)

            except Exception as e:
                logging.error(f"    ERROR running {model_name} for {feature_set_label}: {e}", exc_info=True)
                continue

    all_results_df = pd.DataFrame(all_results)
    logging.info("\n--- Full Model Performance Report ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        logging.info(all_results_df.to_string())

    all_results_df.to_csv(config.DETAILED_RESULTS_CSV, index=False, float_format='%.6f')
    joblib.dump(best_estimators, config.BEST_ESTIMATORS_PATH)
    logging.info(f"\nComprehensive performance results saved to: {config.DETAILED_RESULTS_CSV}")
    logging.info(f"Saved dictionary of best estimators to: {config.BEST_ESTIMATORS_PATH}")

    # --- Performance Visualization and Model-Specific Reports ---
    plt.style.use(config.VISUALIZATION['plot_style'])

    if not all_results_df.empty:
        logging.info("\n--- Generating Performance Bar Chart ---")
        # For both R2 and F1, higher is better, so ascending is always False.
        # For metrics like MSE, you would set ascending=True.
        is_ascending = False
        top_performers = all_results_df.sort_values(by=config.PRIMARY_METRIC_COLUMN, ascending=is_ascending).head(10)

        plt.figure(figsize=(14, 8))
        sns.barplot(x=config.PRIMARY_METRIC_COLUMN, y='Model', data=top_performers, hue='Feature Set Name',
                    palette=config.VISUALIZATION['main_palette'], dodge=True)
        plt.title(f"Top Performing Model Combinations by {config.PRIMARY_METRIC_COLUMN}")
        plt.xlabel(f"{config.PRIMARY_METRIC_COLUMN} Score")
        plt.ylabel('Model Combination')
        plt.legend(title='Feature Set', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        barchart_filename = f"top_performers_{config.PROBLEM_TYPE}_barchart.png"
        plt.savefig(os.path.join(config.BASE_RESULTS_DIR, barchart_filename))
        plt.close()
        logging.info(f"  Saved performance bar chart to {barchart_filename}")

    # --- Generate Task-Specific Reports and Plots ---
    if config.PROBLEM_TYPE == 'classification':
        logging.info("\n--- Generating Classification Reports and Confusion Matrices ---")
        if not all_results_df.empty:
            # For this example, we'll simply plot the top 3 models based on the primary metric.
            # You can re-integrate the McNemar's test logic here if desired.
            models_to_plot_cm = all_results_df.sort_values(by=config.PRIMARY_METRIC_COLUMN, ascending=False).head(3)

            for _, row in models_to_plot_cm.iterrows():
                combo_key = f"{row['Model']}_{row['Feature Set Name']}"
                estimator = best_estimators[combo_key]
                y_pred = all_predictions[combo_key]

                report = classification_report(y_test_ravel, y_pred,
                                               target_names=[f"Class {c}" for c in np.unique(y_test_ravel)],
                                               zero_division=0)
                logging.info(f"\nClassification Report for {combo_key}:\n{report}")

                cm = confusion_matrix(y_test_ravel, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_ravel))
                fig, ax = plt.subplots(figsize=(8, 6))
                disp.plot(ax=ax, cmap='Blues')
                plt.title(f"Confusion Matrix for {combo_key}")
                cm_filename = f"confusion_matrix_{combo_key.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                plt.savefig(os.path.join(config.BASE_RESULTS_DIR, cm_filename))
                plt.close(fig)
                logging.info(f"  Saved confusion matrix for {combo_key}")

    else:  # Regression
        logging.info("\n--- Generating Regression Reports and Plots ---")
        top_regression_models = all_results_df.sort_values(by=config.PRIMARY_METRIC_COLUMN, ascending=False).head(3)
        for _, row in top_regression_models.iterrows():
            combo_key = f"{row['Model']}_{row['Feature Set Name']}"
            y_pred = all_predictions[combo_key]

            logging.info(f"\nRegression Report for {combo_key}:")
            logging.info(f"  R-squared: {r2_score(y_test_ravel, y_pred):.4f}")
            logging.info(f"  Mean Squared Error: {mean_squared_error(y_test_ravel, y_pred):.4f}")
            logging.info(f"  Mean Absolute Error: {mean_absolute_error(y_test_ravel, y_pred):.4f}")

            # << NEW >> Call the plotting function for each top model
            plot_regression_results(y_test_ravel, y_pred, combo_key, config.BASE_RESULTS_DIR)

    logging.info("\n--- Script Finished ---")


if __name__ == '__main__':
    main()