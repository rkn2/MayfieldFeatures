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
    precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import KFold, cross_validate, GridSearchCV


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


def main():
    logging.info(f"--- Starting Script: 4_modeling.py ---")
    os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)

    X_train = load_data(config.TRAIN_X_PATH, "training features")
    y_train = load_data(config.TRAIN_Y_PATH, "training target")
    X_test = load_data(config.TEST_X_PATH, "test features")
    y_test = load_data(config.Y_TEST_PATH, "test target")

    y_train_ravel = y_train.to_numpy().ravel()
    y_test_ravel = y_test.to_numpy().ravel()

    all_results = []
    best_estimators = {}
    all_predictions = {}  # Dictionary to store predictions for McNemar's test

    for threshold in config.CLUSTERING_THRESHOLDS_TO_TEST:
        feature_set_label = f"Clustered (Thresh={threshold})" if threshold is not None else "Original Features"
        logging.info(f"\n===== PROCESSING FEATURE SET: {feature_set_label} =====")

        selected_features = get_selected_features_by_clustering(X_train, threshold, config.CLUSTERING_LINKAGE_METHOD)
        X_train_fs = X_train[selected_features]
        X_test_fs = X_test.reindex(columns=X_train_fs.columns, fill_value=0)

        for model_name, model_template in config.MODELS_TO_BENCHMARK.items():
            logging.info(f"  --- Benchmarking Model: {model_name} ---")
            param_grid = config.PARAM_GRIDS.get(model_name, {})
            kf_cv = KFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_STATE)

            grid_search = GridSearchCV(estimator=model_template, param_grid=param_grid,
                                       scoring=config.GRIDSEARCH_SCORING_METRIC,
                                       cv=kf_cv, n_jobs=-1, error_score='raise')

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
                all_predictions[combo_key] = y_pred_test  # Store predictions

                for metric_name, scorer_name in config.METRICS_TO_EVALUATE.items():
                    if 'f1' in metric_name:
                        score = f1_score(y_test_ravel, y_pred_test, average=metric_name.split('_')[-1], zero_division=0)
                    elif 'precision' in metric_name:
                        score = precision_score(y_test_ravel, y_pred_test, average=metric_name.split('_')[-1],
                                                zero_division=0)
                    elif 'recall' in metric_name:
                        score = recall_score(y_test_ravel, y_pred_test, average=metric_name.split('_')[-1],
                                             zero_division=0)
                    else:
                        score = accuracy_score(y_test_ravel, y_pred_test)
                    result_row[f"Test {metric_name.replace('_', ' ').title()}"] = score

                all_results.append(result_row)

            except Exception as e:
                logging.error(f"    ERROR running {model_name} for {feature_set_label}: {e}")
                continue

    all_results_df = pd.DataFrame(all_results)

    # Log the full results dataframe
    logging.info("\n--- Full Model Performance Report ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        logging.info(all_results_df.to_string())

    all_results_df.to_csv(config.DETAILED_RESULTS_CSV, index=False, float_format='%.6f')
    joblib.dump(best_estimators, config.BEST_ESTIMATORS_PATH)

    logging.info(f"\nComprehensive performance results saved to: {config.DETAILED_RESULTS_CSV}")
    logging.info(f"Saved dictionary of best estimators to: {config.BEST_ESTIMATORS_PATH}")

    # --- McNemar's Test and Selection for Plotting ---
    logging.info("\n--- Starting McNemar's Test for Model Comparison ---")
    models_to_plot_cm = []
    if not all_results_df.empty:
        # Identify the top performing model
        top_model_row = all_results_df.loc[all_results_df['Test F1 Weighted'].idxmax()]
        top_model_key = f"{top_model_row['Model']}_{top_model_row['Feature Set Name']}"
        models_to_plot_cm.append(top_model_key)  # Add the best model to the plot list
        top_model_predictions = all_predictions[top_model_key]
        top_model_feature_set = top_model_row['Feature Set Name']

        logging.info(
            f"Top performing model is '{top_model_key}'. Comparing it with other models on the same feature set ('{top_model_feature_set}').")

        # Filter for other models run on the same feature set
        comparison_df = all_results_df[(all_results_df['Feature Set Name'] == top_model_feature_set) & (
                all_results_df['Model'] != top_model_row['Model'])]

        for _, row in comparison_df.iterrows():
            comparison_model_key = f"{row['Model']}_{row['Feature Set Name']}"
            comparison_model_predictions = all_predictions[comparison_model_key]

            logging.info(f"\n--- McNemar's Test: '{top_model_key}' vs. '{comparison_model_key}' ---")

            # Create the contingency table
            tb = mcnemar_table(y_target=y_test_ravel,
                               y_model1=top_model_predictions,
                               y_model2=comparison_model_predictions)

            logging.info("Contingency Table:")
            logging.info(f"                  | Model 2 Correct | Model 2 Incorrect")
            logging.info(f"------------------|-----------------|------------------")
            logging.info(f"Model 1 Correct   |      {tb[0, 0]:<6}     |      {tb[0, 1]:<6}")
            logging.info(f"Model 1 Incorrect |      {tb[1, 0]:<6}     |      {tb[1, 1]:<6}")

            # Perform the test
            chi2, p = mcnemar(ary=tb, corrected=True)

            logging.info(f"  McNemar's test: chi-squared = {chi2:.4f}, p-value = {p:.4f}")

            # Interpret the result and decide whether to plot the confusion matrix
            if p < 0.05:
                logging.info("  Result: The difference in error rates is statistically significant (p < 0.05).")
            else:
                logging.info(
                    "  Result: The difference in error rates is not statistically significant (p >= 0.05). Adding to plot list.")
                models_to_plot_cm.append(comparison_model_key)

    # --- Generate Performance Bar Chart ---
    logging.info("\n--- Generating Performance Bar Chart ---")
    plt.style.use(config.VISUALIZATION['plot_style'])

    top_performers = all_results_df[all_results_df['Test F1 Weighted'] > config.PERFORMANCE_THRESHOLD_FOR_PLOT].copy()

    if not top_performers.empty:
        top_performers['Model (Feature Set)'] = top_performers['Model'] + ' (' + top_performers[
            'Feature Set Name'] + ')'
        top_performers = top_performers.sort_values(by='Test F1 Weighted', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Test F1 Weighted', y='Model (Feature Set)', data=top_performers,
                    palette=config.VISUALIZATION['main_palette'])
        plt.title(f"Top Performing Model Combinations (Test F1 Weighted > {config.PERFORMANCE_THRESHOLD_FOR_PLOT})")
        plt.xlabel('Test F1 Weighted Score')
        plt.ylabel('Model Combination')
        plt.xlim(left=min(0.8, top_performers['Test F1 Weighted'].min() * 0.98))
        plt.tight_layout()

        barchart_filename = "top_performers_f1_score_barchart.png"
        plt.savefig(os.path.join(config.BASE_RESULTS_DIR, barchart_filename))
        plt.close()
        logging.info(f"  Saved performance bar chart to {barchart_filename}")
    else:
        logging.warning(
            f"No models found with F1 score > {config.PERFORMANCE_THRESHOLD_FOR_PLOT} to generate a bar chart.")

    # --- Generate Confusion Matrices for Best and Statistically Similar Models ---
    logging.info("\n--- Generating Confusion Matrices ---")

    unique_models_to_plot = list(set(models_to_plot_cm))
    logging.info(f"Will generate confusion matrices for the following models: {unique_models_to_plot}")

    for combo_key in unique_models_to_plot:
        estimator = best_estimators[combo_key]
        y_pred = all_predictions[combo_key]

        cm = confusion_matrix(y_test_ravel, y_pred)

        # Log classification report
        report = classification_report(y_test_ravel, y_pred,
                                       target_names=[f"Class {c}" for c in np.unique(y_test_ravel)], zero_division=0)
        logging.info(f"\nClassification Report for {combo_key}:\n{report}")

        # Log confusion matrix
        logging.info(f"Confusion Matrix for {combo_key}:\n{cm}")

        # FIX: Check for .classes_ attribute, otherwise get labels from y_test
        if hasattr(estimator, 'classes_'):
            labels = estimator.classes_
        else:
            labels = np.unique(y_test_ravel)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f"Confusion Matrix for {combo_key}")
        cm_filename = f"confusion_matrix_{combo_key.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(config.BASE_RESULTS_DIR, cm_filename))
        plt.close(fig)
        logging.info(f"  Saved confusion matrix for {combo_key}")

    logging.info("\n--- Script Finished ---")


if __name__ == '__main__':
    main()