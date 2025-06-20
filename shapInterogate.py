import pandas as pd
import numpy as np
import os
import joblib
import logging
import sys
import matplotlib.pyplot as plt
import shap
import config


# --- Logging Configuration ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    """Sets up logging to append to the main pipeline log file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )


setup_logging()


# --- Main Analysis Script ---
def main():
    logging.info("--- Starting Script: 8_shap_interrogate.py ---")

    # Create the output directory if it doesn't exist
    os.makedirs(config.SHAP_DEPENDENCE_PLOTS_DIR, exist_ok=True)

    # --- Load Pre-calculated SHAP objects ---
    logging.info("Loading SHAP values and test samples...")
    try:
        all_shap_values = joblib.load(config.SHAP_VALUES_PATH)
        all_test_samples = joblib.load(config.SHAP_TEST_SAMPLES_PATH)
        logging.info("Successfully loaded SHAP data.")
    except FileNotFoundError:
        logging.error("SHAP data files not found. Please run '7_shap.py' first to generate them.")
        sys.exit(1)

    features_to_plot = config.FEATURES_FOR_DEPENDENCE_PLOTS
    if not features_to_plot:
        logging.warning("No features specified in 'FEATURES_FOR_DEPENDENCE_PLOTS' in config.py. Exiting.")
        return

    logging.info(f"Generating dependence plots for the following features: {features_to_plot}")

    # --- Generate Dependence Plots ---
    for model_key, shap_expl in all_shap_values.items():
        logging.info(f"\n--- Processing model: {model_key} ---")
        test_sample_df = all_test_samples[model_key]

        for feature in features_to_plot:
            if feature not in test_sample_df.columns:
                logging.warning(
                    f"  Feature '{feature}' not found in the test sample for model '{model_key}'. Skipping.")
                continue

            # Check if SHAP values are for multi-class classification
            is_multiclass = len(shap_expl.values.shape) == 3

            if is_multiclass:
                num_classes = shap_expl.values.shape[2]
                for i in range(num_classes):
                    logging.info(f"  Generating plot for feature '{feature}', class {i}...")

                    # Create a temporary Explanation object for the specific class
                    class_shap_values = shap_expl.values[:, :, i]
                    class_shap_expl = shap.Explanation(
                        values=class_shap_values,
                        base_values=shap_expl.base_values[:, i],
                        data=shap_expl.data,
                        feature_names=shap_expl.feature_names
                    )

                    shap.dependence_plot(
                        feature,
                        class_shap_expl.values,
                        test_sample_df,
                        show=False
                    )

                    plt.title(f"Dependence Plot: {feature}\nModel: {model_key} - Class: {i}")

                    filename = f"dependence_{model_key}_{feature}_class_{i}.png".replace(' ', '_').replace('(',
                                                                                                           '').replace(
                        ')', '')
                    save_path = os.path.join(config.SHAP_DEPENDENCE_PLOTS_DIR, filename)
                    plt.savefig(save_path)
                    plt.close()
            else:  # Single output models
                logging.info(f"  Generating plot for feature '{feature}'...")
                shap.dependence_plot(
                    feature,
                    shap_expl.values,
                    test_sample_df,
                    show=False
                )
                plt.title(f"Dependence Plot: {feature}\nModel: {model_key}")

                filename = f"dependence_{model_key}_{feature}.png".replace(' ', '_').replace('(', '').replace(')', '')
                save_path = os.path.join(config.SHAP_DEPENDENCE_PLOTS_DIR, filename)
                plt.savefig(save_path)
                plt.close()

    logging.info(f"\nDependence plots saved to '{config.SHAP_DEPENDENCE_PLOTS_DIR}' directory.")
    logging.info("--- Finished Script: 8_shap_interrogate.py ---")


if __name__ == '__main__':
    main()