# Data Round 2 Output Summary

## 1. Model Performance (F1 & Accuracy Scored)
| Setting | Model | MacroF1 | Accuracy |
| :--- | :--- | :--- | :--- |
| Hazard-Inclusive | RandomForest | 0.655 | 0.884 |
| Hazard-Inclusive | XGBoost | 0.643 | 0.883 |
| Hazard-Inclusive | LinearSVC | 0.630 | 0.831 |
| Hazard-Inclusive | RidgeClassifier | 0.628 | 0.814 |
| Hazard-Inclusive | DecisionTree | 0.612 | 0.832 |
| Hazard-Inclusive | LogisticRegression | 0.596 | 0.819 |
| Hazard-Neutral | RandomForest | 0.574 | 0.814 |
| Hazard-Neutral | XGBoost | 0.573 | 0.814 |
| Hazard-Neutral | LogisticRegression | 0.519 | 0.728 |
| Hazard-Neutral | RidgeClassifier | 0.513 | 0.701 |
| Hazard-Neutral | LinearSVC | 0.504 | 0.705 |
| Hazard-Neutral | DecisionTree | 0.492 | 0.732 |

## 2. Statistical Equivalence (Top Models)
| Setting | Model | Equivalent to Best? | p-value |
| :--- | :--- | :--- | :--- |
| **Hazard-Neutral** | XGBoost | **Yes** | 0.426 |
| Hazard-Neutral | LinearSVC | No | 0.002 |
| **Hazard-Inclusive** | XGBoost | **Yes** | 0.396 |
| **Hazard-Inclusive** | LinearSVC | **Yes** | 0.287 |
| **Hazard-Inclusive** | RidgeClassifier | **Yes** | 0.191 |

## 3. Top Permutation Importance (Hazard-Neutral, Random Forest)
*Features outperforming random noise baseline (0.0029)*

| Feature | Importance |
| :--- | :--- |
| **roof_substrate_type_u_ot** | 0.0096 |
| **parapet_height_m** | 0.0094 |
| **year_built_u** | 0.0075 |
| **wall_thickness_in** | 0.0059 |
| *random_noise_feature* | *0.0029* |

## 4. Top Permutation Importance (Hazard-Inclusive, Random Forest)
*Features outperforming random noise baseline (0.0032)*

| Feature | Importance |
| :--- | :--- |
| **ef_numeric** | 0.146 |
| **roof_substrate_type_u_ot** | 0.0093 |
| **distance_km** | 0.0058 |
| **wall_cladding_u_vinyl_ot** | 0.0039 |
| *random_noise_feature* | *0.0032* |

## 5. Top 10 SHAP Features (Class 2 - Significant Damage)
*Failure Drivers*

| Rank | Feature | Mean |SHAP| |
| :--- | :--- | :--- |
| 1 | ef_numeric | 0.094 |
| 2 | first_floor_elevation_m | 0.037 |
| 3 | distance_km | 0.025 |
| 4 | retrofit_type_u_not_applicable | 0.023 |
| 5 | wall_substrate_u_not_applicable | 0.023 |
| 6 | wall_substrate_u_un | 0.019 |
| 7 | building_area_m2 | 0.017 |
| 8 | buidling_height_m | 0.017 |
| 9 | year_built_u | 0.016 |
| 10 | wall_length_side | 0.016 |

## 6. Top 10 SHAP Features (Class 0 - Undamaged)
*Survival Drivers*

| Rank | Feature | Mean |SHAP| |
| :--- | :--- | :--- |
| 1 | retrofit_type_u_not_applicable | 0.033 |
| 2 | first_floor_elevation_m | 0.031 |
| 3 | ef_numeric | 0.024 |
| 4 | wall_substrate_u_not_applicable | 0.019 |
| 5 | distance_km | 0.018 |
| 6 | wall_cladding_u_brick | 0.012 |
| 7 | wall_substrate_u_un | 0.011 |
| 8 | building_area_m2 | 0.011 |
| 9 | buidling_height_m | 0.010 |
| 10 | year_built_u | 0.010 |

## 7. Top 10 SHAP Features (Class 1 - Low Damage)
*Transition Drivers*

| Rank | Feature | Mean |SHAP| |
| :--- | :--- | :--- |
| 1 | ef_numeric | 0.072 |
| 2 | distance_km | 0.025 |
| 3 | mwfrs_u_wall_wall_diaphragm_masonry | 0.020 |
| 4 | wall_cladding_u_brick | 0.015 |
| 5 | wall_thickness_in | 0.013 |
| 6 | first_floor_elevation_m | 0.012 |
| 7 | building_area_m2 | 0.011 |
| 8 | year_built_u | 0.011 |
| 9 | buidling_height_m | 0.009 |
| 10 | wall_substrate_u_not_applicable | 0.008 |

## 8. Comparison: Permutation vs SHAP (Class 2 Ranks)

| Feature | RF Rank (SHAP) | XGB Rank (SHAP) | Notes |
| :--- | :--- | :--- | :--- |
| ef_numeric | 1 | 1 | Consistent Top |
| retrofit_type (not applicable) | 2 | 16 | **SHAP reveals hidden importance** |
| distance_km | 3 | 2 | Consistent |
| wall_substrate (not_applicable) | 4 | 6 | Consistent |
| building_area | 5 | 5 | Consistent |
| year_built | 6 | 12 | Consistent |
| fenestration (west) | 7 | 4 | High in SHAP |
| height | 8 | 3 | Consistent |
| wall length | 9 | 11 | - |
| occupancy (residential) | 10 | 10 | - |

## Permutation Importance: Hazard-Neutral (All Equivalent Models)

### Model: RandomForest
| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | parapet_height_m | 0.0112 |
| 2 | roof_substrate_type_u | 0.0107 |
| 3 | wall_thickness | 0.0096 |
| 4 | year_built_u | 0.0091 |
| 5 | wall_length_side | 0.0059 |
| 6 | foundation_type_u | 0.0048 |
| 7 | *random_noise_feature* (BASELINE) | 0.0032 |
| 8 | occupany_u | 0.0021 |
| 9 | building_position_on_street | 0.0016 |
| 10 | buidling_height_m | 0.0016 |

### Model: XGBoost
| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | roof_substrate_type_u | 0.0085 |
| 2 | parapet_height_m | 0.0075 |
| 3 | year_built_u | 0.0059 |
| 4 | roof_slope_u | 0.0048 |
| 5 | number_stories | 0.0037 |
| 6 | retrofit_present_u | 0.0032 |
| 7 | roof_shape_u | 0.0027 |
| 8 | *random_noise_feature* (BASELINE) | 0.0027 |
| 9 | foundation_type_u | 0.0021 |
| 10 | occupany_u | 0.0021 |


## Permutation Importance: Hazard-Inclusive (All Equivalent Models)

### Model: RandomForest
| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | ef_numeric | 0.1198 |
| 2 | roof_substrate_type_u | 0.0107 |
| 3 | distance_km | 0.0096 |
| 4 | wall_cladding_u | 0.0059 |
| 5 | archetype | 0.0038 |
| 6 | wall_length_side | 0.0032 |
| 7 | year_built_u | 0.0027 |
| 8 | wall_substrate_u | 0.0016 |
| 9 | *random_noise_feature* (BASELINE) | 0.0016 |
| 10 | parapet_height_m | 0.0016 |

### Model: XGBoost
| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | ef_numeric | 0.1727 |
| 2 | roof_substrate_type_u | 0.0080 |
| 3 | number_stories | 0.0037 |
| 4 | wall_length_side | 0.0037 |
| 5 | distance_km | 0.0032 |
| 6 | wall_cladding_u | 0.0026 |
| 7 | building_area_m2 | 0.0016 |
| 8 | year_built_u | 0.0016 |
| 9 | wall_fenestration_per_s | 0.0016 |
| 10 | archetype | 0.0011 |

### Model: LinearSVC
| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | ef_numeric | 0.1828 |
| 2 | roof_substrate_type_u | 0.0209 |
| 3 | wall_length_side | 0.0187 |
| 4 | archetype | 0.0176 |
| 5 | wall_cladding_u | 0.0176 |
| 6 | wall_length_front | 0.0128 |
| 7 | wall_substrate_u | 0.0128 |
| 8 | structural_wall_system_u | 0.0113 |
| 9 | mwfrs_u_wall | 0.0059 |
| 10 | retrofit_present_u | 0.0053 |

### Model: RidgeClassifier
| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | ef_numeric | 0.1759 |
| 2 | roof_substrate_type_u | 0.0385 |
| 3 | wall_cladding_u | 0.0150 |
| 4 | retrofit_present_u | 0.0144 |
| 5 | roof_slope_u | 0.0064 |
| 6 | wall_substrate_u | 0.0059 |
| 7 | mwfrs_u_roof | 0.0043 |
| 8 | wall_length_side | 0.0032 |
| 9 | occupany_u | 0.0032 |
| 10 | foundation_type_u | 0.0016 |

