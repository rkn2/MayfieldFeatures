import pandas as pd
import numpy as np
import os

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
    
    return df

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

def engineer_features(df):
    print("Calculating distance and engineered features...")
    
    # Distance
    required_coords = ['tornado_start_lat', 'tornado_start_long', 'tornado_end_lat', 'tornado_end_long', 'latitude', 'longitude']
    if all(c in df.columns for c in required_coords):
        df['distance_km'] = df.apply(
            lambda row: point_line_segment_distance(
                row['longitude'], row['latitude'],
                row['tornado_start_long'], row['tornado_start_lat'],
                row['tornado_end_long'], row['tornado_end_lat']
            ), axis=1
        )
    
    # Engineered Geometric Features
    numeric_engineering_cols = [
        'buidling_height_m', 'wall_length_front', 'wall_length_side', 
        'wall_thickness', 'building_area_m2', 'parapet_height_m',
        'wall_fenestration_per_n', 'wall_fenestration_per_s',
        'wall_fenestration_per_e', 'wall_fenestration_per_w'
    ]
    for c in numeric_engineering_cols:
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
    
    fenestration_cols = ['wall_fenestration_per_n', 'wall_fenestration_per_s', 
                         'wall_fenestration_per_e', 'wall_fenestration_per_w']
    valid_fen = [c for c in fenestration_cols if c in df.columns]
    if valid_fen:
        df['mean_fenestration'] = df[valid_fen].mean(axis=1)
    else:
        df['mean_fenestration'] = np.nan
        
    df['max_dimension'] = df[['wall_length_front', 'wall_length_side']].max(axis=1)
    df['plan_aspect_ratio'] = df['max_dimension'] / df['min_dimension']
    
    # Replace inf
    eng_features = ['aspect_ratio', 'wall_slenderness', 'roof_wall_ratio', 
                    'parapet_slenderness', 'mean_fenestration', 'plan_aspect_ratio']
    for f in eng_features:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
        
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    df = engineer_features(df)
    
    # Drop "unnamed" artifact columns
    unnamed_cols = [c for c in df.columns if c.lower().startswith('unnamed')]
    if unnamed_cols:
        print(f"Dropping {len(unnamed_cols)} unnamed artifact columns...")
        df = df.drop(columns=unnamed_cols)
    
    output_path = 'updatedData/forYishuang.csv'
    df.to_csv(output_path, index=False)
    print(f"File saved to {output_path}")
