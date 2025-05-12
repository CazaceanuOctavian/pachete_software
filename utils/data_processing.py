import pandas as pd
import numpy as np

def prepare_model_data(filtered_df):
    """
    Prepare data for modeling by processing and cleaning features
    """
    df_model_training = pd.DataFrame()
    
    # Process 5G and 4G fields
    if '5G' in filtered_df.columns:
        df_model_training['5G'] = filtered_df['5G'].fillna(0)
        df_model_training['5G'] = df_model_training['5G'].replace('Da', 1).replace('Nu', 0)
    
    if '4G' in filtered_df.columns:
        df_model_training['4G'] = filtered_df['4G'].fillna(0)
        df_model_training['4G'] = df_model_training['4G'].replace('Da', 1).replace('Nu', 0)
    
    # Process resolution
    if 'Rezolutie maxima (px)' in filtered_df.columns:
        try:
            df_model_training[['resolution width', 'resolution height']] = filtered_df['Rezolutie maxima (px)'].str.split(' x ', expand=True)
            df_model_training['resolution width'] = pd.to_numeric(df_model_training['resolution width'], errors='coerce').fillna(0)
            df_model_training['resolution height'] = pd.to_numeric(df_model_training['resolution height'], errors='coerce').fillna(0)
        except:
            print("Could not parse resolution data properly. Using default values.")
            df_model_training['resolution width'] = 0
            df_model_training['resolution height'] = 0
    
    # Process screen size
    if 'Diagonala (inch)' in filtered_df.columns:
        df_model_training['Diagonala'] = pd.to_numeric(filtered_df['Diagonala (inch)'], errors='coerce').fillna(0)
    
    # Process CPU cores
    if 'Numar nuclee' in filtered_df.columns:
        df_model_training['Numar nuclee'] = filtered_df['Numar nuclee'].astype(str).str.split('(').str[0]
        df_model_training['Numar nuclee'] = pd.to_numeric(df_model_training['Numar nuclee'], errors='coerce').fillna(0)
    
    # Process RAM and Flash memory
    if 'Memorie RAM' in filtered_df.columns:
        df_model_training['Memorie RAM'] = filtered_df['Memorie RAM'].astype(str).str.split(' ').str[0]
        df_model_training['Memorie RAM'] = pd.to_numeric(df_model_training['Memorie RAM'], errors='coerce').fillna(0)
    
    if 'Memorie Flash' in filtered_df.columns:
        df_model_training['Memorie Flash'] = filtered_df['Memorie Flash'].astype(str).str.split(' ').str[0]
        df_model_training['Memorie Flash'] = pd.to_numeric(df_model_training['Memorie Flash'], errors='coerce').fillna(0)
    
    # Process wireless charging
    if 'Incarcare Wireless' in filtered_df.columns:
        df_model_training['Incarcare Wireless'] = filtered_df['Incarcare Wireless'].fillna(0)
        df_model_training['Incarcare Wireless'] = df_model_training['Incarcare Wireless'].replace('Da', 1).replace('Nu', 0)
    
    # Process battery capacity
    if 'Capacitate' in filtered_df.columns:
        df_model_training['Capacitate Baterie'] = filtered_df['Capacitate'].astype(str).str.split(' ').str[0]
        df_model_training['Capacitate Baterie'] = pd.to_numeric(df_model_training['Capacitate Baterie'], errors='coerce').fillna(0)
    
    # Process Dual SIM
    if 'Dual SIM' in filtered_df.columns:
        df_model_training['Dual SIM'] = filtered_df['Dual SIM'].fillna(0)
        df_model_training['Dual SIM'] = df_model_training['Dual SIM'].replace('Da', 1).replace('Nu', 0)
    
    # Add price column if it exists
    if 'price' in filtered_df.columns:
        df_model_training['price'] = filtered_df['price']
    
    return df_model_training