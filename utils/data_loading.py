import streamlit as st
import pandas as pd
from pandas import json_normalize

@st.cache_data
def extract_smartphones(df):
    """
    Extract all products that have "Smartphone": "Da" in their specifications column.
    """
    smartphone_mask = df['specifications'].apply(
        lambda specs: isinstance(specs, dict) and specs.get('Smartphone') == 'Da'
    )
    smartphones_df = df[smartphone_mask].copy()
    return smartphones_df

@st.cache_data
def flatten_json_column(df, json_column):
    """
    Flatten a JSON column in a DataFrame so that the fields become separate columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Check if the JSON column exists in the DataFrame
    if json_column not in result_df.columns:
        raise ValueError(f"Column '{json_column}' not found in DataFrame")
    
    # Normalize the JSON column
    try:
        # Handle cases where some rows might have None/NaN values in the JSON column
        mask = result_df[json_column].notna()
        
        if mask.any():
            # Apply json_normalize only to rows that have valid JSON
            normalized_df = json_normalize(result_df.loc[mask, json_column])
            
            # Drop the original JSON column from the result
            result_subset = result_df.loc[mask].drop(json_column, axis=1)
            
            # Combine the original DataFrame (minus the JSON column) with the normalized data
            flattened_subset = pd.concat([result_subset.reset_index(drop=True), 
                                          normalized_df.reset_index(drop=True)], 
                                         axis=1)
            
            # Merge back with rows that had None/NaN values
            if (~mask).any():
                result_df = pd.concat([flattened_subset, 
                                       result_df.loc[~mask]]).sort_index()
            else:
                result_df = flattened_subset
        
        return result_df
        
    except Exception as e:
        raise ValueError(f"Error flattening JSON column: {str(e)}")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_json('datasets/evomag_2024_11_13.json')
        df_smartphone = extract_smartphones(df)
        df_smartphone_normalised = flatten_json_column(df_smartphone, 'specifications')
        return df, df_smartphone, df_smartphone_normalised
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None