import streamlit as st
import pandas as pd

def show_overview(filtered_df):
    """Display overview information about the dataset"""
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Smartphone Data Sample")
        st.dataframe(filtered_df.head(10), use_container_width=True)
        
    with col2:
        st.subheader("Data Statistics")
        
        # Get numeric columns
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for statistics")
    
    # Show columns and missing values
    st.subheader("Column Information")
    
    # Calculate missing values for each column
    missing_data = pd.DataFrame({
        'Column': filtered_df.columns,
        'Data Type': filtered_df.dtypes,
        'Missing Values': filtered_df.isna().sum(),
        'Missing (%)': (filtered_df.isna().sum() / len(filtered_df) * 100).round(2)
    })
    
    st.dataframe(missing_data.sort_values('Missing (%)', ascending=False), use_container_width=True)