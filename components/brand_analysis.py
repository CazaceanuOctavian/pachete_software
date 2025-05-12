import streamlit as st
import pandas as pd
import plotly.express as px

def show_brand_analysis(filtered_df):
    """Display brand analysis visualizations"""
    st.header("Brand Analysis")
    
    if 'manufacturer' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Brand Distribution")
            
            # Get top N brands
            top_n = st.slider("Number of top brands to display", 5, 20, 10, key="brand_dist_slider")
            brand_counts = filtered_df['manufacturer'].value_counts().nlargest(top_n)
            
            fig = px.bar(
                x=brand_counts.index,
                y=brand_counts.values,
                title=f"Top {top_n} Smartphone Brands by Count",
                labels={'x': 'Brand', 'y': 'Count'},
                color=brand_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Brand Percentage")
            
            fig = px.pie(
                values=brand_counts.values,
                names=brand_counts.index,
                title=f"Market Share of Top {top_n} Brands",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature distribution across brands
        st.subheader("Feature Distribution Across Brands")
        
        # Get categorical columns that might be interesting
        cat_cols = [col for col in filtered_df.columns if filtered_df[col].dtype == 'object' 
                   and col != 'manufacturer' and filtered_df[col].nunique() < 20]
        
        if cat_cols:
            selected_feature = st.selectbox("Select feature", cat_cols)
            
            # Get top brands
            top_brands = filtered_df['manufacturer'].value_counts().nlargest(8).index.tolist()
            brand_filtered_df = filtered_df[filtered_df['manufacturer'].isin(top_brands)]
            
            # Group by brand and selected feature
            feature_dist = pd.crosstab(brand_filtered_df['manufacturer'], brand_filtered_df[selected_feature])
            
            # Convert to percentage
            feature_dist_pct = feature_dist.div(feature_dist.sum(axis=1), axis=0) * 100
            
            # Plot
            fig = px.imshow(
                feature_dist_pct,
                labels=dict(x=selected_feature, y="Brand", color="Percentage (%)"),
                color_continuous_scale="Viridis",
                title=f"{selected_feature} Distribution Across Top Brands (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show actual counts
            st.subheader("Feature Count by Brand")
            st.dataframe(feature_dist, use_container_width=True)
        else:
            st.info("No suitable categorical features found for analysis")
    else:
        st.info("Brand column not found in the dataset")