import streamlit as st
import plotly.express as px

def show_price_analysis(filtered_df):
    """Display price analysis visualizations"""
    st.header("Price Analysis")
    
    if 'price' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution")
            fig = px.histogram(
                filtered_df, 
                x='price', 
                nbins=50, 
                title="Price Distribution",
                labels={'price': 'Price (RON)'},
                color_discrete_sequence=['#3366CC']
            )
            fig.update_layout(bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Price Box Plot")
            fig = px.box(
                filtered_df,
                y='price',
                title="Price Range and Outliers",
                labels={'price': 'Price (RON)'},
                color_discrete_sequence=['#3366CC']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Price by brand if brand column exists
        if 'manufacturer' in filtered_df.columns:
            st.subheader("Price by Brand")
            
            # Get top N brands by count
            top_n = st.slider("Number of brands to display", 5, 20, 10)
            top_brands = filtered_df['manufacturer'].value_counts().nlargest(top_n).index.tolist()
            
            brand_df = filtered_df[filtered_df['manufacturer'].isin(top_brands)]
            
            fig = px.box(
                brand_df,
                x='manufacturer',
                y='price',
                title=f"Price Distribution by Top {top_n} Brands",
                labels={'price': 'Price (RON)', 'manufacturer': 'Brand'},
                color='manufacturer',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price statistics by brand
            price_by_brand = filtered_df.groupby('manufacturer')['price'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index()
            price_by_brand = price_by_brand.sort_values('count', ascending=False).head(top_n)
            price_by_brand = price_by_brand.rename(columns={
                'manufacturer': 'Brand',
                'count': 'Count', 
                'mean': 'Average Price', 
                'median': 'Median Price',
                'min': 'Min Price',
                'max': 'Max Price'
            })
            
            # Format price columns
            for col in ['Average Price', 'Median Price', 'Min Price', 'Max Price']:
                price_by_brand[col] = price_by_brand[col].round(2)
            
            st.dataframe(price_by_brand, use_container_width=True)
    else:
        st.info("Price column not found in the dataset")