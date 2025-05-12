import streamlit as st
from utils.data_loading import load_data
from components import overview, price_analysis, correlation, brand_analysis, price_prediction, clustering

# Set page configuration
st.set_page_config(
    page_title="Smartphone Data Visualization",
    page_icon="📱",
    layout="wide"
)

# Add title and description
st.title("📱 Smartphone Data Analysis Dashboard")
st.markdown("An interactive dashboard for exploring and visualizing smartphone data from the evomag dataset.")

# Load the data
with st.spinner("Loading data..."):
    df, df_smartphone, df_smartphone_normalised = load_data()

if df_smartphone_normalised is not None:
    # Display dataset info in sidebar
    with st.sidebar:
        st.header("Dataset Information")
        st.write(f"Total products: {len(df):,}")
        st.write(f"Smartphones: {len(df_smartphone):,}")
        
        # Add filters
        st.header("Filters")
        
        # Filter by price range
        if 'price' in df_smartphone_normalised.columns:
            price_min = float(df_smartphone_normalised['price'].min())
            price_max = float(df_smartphone_normalised['price'].max())
            price_range = st.slider("Price Range (RON)", price_min, price_max, (price_min, price_max))
        else:
            price_range = (0, 10000)  # Default values if price column doesn't exist
        
        # Filter by brand if available
        if 'manufacturer' in df_smartphone_normalised.columns:
            all_brands = df_smartphone_normalised['manufacturer'].dropna().unique().tolist()
            selected_brands = st.multiselect("Brands", all_brands, default=all_brands[:5] if len(all_brands) > 5 else all_brands)
        else:
            selected_brands = []
            
        # Apply filters
        filtered_df = df_smartphone_normalised.copy()
        
        if 'price' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                                       (filtered_df['price'] <= price_range[1])]
        
        if selected_brands and 'manufacturer' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['manufacturer'].isin(selected_brands)]
            
        st.write(f"Filtered smartphones: {len(filtered_df):,}")
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_smartphones.csv",
            mime="text/csv",
        )

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "💰 Price Analysis", "📈 Feature Correlation", "🔍 Brand Analysis", "🤖 Price Prediction", "🧩 Clustering"])

    # Tab 1: Overview
    with tab1:
        overview.show_overview(filtered_df)

    # Tab 2: Price Analysis
    with tab2:
        price_analysis.show_price_analysis(filtered_df)

    # Tab 3: Feature Correlation
    with tab3:
        correlation.show_correlation_analysis(filtered_df)

    # Tab 4: Brand Analysis
    with tab4:
        brand_analysis.show_brand_analysis(filtered_df)

    # Tab 5: Price Prediction
    with tab5:
        price_prediction.show_price_prediction(filtered_df)

    # Tab 6: Clustering
    with tab6:
        clustering.show_clustering(filtered_df)
else:
    st.error("Failed to load the smartphone data. Please check if the dataset file exists.")

# Add footer
st.markdown("---")
st.markdown("Smartphone Data Visualization Dashboard | Created with Streamlit")