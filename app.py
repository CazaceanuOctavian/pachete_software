import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas import json_normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import XGBoost for regression
import xgboost as xgb

# Set page configuration
st.set_page_config(
    page_title="Smartphone Data Visualization",
    page_icon="📱",
    layout="wide"
)

# Add title and description
st.title("📱 Smartphone Data Analysis Dashboard")
st.markdown("An interactive dashboard for exploring and visualizing smartphone data from the evomag dataset.")

# Define helper functions
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

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💰 Price Analysis", "📈 Feature Correlation", "🔍 Brand Analysis", "🤖 Price Prediction"])

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

    # Tab 1: Overview
    with tab1:
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

    # Tab 2: Price Analysis
    with tab2:
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

    # Tab 3: Feature Correlation
    with tab3:
        st.header("Feature Correlation Analysis")
        
        # Get numeric columns for correlation
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Remove columns with all NaN or single unique value
            valid_cols = []
            for col in numeric_cols:
                if filtered_df[col].nunique() > 1 and not filtered_df[col].isna().all():
                    valid_cols.append(col)
            
            if len(valid_cols) > 1:
                # Allow user to select columns for correlation
                selected_cols = st.multiselect(
                    "Select columns for correlation analysis", 
                    valid_cols,
                    default=valid_cols[:5] if len(valid_cols) > 5 else valid_cols
                )
                
                if len(selected_cols) > 1:
                    # Calculate correlation
                    correlation = filtered_df[selected_cols].corr()
                    
                    st.subheader("Correlation Matrix")
                    fig = px.imshow(
                        correlation,
                        text_auto=True,
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Allow user to select features for scatter plot
                    st.subheader("Feature Relationship")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_feature = st.selectbox("X-axis feature", selected_cols, index=0)
                    with col2:
                        y_feature = st.selectbox("Y-axis feature", selected_cols, index=min(1, len(selected_cols)-1))
                    with col3:
                        if 'manufacturer' in filtered_df.columns:
                            color_by = st.selectbox("Color by", ["None", "manufacturer"], index=1)
                        else:
                            color_by = "None"
                    
                    # Create scatter plot
                    if color_by == "None":
                        fig = px.scatter(
                            filtered_df,
                            x=x_feature,
                            y=y_feature,
                            title=f"{y_feature} vs {x_feature}",
                            opacity=0.7
                        )
                    else:
                        fig = px.scatter(
                            filtered_df,
                            x=x_feature,
                            y=y_feature,
                            color=color_by,
                            title=f"{y_feature} vs {x_feature} by {color_by}",
                            opacity=0.7
                        )
                        
                    fig.update_layout(xaxis_title=x_feature, yaxis_title=y_feature)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 columns for correlation analysis")
            else:
                st.info("Not enough valid numeric columns for correlation analysis")
        else:
            st.info("Not enough numeric columns for correlation analysis")

    # Tab 4: Brand Analysis
    with tab4:
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

    # Tab 5: Price Prediction
    with tab5:
        st.header("Price Prediction Model")
        
        if 'price' in filtered_df.columns:
            st.subheader("Train an XGBoost Regressor to predict smartphone prices")
            
            # Preprocessing steps for model training
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
                    st.warning("Could not parse resolution data properly. Using default values.")
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
            
            # Add price column
            df_model_training['price'] = filtered_df['price']
            
            # Display available features
            available_features = list(df_model_training.columns)
            if 'price' in available_features:
                available_features.remove('price')
            
            if len(available_features) > 0:
                # Let user select features
                selected_features = st.multiselect(
                    "Select features for price prediction", 
                    available_features,
                    default=available_features[:min(5, len(available_features))]
                )
                
                if selected_features:
                    # Split data
                    X = df_model_training[selected_features].copy()
                    y = df_model_training['price']
                    
                    # Handle missing values
                    X = X.fillna(X.median())
                    
                    # Train-test split
                    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
                    random_state = st.slider("Random seed", 0, 100, 42)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # XGBoost parameters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.slider("Number of trees", 10, 200, 100)
                    with col2:
                        max_depth = st.slider("Maximum tree depth", 2, 20, 6)
                    with col3:
                        learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, step=0.01)
                    
                    # Train button
                    if st.button("Train Model"):
                        with st.spinner("Training XGBoost model..."):
                            try:
                                # Train model
                                xgb_model = xgb.XGBRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    random_state=random_state
                                )
                                xgb_model.fit(X_train, y_train)
                                
                                # Make predictions
                                train_preds = xgb_model.predict(X_train)
                                test_preds = xgb_model.predict(X_test)
                                
                                # Calculate metrics
                                train_r2 = r2_score(y_train, train_preds)
                                test_r2 = r2_score(y_test, test_preds)
                                
                                train_mae = mean_absolute_error(y_train, train_preds)
                                test_mae = mean_absolute_error(y_test, test_preds)
                                
                                train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
                                test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
                                
                                # Display metrics
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("Model Performance")
                                    metrics_df = pd.DataFrame({
                                        'Metric': ['R² Score', 'MAE', 'RMSE'],
                                        'Training Set': [train_r2, train_mae, train_rmse],
                                        'Testing Set': [test_r2, test_mae, test_rmse]
                                    })
                                    
                                    # Format metrics
                                    metrics_df['Training Set'] = metrics_df['Training Set'].round(3)
                                    metrics_df['Testing Set'] = metrics_df['Testing Set'].round(3)
                                    
                                    st.dataframe(metrics_df, use_container_width=True)
                                
                                with col2:
                                    st.subheader("Feature Importance")
                                    
                                    # Get feature importance
                                    importance = xgb_model.feature_importances_
                                    
                                    # Create DataFrame for feature importance
                                    importance_df = pd.DataFrame({
                                        'Feature': selected_features,
                                        'Importance': importance
                                    })
                                    importance_df = importance_df.sort_values('Importance', ascending=False)
                                    
                                    # Plot feature importance
                                    fig = px.bar(
                                        importance_df,
                                        x='Importance',
                                        y='Feature',
                                        orientation='h',
                                        title="Feature Importance",
                                        color='Importance',
                                        color_continuous_scale='Viridis'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Prediction vs Actual plot
                                st.subheader("Predicted vs Actual Prices")
                                
                                # Create DataFrame for prediction results
                                results_df = pd.DataFrame({
                                    'Actual': y_test,
                                    'Predicted': test_preds
                                })
                                
                                # Plot
                                fig = px.scatter(
                                    results_df,
                                    x='Actual',
                                    y='Predicted',
                                    title="Predicted vs Actual Prices",
                                    labels={'Actual': 'Actual Price (RON)', 'Predicted': 'Predicted Price (RON)'},
                                    opacity=0.7
                                )
                                
                                # Add perfect prediction line
                                min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
                                max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
                                fig.add_shape(
                                    type="line", line=dict(dash="dash", color="gray"),
                                    x0=min_val, y0=min_val, x1=max_val, y1=max_val
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interactive price prediction
                                st.subheader("Try the Model")
                                st.write("Adjust the feature values to see the predicted price")
                                
                                # Create input sliders for each feature
                                input_values = {}
                                for feature in selected_features:
                                    min_val = float(X[feature].min())
                                    max_val = float(X[feature].max())
                                    mean_val = float(X[feature].mean())
                                    
                                    step = (max_val - min_val) / 100
                                    step = max(step, 0.01)  # Minimum step of 0.01
                                    
                                    input_values[feature] = st.slider(
                                        f"{feature}",
                                        min_val,
                                        max_val,
                                        mean_val,
                                        step=step
                                    )
                                
                                # Make prediction
                                input_df = pd.DataFrame([input_values])
                                prediction = xgb_model.predict(input_df)[0]
                                
                                # Display prediction
                                st.markdown(f"### Predicted Price: **{prediction:.2f} RON**")
                                
                            except Exception as e:
                                st.error(f"Error training the model: {str(e)}")
                else:
                    st.info("Please select at least one feature for prediction")
            else:
                st.info("No features available for prediction")
        else:
            st.info("Price column not found in the dataset")
else:
    st.error("Failed to load the smartphone data. Please check if the dataset file exists.")

# Add footer
st.markdown("---")
st.markdown("Smartphone Data Visualization Dashboard | Created with Streamlit")