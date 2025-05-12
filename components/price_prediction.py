import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from utils.data_processing import prepare_model_data

def show_price_prediction(filtered_df):
    """Display price prediction model functionality"""
    st.header("Price Prediction Model")
    
    if 'price' in filtered_df.columns:
        st.subheader("Train an XGBoost Regressor to predict smartphone prices")
        
        # Preprocessing steps for model training
        df_model_training = prepare_model_data(filtered_df)
        
        # Display available features
        available_features = list(df_model_training.columns)
        if 'price' in available_features:
            available_features.remove('price')
        
        if len(available_features) > 0:
            # Let user select features
            selected_features = st.multiselect(
                "Select features for price prediction", 
                available_features,
                default=available_features[:min(5, len(available_features))],
                key="price_prediction_features"
            )
            
            if selected_features:
                # Split data
                X = df_model_training[selected_features].copy()
                y = df_model_training['price']
                
                # Handle missing values
                X = X.fillna(X.median())
                
                # Train-test split
                test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
                random_state = st.slider("Random seed", 0, 100, 42, key="price_prediction_seed")
                
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