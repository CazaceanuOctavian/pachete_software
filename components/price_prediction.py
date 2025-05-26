import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from utils.data_processing import prepare_model_data


def _initialize_price_pred_session_state():
    """Initializes session state variables for price prediction if they don't exist."""
    if 'xgb_model' not in st.session_state:
        st.session_state.xgb_model = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    if 'metrics_df' not in st.session_state:
        st.session_state.metrics_df = None
    if 'sig_df' not in st.session_state:
        st.session_state.sig_df = None
    if 'res_df' not in st.session_state:
        st.session_state.res_df = None
    if 'x_interactive' not in st.session_state:
        st.session_state.x_interactive = None


def _train_price_model(X_train, y_train, X_test, y_test, model_features, xgb_params):
    try:
        model = xgb.XGBRegressor(
            n_estimators=xgb_params['n_estimators'],
            max_depth=xgb_params['max_depth'],
            learning_rate=xgb_params['learning_rate'],
            random_state=xgb_params['random_state_model'],
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        metrics = pd.DataFrame({
            'Metric': ['R² Score', 'MAE (RON)', 'RMSE (RON)'],
            'Training Set': [f"{train_r2:.3f}", f"{train_mae:.2f}", f"{train_rmse:.2f}"],
            'Testing Set': [f"{test_r2:.3f}", f"{test_mae:.2f}", f"{test_rmse:.2f}"]
        })

        importance = model.feature_importances_
        significance = pd.DataFrame({
            'Feature': model_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        results = pd.DataFrame({'Actual': y_test, 'Predicted': test_preds})

        st.success("Model trained successfully!")
        return model, metrics, significance, results
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None, None, None, None


def _display_model_performance(metrics, significance):
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        if metrics is not None:
            st.markdown("##### Metrics Table")
            st.dataframe(metrics, use_container_width=True)
    with col2:
        if significance is not None:
            st.markdown("##### Feature Importance")
            fig = px.bar(
                significance, x='Importance', y='Feature',
                orientation='h', title="Feature Importance", color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)


def _display_predictions_vs_actual(results):
    if results is not None:
        st.subheader("Predicted vs Actual Prices (Test Set)")
        fig = px.scatter(
            results, x='Actual', y='Predicted',
            title="Predicted vs Actual Prices",
            labels={'Actual': 'Actual Price (RON)', 'Predicted': 'Predicted Price (RON)'},
            opacity=0.7, trendline="ols", trendline_color_override="red"
        )
        min_val = min(results['Actual'].min(), results['Predicted'].min())
        max_val = max(results['Actual'].max(), results['Predicted'].max())
        fig.add_shape(
            type="line", line=dict(dash="dash", color="gray"),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        st.plotly_chart(fig, use_container_width=True)


def _display_interactive_predictor(model, interactive_X_data, model_features):
    if model is None or interactive_X_data is None or not model_features:
        st.info("Train a model first to enable interactive predictions with the correct features.")
        return

    st.subheader("Try the Model: Predict Price for Custom Input")
    st.write("Adjust the feature values below to see the predicted price.")

    input_values = {}
    with st.form(key="interactive_prediction_form"):
        for feature in model_features:
            min_val_feat = float(interactive_X_data[feature].min())
            max_val_feat = float(interactive_X_data[feature].max())
            mean_val_feat = float(interactive_X_data[feature].mean())

            step_feat = 1.0 if interactive_X_data[feature].dtype in ['int64', 'int32'] and (
                        max_val_feat - min_val_feat) < 100 else max((max_val_feat - min_val_feat) / 100, 0.01)

            input_values[feature] = st.slider(
                f"Set {feature}", min_val_feat, max_val_feat,
                value=mean_val_feat, step=step_feat,
                key=f"price_pred_input_slider_{feature}"
            )
        predict_button_interactive = st.form_submit_button("Predict Price with Custom Inputs")

    if predict_button_interactive:
        input_df = pd.DataFrame([input_values])
        input_df = input_df[model_features]
        custom_prediction = model.predict(input_df)[0]
        st.markdown(f"### Predicted Price: **{custom_prediction:.2f} RON**")


def show_price_prediction(filtered_df):
    st.header("Price Prediction Model")
    _initialize_price_pred_session_state()

    if 'price' not in filtered_df.columns:
        st.info("Price column not found in the dataset")
        st.session_state.xgb_model = None
        st.session_state.selected_features = []
        st.session_state.metrics_df = None
        st.session_state.sig_df = None
        st.session_state.res_df = None
        st.session_state.x_interactive = None
        return

    st.subheader("Train an XGBoost Regressor to predict smartphone prices")
    df_model_training = prepare_model_data(filtered_df.copy())
    available_features = [col for col in df_model_training.columns if col != 'price']

    if not available_features:
        st.info("No features available for prediction after initial processing.")
        st.session_state.xgb_model = None
        st.session_state.selected_features = []
        st.session_state.metrics_df = None
        st.session_state.sig_df = None
        st.session_state.res_df = None
        st.session_state.x_interactive = None
        return

    with st.form(key="price_prediction_form"):
        st.markdown("##### Feature Selection & Model Parameters")
        current_selected_features = st.multiselect(
            "Select features for price prediction",
            available_features,
            default=st.session_state.selected_features or available_features[:min(5, len(available_features))],
            key="price_pred_current_features_multiselect"
        )

        test_size_percentage = st.slider("Test set size (%)", 10, 50, 20, key="price_pred_test_size_form")
        random_state_split = st.slider("Random seed (for train-test split)", 0, 100, 42,
                                       key="price_pred_seed_split_form")

        st.markdown("###### XGBoost Hyperparameters")
        col1_xgb, col2_xgb, col3_xgb = st.columns(3)
        with col1_xgb:
            n_estimators = st.slider("Number of trees (n_estimators)", 10, 300, 100, key="price_pred_n_estimators_form")
        with col2_xgb:
            max_depth = st.slider("Maximum tree depth (max_depth)", 2, 20, 6, key="price_pred_max_depth_form")
        with col3_xgb:
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, step=0.01, key="price_pred_learning_rate_form")
        random_state_model_form = st.slider("Random seed (for XGBoost model)", 0, 100, 42,
                                            key="price_pred_seed_model_form")

        train_button_form = st.form_submit_button(label="Train Price Prediction Model")

    if train_button_form:
        if not current_selected_features:
            st.warning("Please select at least one feature for prediction.")
        else:
            with st.spinner("Training XGBoost model..."):
                X = df_model_training[current_selected_features].copy()
                y = df_model_training['price']
                X = X.fillna(X.median())

                st.session_state.x_interactive = X.copy()
                st.session_state.selected_features = current_selected_features

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=(test_size_percentage / 100.0), random_state=random_state_split
                )

                xgb_params = {
                    'n_estimators': n_estimators, 'max_depth': max_depth,
                    'learning_rate': learning_rate, 'random_state_model': random_state_model_form
                }

                model, metrics, significance, results = _train_price_model(
                    X_train, y_train, X_test, y_test, current_selected_features, xgb_params
                )

                st.session_state.xgb_model = model
                st.session_state.metrics_df = metrics
                st.session_state.sig_df = significance
                st.session_state.res_df = results

    # Display results if a model is trained and stored in session state
    if st.session_state.xgb_model is not None:
        _display_model_performance(st.session_state.metrics_df, st.session_state.sig_df)
        _display_predictions_vs_actual(st.session_state.res_df)
        _display_interactive_predictor(
            st.session_state.xgb_model,
            st.session_state.x_interactive,
            st.session_state.selected_features
        )
    elif not train_button_form:
        st.info("Configure parameters and train a model to see predictions and results.")

