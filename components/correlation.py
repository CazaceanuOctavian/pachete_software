import streamlit as st
import plotly.express as px

def show_correlation_analysis(filtered_df):
    """Display feature correlation analysis"""
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