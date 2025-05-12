import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from utils.data_processing import prepare_model_data

def show_clustering(filtered_df):
    """Display clustering analysis functionality"""
    st.header("Smartphone Clustering Analysis")
    
    # Preprocessing steps for model training
    df_model_training = prepare_model_data(filtered_df)
    
    # Process manufacturer
    if 'manufacturer' in filtered_df.columns:
        # Convert to categorical encoding
        df_model_training['Manufacturer'] = pd.Categorical(filtered_df['manufacturer']).codes
    
    # Display available features
    available_features = list(df_model_training.columns)
    
    if len(available_features) > 0:
        # Let user select features for clustering
        st.subheader("Select Features for Clustering")
        
        selected_features = st.multiselect(
            "Select features for clustering", 
            available_features,
            default=available_features[:min(5, len(available_features))],
            key="clustering_features"
        )
        
        if selected_features:
            # Additional parameters
            col1, col2 = st.columns(2)
            
            with col1:
                max_clusters = st.slider("Maximum number of clusters to evaluate", 2, 15, 10)
            
            with col2:
                random_state = st.slider("Random seed", 0, 100, 42, key="clustering_seed")
            
            # Perform clustering button
            if st.button("Run Clustering Analysis"):
                with st.spinner("Performing clustering analysis..."):
                    try:
                        # Extract features for clustering
                        X = df_model_training[selected_features].copy()
                        
                        # Handle missing values
                        X = X.fillna(X.median())
                        
                        # Standardize the data
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Find optimal number of clusters
                        st.subheader("Determining Optimal Number of Clusters")
                        
                        # Create placeholders for plots
                        elbow_plot = st.empty()
                        silhouette_plot = st.empty()
                        
                        # Calculate inertia and silhouette scores
                        inertia_values = []
                        silhouette_scores = []
                        k_range = range(2, max_clusters+1)
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        
                        for i, k in enumerate(k_range):
                            # K-means clustering
                            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                            cluster_labels = kmeans.fit_predict(X_scaled)
                            
                            # Inertia (for Elbow method)
                            inertia_values.append(kmeans.inertia_)
                            
                            # Silhouette score
                            sil_score = silhouette_score(X_scaled, cluster_labels)
                            silhouette_scores.append(sil_score)
                            
                            progress_bar.progress((i + 1) / len(k_range))
                        
                        # Reset progress
                        progress_bar.empty()
                        
                        # Plot the Elbow method
                        fig_elbow = px.line(
                            x=list(k_range), 
                            y=inertia_values,
                            markers=True,
                            title="Elbow Method for Optimal k",
                            labels={"x": "Number of clusters (k)", "y": "Inertia"},
                            width=600, height=400
                        )
                        elbow_plot.plotly_chart(fig_elbow, use_container_width=True)
                        
                        # Plot the Silhouette scores
                        fig_silhouette = px.line(
                            x=list(k_range), 
                            y=silhouette_scores,
                            markers=True,
                            title="Silhouette Scores for Different k",
                            labels={"x": "Number of clusters (k)", "y": "Silhouette Score"},
                            width=600, height=400
                        )
                        silhouette_plot.plotly_chart(fig_silhouette, use_container_width=True)
                        
                        # Best k based on silhouette score
                        best_k_silhouette = k_range[np.argmax(silhouette_scores)]
                        
                        # Allow user to select the number of clusters
                        selected_k = st.slider(
                            "Select number of clusters (k)", 
                            min_value=2, 
                            max_value=max_clusters, 
                            value=best_k_silhouette,
                            help="The optimal k based on silhouette score is " + str(best_k_silhouette)
                        )
                        
                        # Apply K-means with the selected number of clusters
                        st.subheader(f"K-means Clustering with {selected_k} Clusters")
                        
                        # Apply K-means with the selected number of clusters
                        kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
                        cluster_labels = kmeans.fit_predict(X_scaled)
                        
                        # Add cluster labels to the dataframe
                        df_model_training['Cluster'] = cluster_labels
                        
                        # Visualize clusters using PCA for dimensionality reduction
                        st.write("### Cluster Visualization using PCA")
                        
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # Create a DataFrame for visualization
                        pca_df = pd.DataFrame({
                            'PCA Component 1': X_pca[:, 0],
                            'PCA Component 2': X_pca[:, 1],
                            'Cluster': cluster_labels
                        })
                        
                        # If manufacturer is available, add it to the PCA dataframe
                        if 'manufacturer' in filtered_df.columns:
                            pca_df['Manufacturer'] = filtered_df['manufacturer'].values
                        
                        # Create scatter plot
                        if 'Manufacturer' in pca_df.columns:
                            fig = px.scatter(
                                pca_df,
                                x='PCA Component 1',
                                y='PCA Component 2',
                                color='Cluster',
                                hover_data=['Manufacturer'],
                                title="Cluster Visualization using PCA",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                        else:
                            fig = px.scatter(
                                pca_df,
                                x='PCA Component 1',
                                y='PCA Component 2',
                                color='Cluster',
                                title="Cluster Visualization using PCA",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Analyze cluster characteristics
                        st.write("### Cluster Characteristics")
                        
                        # Add original features for analysis if they exist in filtered_df
                        analysis_df = df_model_training.copy()
                        if 'manufacturer' in filtered_df.columns:
                            analysis_df['Manufacturer Name'] = filtered_df['manufacturer'].values
                        if 'price' in filtered_df.columns:
                            analysis_df['Price'] = filtered_df['price'].values
                        
                        # Display the number of smartphones in each cluster
                        cluster_counts = analysis_df['Cluster'].value_counts().sort_index()
                        
                        fig = px.bar(
                            x=cluster_counts.index,
                            y=cluster_counts.values,
                            title="Number of Smartphones in Each Cluster",
                            labels={'x': 'Cluster', 'y': 'Count'},
                            color=cluster_counts.index,
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display cluster statistics for selected features
                        cluster_stats = analysis_df.groupby('Cluster')[selected_features].mean()
                        
                        # Normalize values for radar chart
                        normalized_stats = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())
                        
                        # Select cluster to analyze
                        selected_cluster = st.selectbox(
                            "Select a cluster to analyze in detail",
                            range(selected_k)
                        )
                        
                        # Create radar chart for the selected cluster
                        fig = px.line_polar(
                            r=normalized_stats.loc[selected_cluster].values,
                            theta=normalized_stats.columns,
                            line_close=True,
                            title=f"Characteristics of Cluster {selected_cluster}",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display the feature means for each cluster
                        st.write("### Mean Feature Values by Cluster")
                        
                        # Format the cluster_stats
                        formatted_stats = cluster_stats.round(2)
                        st.dataframe(formatted_stats, use_container_width=True)
                        
                        # Show most common manufacturers in each cluster if available
                        if 'Manufacturer Name' in analysis_df.columns:
                            st.write("### Top Manufacturers in Each Cluster")
                            
                            # Create a table for top manufacturers
                            top_manufacturers = pd.DataFrame()
                            
                            for cluster in range(selected_k):
                                cluster_manufacturers = analysis_df[analysis_df['Cluster'] == cluster]['Manufacturer Name'].value_counts().nlargest(3)
                                
                                # Create a string of top manufacturers
                                manufacturer_list = [f"{mfr} ({count})" for mfr, count in cluster_manufacturers.items()]
                                manufacturer_str = ", ".join(manufacturer_list)
                                
                                # Add to the DataFrame
                                top_manufacturers.loc[cluster, 'Top Manufacturers'] = manufacturer_str
                            
                            st.dataframe(top_manufacturers, use_container_width=True)
                        
                        # Show price statistics if available
                        if 'Price' in analysis_df.columns:
                            st.write("### Price Distribution Across Clusters")
                            
                            price_stats = analysis_df.groupby('Cluster')['Price'].agg(['count', 'mean', 'median', 'min', 'max']).round(2)
                            st.dataframe(price_stats, use_container_width=True)
                            
                            # Box plot of prices by cluster
                            fig = px.box(
                                analysis_df,
                                x='Cluster',
                                y='Price',
                                color='Cluster',
                                title="Price Distribution by Cluster",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show sample smartphones from each cluster
                        st.write("### Sample Smartphones from Each Cluster")
                        
                        # Create a selector for the cluster
                        sample_cluster = st.selectbox(
                            "Select a cluster to see sample smartphones",
                            range(selected_k),
                            key="sample_cluster"
                        )
                        
                        # Display samples
                        samples = analysis_df[analysis_df['Cluster'] == sample_cluster]
                        
                        # If manufacturer is available, sort by it
                        if 'Manufacturer Name' in samples.columns:
                            samples = samples.sort_values('Manufacturer Name')
                        
                        # Select columns to display
                        display_cols = ['Cluster']
                        if 'Manufacturer Name' in samples.columns:
                            display_cols.append('Manufacturer Name')
                        if 'Price' in samples.columns:
                            display_cols.append('Price')
                        
                        # Add selected features
                        display_cols.extend([f for f in selected_features if f not in display_cols])
                        
                        # Display samples
                        st.dataframe(samples[display_cols].head(10), use_container_width=True)
                        
                        # Feature importance for clusters
                        st.write("### Feature Importance for Cluster Separation")
                        
                        # Calculate feature importance based on cluster centroids
                        centers = kmeans.cluster_centers_
                        overall_mean = np.mean(X_scaled, axis=0)
                        feature_importance = {}
                        
                        for feature_idx, feature_name in enumerate(selected_features):
                            # Calculate maximum deviation from the overall mean across all clusters
                            max_deviation = max([abs(centers[i, feature_idx] - overall_mean[feature_idx]) for i in range(selected_k)])
                            feature_importance[feature_name] = max_deviation
                        
                        # Create DataFrame and sort
                        importance_df = pd.DataFrame({
                            'Feature': list(feature_importance.keys()),
                            'Importance': list(feature_importance.values())
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Feature Importance for Cluster Separation",
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save clustering model
                        if st.button("Save Clustering Model"):
                            import joblib
                            try:
                                joblib.dump(kmeans, 'smartphone_clusters_model.pkl')
                                joblib.dump(scaler, 'smartphone_clusters_scaler.pkl')
                                st.success("Clustering model and scaler saved successfully!")
                            except Exception as e:
                                st.error(f"Error saving model: {str(e)}")
                        
                    except Exception as e:
                        st.error(f"Error performing clustering: {str(e)}")
        else:
            st.info("Please select at least one feature for clustering")
    else:
        st.info("No features available for clustering")