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
    st.header("Smartphone Clustering Analysis")

    if 'cluster_kmeans_model' not in st.session_state:
        st.session_state.cluster_kmeans_model = None
    if 'cluster_scaler' not in st.session_state:
        st.session_state.cluster_scaler = None
    if 'cluster_selected_features' not in st.session_state:
        st.session_state.cluster_selected_features = []
    if 'cluster_analysis_df' not in st.session_state:
        st.session_state.cluster_analysis_df = None
    if 'cluster_pca_df' not in st.session_state:
        st.session_state.cluster_pca_df = None
    if 'cluster_plot_2d_df' not in st.session_state:
        st.session_state.cluster_plot_2d_df = None
    if 'cluster_selected_k' not in st.session_state:
        st.session_state.cluster_selected_k = 3
    if 'cluster_elbow_fig' not in st.session_state:
        st.session_state.cluster_elbow_fig = None
    if 'cluster_silhouette_fig' not in st.session_state:
        st.session_state.cluster_silhouette_fig = None
    if 'cluster_X_scaled_intermediate' not in st.session_state:
        st.session_state.cluster_X_scaled_intermediate = None
    if 'cluster_scaler_intermediate' not in st.session_state:
        st.session_state.cluster_scaler_intermediate = None
    if 'cluster_selected_features_intermediate' not in st.session_state:
        st.session_state.cluster_selected_features_intermediate = []
    if 'cluster_original_features_with_labels' not in st.session_state:
        st.session_state.cluster_original_features_with_labels = None

    df_model_clustering = prepare_model_data(filtered_df.copy())

    if 'manufacturer' in df_model_clustering.columns and df_model_clustering['manufacturer'].dtype == 'object':
        df_model_clustering['manufacturer_code'] = pd.Categorical(df_model_clustering['manufacturer']).codes

    numeric_features_for_clustering = df_model_clustering.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_features_for_clustering) > 0:
        with st.form(key="clustering_analysis_form"):
            st.subheader("Select Features & Parameters for Clustering")
            current_cluster_selected_features = st.multiselect(
                "Select features for clustering",
                numeric_features_for_clustering,
                default=st.session_state.cluster_selected_features_intermediate or numeric_features_for_clustering[
                                                                                   :min(5,
                                                                                        len(numeric_features_for_clustering))],
                key="clustering_current_features_multiselect_form"
            )

            st.markdown("###### Optimal k Evaluation Parameters")
            col1_cluster_form, col2_cluster_form = st.columns(2)
            with col1_cluster_form:
                max_k_to_eval_form = st.slider("Max clusters to evaluate (k)", 2, 15, 10,
                                               key="clustering_max_k_form_slider")
            with col2_cluster_form:
                random_state_cluster_form = st.slider("Random seed (for K-Means evaluation)", 0, 100, 42,
                                                      key="clustering_seed_form_slider")

            run_clustering_button_form = st.form_submit_button("Run Clustering Analysis & Determine Optimal k")

        if run_clustering_button_form and current_cluster_selected_features:
            with st.spinner("Performing clustering analysis and determining optimal k..."):
                try:
                    X_cluster = df_model_clustering[current_cluster_selected_features].copy()
                    X_cluster = X_cluster.fillna(X_cluster.median())

                    scaler_instance = StandardScaler()
                    X_scaled = scaler_instance.fit_transform(X_cluster)

                    st.session_state.cluster_X_scaled_intermediate = X_scaled
                    st.session_state.cluster_scaler_intermediate = scaler_instance
                    st.session_state.cluster_selected_features_intermediate = current_cluster_selected_features

                    inertia_values, silhouette_scores_list = [], []
                    k_range = range(2, max_k_to_eval_form + 1)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, k_val in enumerate(k_range):
                        status_text.text(f"Evaluating k={k_val}...")
                        kmeans_eval = KMeans(n_clusters=k_val, random_state=random_state_cluster_form, n_init='auto')
                        cluster_labels_eval = kmeans_eval.fit_predict(X_scaled)
                        inertia_values.append(kmeans_eval.inertia_)
                        if k_val >= 2:
                            silhouette_scores_list.append(silhouette_score(X_scaled, cluster_labels_eval))
                        else:
                            silhouette_scores_list.append(np.nan)
                        progress_bar.progress((i + 1) / len(k_range))

                    status_text.text("Optimal k evaluation complete.")
                    progress_bar.empty()

                    st.session_state.cluster_elbow_fig = px.line(
                        x=list(k_range), y=inertia_values, markers=True,
                        title="Elbow Method for Optimal k",
                        labels={"x": "Number of clusters (k)", "y": "Inertia (WCSS)"}
                    )
                    st.session_state.cluster_silhouette_fig = px.line(
                        x=list(k_range), y=silhouette_scores_list, markers=True,
                        title="Silhouette Scores for Different k",
                        labels={"x": "Number of clusters (k)", "y": "Silhouette Score"}
                    )

                    valid_sil_scores = [s for s in silhouette_scores_list if not np.isnan(s)]
                    if valid_sil_scores:
                        best_k_sil_idx = np.argmax(valid_sil_scores)
                        st.session_state.cluster_selected_k = k_range[
                            silhouette_scores_list.index(valid_sil_scores[best_k_sil_idx])]
                    else:
                        st.session_state.cluster_selected_k = 3

                    st.success(
                        f"Optimal k evaluation done. Suggested k based on Silhouette: {st.session_state.cluster_selected_k}")

                    st.session_state.cluster_kmeans_model = None
                    st.session_state.cluster_analysis_df = None
                    st.session_state.cluster_pca_df = None
                    st.session_state.cluster_plot_2d_df = None
                    st.session_state.cluster_original_features_with_labels = None

                except Exception as e:
                    st.error(f"Error during optimal k evaluation: {str(e)}")
        elif run_clustering_button_form and not current_cluster_selected_features:
            st.warning("Please select at least one feature for clustering.")

        if st.session_state.cluster_elbow_fig and st.session_state.cluster_silhouette_fig:
            st.subheader("Determining Optimal Number of Clusters")
            st.plotly_chart(st.session_state.cluster_elbow_fig, use_container_width=True)
            st.plotly_chart(st.session_state.cluster_silhouette_fig, use_container_width=True)

            final_k_selection_val = st.number_input(
                "Select final number of clusters (k) for analysis:",
                min_value=2,
                max_value=max_k_to_eval_form if 'max_k_to_eval_form' in locals() else 15,
                value=int(st.session_state.cluster_selected_k),
                key="cluster_final_k_selection_input_widget"
            )

            if st.button("Apply K-means with Selected k & Analyze Clusters", key="apply_final_kmeans_button_widget"):
                if 'cluster_X_scaled_intermediate' in st.session_state and st.session_state.cluster_X_scaled_intermediate is not None:
                    with st.spinner(f"Applying K-Means with {final_k_selection_val} clusters..."):
                        try:
                            X_scaled_final = st.session_state.cluster_X_scaled_intermediate
                            scaler_final = st.session_state.cluster_scaler_intermediate
                            selected_features_final = st.session_state.cluster_selected_features_intermediate
                            final_random_state = random_state_cluster_form if 'random_state_cluster_form' in locals() else 42

                            kmeans_final_model_instance = KMeans(n_clusters=final_k_selection_val,
                                                                 random_state=final_random_state, n_init='auto')
                            final_cluster_labels = kmeans_final_model_instance.fit_predict(X_scaled_final)

                            st.session_state.cluster_kmeans_model = kmeans_final_model_instance
                            st.session_state.cluster_scaler = scaler_final
                            st.session_state.cluster_selected_features = selected_features_final
                            st.session_state.cluster_selected_k = final_k_selection_val

                            analysis_df_final = filtered_df.copy()
                            analysis_df_final['Cluster'] = final_cluster_labels[:len(analysis_df_final)]
                            st.session_state.cluster_analysis_df = analysis_df_final

                            X_cluster_original_for_means = df_model_clustering[selected_features_final].copy().fillna(
                                df_model_clustering[selected_features_final].median())
                            X_cluster_original_for_means = X_cluster_original_for_means.iloc[:len(final_cluster_labels)]
                            X_cluster_original_for_means['Cluster'] = final_cluster_labels
                            st.session_state.cluster_original_features_with_labels = X_cluster_original_for_means

                            if X_scaled_final.shape[1] > 2:
                                pca = PCA(n_components=2, random_state=final_random_state)
                                X_pca = pca.fit_transform(X_scaled_final)
                                pca_df_res = pd.DataFrame(
                                    {'PCA Component 1': X_pca[:, 0], 'PCA Component 2': X_pca[:, 1],
                                     'Cluster': final_cluster_labels})
                                if 'manufacturer' in analysis_df_final.columns:
                                    pca_df_res['Manufacturer'] = analysis_df_final['manufacturer'].values[
                                                                 :len(pca_df_res)]
                                st.session_state.cluster_pca_df = pca_df_res
                                st.session_state.cluster_plot_2d_df = None
                            elif X_scaled_final.shape[1] == 2:
                                plot_df_2d_res = pd.DataFrame({
                                    selected_features_final[0]: X_scaled_final[:, 0],
                                    selected_features_final[1]: X_scaled_final[:, 1],
                                    'Cluster': final_cluster_labels
                                })
                                if 'manufacturer' in analysis_df_final.columns:
                                    plot_df_2d_res['Manufacturer'] = analysis_df_final['manufacturer'].values[
                                                                     :len(plot_df_2d_res)]
                                st.session_state.cluster_plot_2d_df = plot_df_2d_res
                                st.session_state.cluster_pca_df = None

                            st.success(
                                f"K-Means applied with {final_k_selection_val} clusters. Results are now available for analysis.")

                        except Exception as e:
                            st.error(f"Error applying final K-Means: {str(e)}")
                            st.exception(e)
                else:
                    st.warning(
                        "Please run 'Run Clustering Analysis & Determine Optimal k' first to prepare data before applying final K-Means.")

        if st.session_state.cluster_kmeans_model is not None and st.session_state.cluster_analysis_df is not None:
            st.subheader(f"Cluster Analysis Results (k={st.session_state.cluster_selected_k})")

            if st.session_state.cluster_pca_df is not None:
                st.markdown("##### Cluster Visualization (PCA - 2 Components)")
                fig_pca_res = px.scatter(
                    st.session_state.cluster_pca_df, x='PCA Component 1', y='PCA Component 2', color='Cluster',
                    hover_data=['Manufacturer'] if 'Manufacturer' in st.session_state.cluster_pca_df.columns else None,
                    title="Cluster Visualization using PCA", color_discrete_sequence=px.colors.qualitative.Plotly
                )
                st.plotly_chart(fig_pca_res, use_container_width=True)
            elif st.session_state.cluster_plot_2d_df is not None and st.session_state.cluster_selected_features and len(
                    st.session_state.cluster_selected_features) == 2:
                st.markdown("##### Cluster Visualization (Original 2 Scaled Features)")
                features_for_2d_plot = st.session_state.cluster_selected_features
                fig_2d_res = px.scatter(
                    st.session_state.cluster_plot_2d_df, x=features_for_2d_plot[0], y=features_for_2d_plot[1],
                    color='Cluster',
                    hover_data=[
                        'Manufacturer'] if 'Manufacturer' in st.session_state.cluster_plot_2d_df.columns else None,
                    title=f"Cluster Visualization ({features_for_2d_plot[0]} vs {features_for_2d_plot[1]})",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                st.plotly_chart(fig_2d_res, use_container_width=True)

            st.markdown("##### Cluster Characteristics")
            analysis_df_display = st.session_state.cluster_analysis_df
            cluster_counts = analysis_df_display['Cluster'].value_counts().sort_index()
            fig_counts_res = px.bar(
                cluster_counts, x=cluster_counts.index, y=cluster_counts.values,
                title="Number of Smartphones in Each Cluster",
                labels={'x': 'Cluster', 'y': 'Count'}, color=cluster_counts.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_counts_res.update_layout(xaxis_title="Cluster")
            st.plotly_chart(fig_counts_res, use_container_width=True)

            if st.session_state.cluster_original_features_with_labels is not None and st.session_state.cluster_selected_features:
                mean_features_display = st.session_state.cluster_original_features_with_labels.groupby('Cluster')[
                    st.session_state.cluster_selected_features].mean().round(2)
                st.markdown("###### Mean Feature Values by Cluster (Original Feature Scales)")
                st.dataframe(mean_features_display, use_container_width=True)

                cluster_centers_scaled_res = st.session_state.cluster_kmeans_model.cluster_centers_
                radar_df_scaled_res = pd.DataFrame(cluster_centers_scaled_res,
                                                   columns=st.session_state.cluster_selected_features)

                cluster_to_profile = st.selectbox(
                    "Select a cluster to view its profile (Radar Chart - Scaled Centroids)",
                    options=sorted(analysis_df_display['Cluster'].unique()),
                    key="clustering_radar_select_results_widget"
                )
                if cluster_to_profile is not None:
                    fig_radar_res = px.line_polar(
                        radar_df_scaled_res.iloc[[cluster_to_profile]],
                        r=radar_df_scaled_res.iloc[cluster_to_profile].values,
                        theta=st.session_state.cluster_selected_features,
                        line_close=True,
                        title=f"Profile of Cluster {cluster_to_profile} (Scaled Feature Centroids)"
                    )
                    fig_radar_res.update_traces(fill='toself')
                    st.plotly_chart(fig_radar_res, use_container_width=True)

            if 'manufacturer' in analysis_df_display.columns:
                st.markdown("###### Top Manufacturers in Each Cluster")
                for cluster_num_res in sorted(analysis_df_display['Cluster'].unique()):
                    st.markdown(f"**Cluster {cluster_num_res}:**")
                    top_mfrs_res = analysis_df_display[analysis_df_display['Cluster'] == cluster_num_res][
                        'manufacturer'].value_counts().nlargest(3)
                    if not top_mfrs_res.empty:
                        st.table(top_mfrs_res)
                    else:
                        st.write("No manufacturer data or no items in this cluster.")

            if 'price' in analysis_df_display.columns:
                st.markdown("###### Price Distribution Across Clusters")
                fig_price_cluster_res = px.box(
                    analysis_df_display, x='Cluster', y='price', color='Cluster', title="Price Distribution by Cluster",
                    labels={'price': 'Price (RON)'}, color_discrete_sequence=px.colors.qualitative.Plotly
                )
                st.plotly_chart(fig_price_cluster_res, use_container_width=True)

        elif st.session_state.cluster_elbow_fig and st.session_state.cluster_silhouette_fig and st.session_state.cluster_kmeans_model is None:
            st.info(
                "Optimal k plots are shown above. Please select a final 'k' and click 'Apply K-means with Selected k & Analyze Clusters' to see detailed cluster analysis.")
        elif not run_clustering_button_form and not (
                st.session_state.cluster_elbow_fig and st.session_state.cluster_silhouette_fig):
            st.info(
                "Select features and parameters, then run the clustering analysis to determine optimal k and view results.")

    else:
        st.info("No numeric features available for clustering after initial processing. Please check data preparation.")
        for key in list(st.session_state.keys()):
            if key.startswith('cluster_'):
                del st.session_state[key]
