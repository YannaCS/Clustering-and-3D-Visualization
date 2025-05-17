import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

import utils

def show():
    """Display the built-in datasets page"""
    st.header("Pre-loaded Dataset Analysis")
    
    # Dataset selection
    dataset_choice = st.radio("Choose a dataset", ["Data1", "Data5"])
    
    if dataset_choice == "Data1":
        # Load Data1 from the provided CSV or create it
        try:
            data1 = pd.read_csv("./data/Data1.csv")
            if "Unnamed: 0" in data1.columns:
                data1 = data1.drop(columns=["Unnamed: 0"])
        except:
            # Generate sample data if CSV not found
            st.info("Using synthetic Data1 as example (original file not found)")
            data1 = utils.generate_synthetic_data1()
        
        # Display dataset info
        with st.expander("Dataset Information"):
            st.write("**Data1**: This dataset consists of 800 points in 3D space, divided into 2 classes.")
            st.dataframe(data1.head())
            st.write(f"Shape: {data1.shape}")
            st.write(f"Class distribution: {data1['Class'].value_counts().to_dict()}")
        
        # Run clustering analysis
        results, error = utils.run_clustering_analysis(data1)
        
    elif dataset_choice == "Data5":
        # Load Data5 from the provided CSV or create it
        try:
            data5 = pd.read_csv("./data/Data5.csv")
            if "Unnamed: 0" in data5.columns:
                data5 = data5.drop(columns=["Unnamed: 0"])
        except:
            # Generate sample data if CSV not found
            st.info("Using synthetic Data5 as example (original file not found)")
            data5 = utils.generate_synthetic_data5()
        
        # Display dataset info
        with st.expander("Dataset Information"):
            st.write("**Data5**: This dataset consists of 212 points in 3D space, divided into 7 classes.")
            st.dataframe(data5.head())
            st.write(f"Shape: {data5.shape}")
            st.write(f"Class distribution: {data5['Class'].value_counts().to_dict()}")
        
        # Run clustering analysis
        results, error = utils.run_clustering_analysis(data5)
    
    # Display results or error
    if error:
        st.error(error)
    elif results:
        data_to_use = None
        if dataset_choice == "Data1" and 'data1' in locals() and data1 is not None:
            data_to_use = data1
        elif dataset_choice == "Data5" and 'data5' in locals() and data5 is not None:
            data_to_use = data5
            
        display_results(results, data_to_use is not None, data_to_use)

def get_clustering_recommendation(kmeans_performances, hierarchical_performances, ensemble_k):
    """
    Generate a recommendation for the best clustering method and k value.
    """
    kmeans_k_idx = f'k={ensemble_k}'
    hierarchical_k_idx = f'k={ensemble_k}'
    
    # Check if the ensemble k exists in both performance dataframes
    if kmeans_k_idx in kmeans_performances.index and hierarchical_k_idx in hierarchical_performances.index:
        # Get performance metrics at ensemble k
        kmeans_metrics = kmeans_performances.loc[kmeans_k_idx]
        hierarchical_metrics = hierarchical_performances.loc[hierarchical_k_idx]
        
        # Compare metrics to determine best method
        # We'll use Rand index and F-measure as the main comparison metrics
        kmeans_score = kmeans_metrics['Rand'] * 0.6 + kmeans_metrics['FM'] * 0.4
        hierarchical_score = hierarchical_metrics['Rand'] * 0.6 + hierarchical_metrics['FM'] * 0.4
        
        # Determine which method is better
        if kmeans_score > hierarchical_score * 1.05:  # 5% better
            recommendation = "K-means"
            score_diff = ((kmeans_score - hierarchical_score) / hierarchical_score) * 100
            reason = f"K-means outperforms Hierarchical clustering by {score_diff:.1f}% on combined performance metrics."
        elif hierarchical_score > kmeans_score * 1.05:  # 5% better
            recommendation = "Hierarchical"
            score_diff = ((hierarchical_score - kmeans_score) / kmeans_score) * 100
            reason = f"Hierarchical clustering outperforms K-means by {score_diff:.1f}% on combined performance metrics."
        else:
            # If they're within 5% of each other, check specific metrics
            if kmeans_metrics['Pr'] > hierarchical_metrics['Pr'] and kmeans_metrics['Recall'] > hierarchical_metrics['Recall']:
                recommendation = "K-means"
                reason = "K-means provides better precision and recall with similar overall performance."
            elif hierarchical_metrics['Pr'] > kmeans_metrics['Pr'] and hierarchical_metrics['Recall'] > kmeans_metrics['Recall']:
                recommendation = "Hierarchical"
                reason = "Hierarchical clustering provides better precision and recall with similar overall performance."
            else:
                # If still unclear, recommend based on computational efficiency
                recommendation = "K-means"
                reason = "Both methods perform similarly, but K-means is generally more computationally efficient."
        
        # Find best k value beyond the ensemble k
        # Sometimes the ensemble might not be the absolute best
        # Use a manual approach to find max value to avoid idxmax() issues
        best_kmeans_k = ensemble_k
        best_hierarchical_k = ensemble_k
        
        try:
            # Manually find the k with the highest Rand index for K-means
            max_rand_kmeans = -1
            for idx in kmeans_performances.index:
                k_val = int(idx.split('=')[1])
                rand_val = kmeans_performances.loc[idx, 'Rand']
                if rand_val > max_rand_kmeans:
                    max_rand_kmeans = rand_val
                    best_kmeans_k = k_val
            
            # Manually find the k with the highest Rand index for Hierarchical
            max_rand_hierarchical = -1
            for idx in hierarchical_performances.index:
                k_val = int(idx.split('=')[1])
                rand_val = hierarchical_performances.loc[idx, 'Rand']
                if rand_val > max_rand_hierarchical:
                    max_rand_hierarchical = rand_val
                    best_hierarchical_k = k_val
        except Exception as e:
            print(f"Error finding best k: {str(e)}")
            # If there's an error, just use the ensemble k
            best_kmeans_k = ensemble_k
            best_hierarchical_k = ensemble_k
        
        # Get the best k for the recommended method
        if recommendation == "K-means":
            best_k = best_kmeans_k
            if best_k != ensemble_k:
                alternate_reason = f"Note: While k={ensemble_k} is the ensemble recommendation, k={best_k} provides the best performance for K-means."
            else:
                alternate_reason = f"The ensemble recommendation of k={ensemble_k} is confirmed as optimal for K-means."
        else:
            best_k = best_hierarchical_k
            if best_k != ensemble_k:
                alternate_reason = f"Note: While k={ensemble_k} is the ensemble recommendation, k={best_k} provides the best performance for Hierarchical clustering."
            else:
                alternate_reason = f"The ensemble recommendation of k={ensemble_k} is confirmed as optimal for Hierarchical clustering."
        
        return {
            "recommended_method": recommendation,
            "optimal_k": best_k,
            "ensemble_k": ensemble_k,
            "reason": reason,
            "alternate_k_note": alternate_reason,
            "kmeans_metrics": kmeans_metrics,
            "hierarchical_metrics": hierarchical_metrics
        }
    
    # Default return if we can't make a proper recommendation
    return {
        "recommended_method": "K-means",
        "optimal_k": ensemble_k,
        "ensemble_k": ensemble_k,
        "reason": "Insufficient data to make a detailed recommendation. K-means is suggested as a default choice.",
        "alternate_k_note": "",
        "kmeans_metrics": None,
        "hierarchical_metrics": None
    }

def create_performance_comparison_plot(kmeans_performances, hierarchical_performances, kmeans_k, hierarchical_k):
    """Create plot comparing performance metrics across all k values with highlights for selected k"""
    metrics = ['Pr', 'Recall', 'J', 'Rand', 'FM']
    
    # Create a figure with two vertical subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("K-means Performance Metrics", "Hierarchical Performance Metrics"),
                        vertical_spacing=0.15,
                        shared_xaxes=True)
    
    # Colors for different metrics
    colors = px.colors.qualitative.Bold
    
    # K-means plot
    for i, metric in enumerate(metrics):
        # Add line for each metric
        fig.add_trace(
            go.Scatter(
                x=list(range(2, len(kmeans_performances) + 2)),  # +2 because k starts at 2
                y=kmeans_performances[metric].values,
                mode='lines+markers',
                name=f"K-means {metric}",
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                legendgroup=metric,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add vertical line and annotation for selected k
        if f'k={kmeans_k}' in kmeans_performances.index:
            k_idx = list(kmeans_performances.index).index(f'k={kmeans_k}')
            y_val = kmeans_performances[metric].iloc[k_idx]
            
            # Vertical line at selected k
            fig.add_shape(
                type="line",
                x0=kmeans_k, y0=0,
                x1=kmeans_k, y1=1,
                yref="paper",
                xref="x",
                line=dict(color="red", width=1, dash="dash"),
                row=1, col=1
            )
            
            # Add annotation for the value at selected k
            fig.add_annotation(
                x=kmeans_k, y=y_val,
                text=f"{y_val:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="black",
                font=dict(size=10),
                row=1, col=1
            )
    
    # Hierarchical plot
    for i, metric in enumerate(metrics):
        # Add line for each metric
        fig.add_trace(
            go.Scatter(
                x=list(range(2, len(hierarchical_performances) + 2)),  # +2 because k starts at 2
                y=hierarchical_performances[metric].values,
                mode='lines+markers',
                name=f"Hierarchical {metric}",
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                legendgroup=metric,
                showlegend=False  # Don't show in legend as it's the same as K-means
            ),
            row=2, col=1
        )
        
        # Add vertical line and annotation for selected k
        if f'k={hierarchical_k}' in hierarchical_performances.index:
            k_idx = list(hierarchical_performances.index).index(f'k={hierarchical_k}')
            y_val = hierarchical_performances[metric].iloc[k_idx]
            
            # Vertical line at selected k
            fig.add_shape(
                type="line",
                x0=hierarchical_k, y0=0,
                x1=hierarchical_k, y1=1,
                yref="paper",
                xref="x",
                line=dict(color="red", width=1, dash="dash"),
                row=2, col=1
            )
            
            # Add annotation for the value at selected k
            fig.add_annotation(
                x=hierarchical_k, y=y_val,
                text=f"{y_val:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="black",
                font=dict(size=10),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=700,
        width=1000,
        title_text=f"Performance Metrics - K-means (k={kmeans_k}) vs Hierarchical (k={hierarchical_k})",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Number of Clusters (k)", row=2, col=1)
    fig.update_yaxes(title_text="Performance Value", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Performance Value", range=[0, 1], row=2, col=1)
    
    return fig

def display_summary(results, has_data=False, data=None):
    """Display clustering summary and recommendations"""
    st.subheader("Summary and Recommendations")
    
    # Get clustering recommendation
    ensemble_k = int(results['ensemble_best_k'])
    recommendation = get_clustering_recommendation(
        results['kmeans_performances'], 
        results['hierarchical_performances'], 
        ensemble_k
    )
    
    # Create a styled container for the recommendation
    st.markdown(f"""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 5px solid #1e90ff;">
        <h4 style="margin-top: 0; color: #1e90ff;">Clustering Recommendation</h4>
        <p><strong>Best Clustering Method:</strong> {recommendation['recommended_method']}</p>
        <p><strong>Optimal Number of Clusters (k):</strong> {recommendation['optimal_k']}</p>
        <p><strong>Ensemble Recommended k:</strong> {recommendation['ensemble_k']}</p>
        <p><strong>Reason:</strong> {recommendation['reason']}</p>
        <p><em>{recommendation['alternate_k_note']}</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add detailed metrics comparison
    st.subheader("Performance Metrics Comparison")
    
    # Show comparison between K-means and Hierarchical at optimal k
    kmeans_k = recommendation['optimal_k']
    hierarchical_k = recommendation['optimal_k']
    
    # Create a comparison table
    if has_data and data is not None:
        # Create and display performance comparison plot
        performance_fig = create_performance_comparison_plot(
            results['kmeans_performances'], 
            results['hierarchical_performances'],
            kmeans_k,
            hierarchical_k
        )
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Add explanation of metrics
        with st.expander("Metrics Explanation"):
            st.write("""
            - **Precision (Pr)**: The ratio of true positive pairs to all pairs predicted to be in the same cluster
            - **Recall**: The ratio of true positive pairs to all pairs that should be in the same cluster
            - **Jaccard Index (J)**: The size of the intersection divided by the size of the union of the sample sets
            - **Rand Index**: The percentage of correct decisions (true positives and true negatives)
            - **Fowlkes-Mallows Score (FM)**: Geometric mean of precision and recall
            
            The red dashed lines mark the optimal k value, with annotations showing the exact metric values at that point.
            """)
    
    # Show the strengths and weaknesses of each clustering method
    st.subheader("Clustering Methods Comparison")
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### K-means Clustering")
        st.markdown("""
        **Strengths:**
        - Generally faster for large datasets
        - Produces more compact, spherical clusters
        - Often performs well on clearly separated data
        - Works better when clusters are roughly equal in size
        
        **Limitations:**
        - Sensitive to initialization
        - Assumes spherical cluster shapes
        - Requires specifying k in advance
        - May converge to local optima
        """)
    
    with cols[1]:
        st.markdown("#### Hierarchical Clustering")
        st.markdown("""
        **Strengths:**
        - Does not require specifying k in advance
        - Produces a dendrogram showing cluster relationships
        - Can handle clusters of different shapes
        - Not affected by initialization
        
        **Limitations:**
        - Computationally expensive for large datasets
        - Cannot adjust clusters once formed
        - Sensitive to outliers
        - Choice of linkage method affects results
        """)
    
    # Add applications section
    st.subheader("Practical Applications")
    st.markdown("""
    **When to use K-means:**
    - Customer segmentation for marketing
    - Image compression
    - Anomaly detection
    - Document clustering
    - Quick exploratory data analysis
    
    **When to use Hierarchical Clustering:**
    - Taxonomy creation
    - Social network analysis
    - Gene expression data analysis
    - When cluster hierarchy visualization is needed
    - When you need to explore different k values without rerunning the algorithm
    """)

def display_results(results, has_data=False, data=None):
    """Display the results of clustering analysis"""
    # Create tabs for organizing results
    tabs = st.tabs(["Optimal Clusters", "3D Visualization", "Summary & Recommendations", "Performance Metrics", "Detailed Results"])
    
    # Optimal Clusters tab
    with tabs[0]:
        st.subheader("Choose K")
        
        # Display combined plot of all evaluation metrics
        st.pyplot(results['combined_plot'])
        
        # Show the best K values
        st.subheader("Automated Best K Detection")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Elbow Method (K-means)", int(results['best_k_values']['elbow']))
        with col2:
            st.metric("Silhouette Method (K-means)", int(results['best_k_values']['silhouette']))
        with col3:
            st.metric("CH Method (K-means)", int(results['best_k_values']['ch']))
        with col4:
            st.metric("Ensemble Best K", int(results['ensemble_best_k']), 
                      help="Combined result from all methods")
        
        with st.expander("Explanation of Methods"):
            st.write("""
            The application uses three popular methods for determining the optimal number of clusters:
            
            1. **Elbow Method (K-means)**: Plots the sum of squared distances from each point to its assigned center.
               The "elbow" point is where adding more clusters provides diminishing returns.
            
            2. **Silhouette Method (K-means)**: Measures how similar points are to their own cluster compared to other clusters.
               Higher values (closer to 1) indicate better defined clusters.
            
            3. **Calinski-Harabasz (CH) Method (K-means)**: Measures the ratio of between-cluster to within-cluster variance.
               Higher values indicate better cluster separation.
            
            4. **Hierarchical Dendrogram**: Shows how points are grouped at different levels in hierarchical clustering.
               The red horizontal line indicates where to cut the tree to get the optimal number of clusters.
            
            5. **Ensemble Approach**: Combines the results from all three K-means methods:
               - If at least two methods agree on a value, that becomes the final k
               - If all three methods disagree, the median value is used
            
            The green dashed lines in the Silhouette and CH plots show the rate of change (first derivative),
            which helps identify where the improvement slows down.
            """)
    
    # 3D Visualization tab
    with tabs[1]:
        st.subheader("3D Visualization of Clusters")
        
        if has_data and data is not None:
            # Get feature data
            features = data.drop(columns=['Class'])
            labels = data['Class']
            
            # Determine available k values
            max_k = min(10, len(data) - 1)
            k_values = list(range(2, max_k + 1))
            
            # Create selection widgets
            col1, col2 = st.columns(2)
            
            # Get clustering recommendation
            ensemble_k = int(results['ensemble_best_k'])
            recommendation = get_clustering_recommendation(
                results['kmeans_performances'], 
                results['hierarchical_performances'], 
                ensemble_k
            )
            
            # Convert NumPy int values to Python int
            elbow_k = int(results['best_k_values']['elbow'])
            silhouette_k = int(results['best_k_values']['silhouette'])
            ch_k = int(results['best_k_values']['ch'])
            ensemble_k = int(results['ensemble_best_k'])
            optimal_k = recommendation['optimal_k']  # Use the recommended optimal k
            
            # K-means clustering selection
            with col1:
                st.markdown("### K-means Clustering")
                # Select the best K from each method to highlight
                kmeans_optimal_k = [elbow_k, silhouette_k, ch_k, ensemble_k, optimal_k]
                
                # Create radio buttons with optimal k values highlighted
                kmeans_options = []
                for k in k_values:
                    option_text = f"k={k}"
                    if k in kmeans_optimal_k:
                        if k == elbow_k:
                            option_text += " (Best from Elbow)"
                        elif k == silhouette_k:
                            option_text += " (Best from Silhouette)"
                        elif k == ch_k:
                            option_text += " (Best from CH)"
                        
                        if k == ensemble_k:
                            option_text += " ⭐ (Ensemble Best)"
                            
                        if k == optimal_k and k != ensemble_k:
                            option_text += " 🏆 (Optimal)"
                    
                    kmeans_options.append(option_text)
                
                # Select the optimal k as default
                optimal_index = k_values.index(optimal_k) if optimal_k in k_values else 0
                
                kmeans_k_option = st.radio(
                    "Select number of clusters for K-means:",
                    options=kmeans_options,
                    index=optimal_index
                )
                kmeans_k = int(kmeans_k_option.split('=')[1].split(' ')[0])  # Extract k value
            
            # Hierarchical clustering selection
            with col2:
                st.markdown("### Hierarchical Clustering")
                # Create radio buttons with optimal k values highlighted
                hierarchical_options = []
                for k in k_values:
                    option_text = f"k={k}"
                    if k == ensemble_k:
                        option_text += " ⭐ (Ensemble Best)"
                    if k == optimal_k and recommendation['recommended_method'] == "Hierarchical":
                        option_text += " 🏆 (Optimal)"
                    
                    hierarchical_options.append(option_text)
                
                # Select the optimal k as default if hierarchical is recommended, otherwise ensemble
                if recommendation['recommended_method'] == "Hierarchical":
                    default_index = k_values.index(optimal_k) if optimal_k in k_values else 0
                else:
                    default_index = k_values.index(ensemble_k) if ensemble_k in k_values else 0
                
                hierarchical_k_option = st.radio(
                    "Select number of clusters for Hierarchical:",
                    options=hierarchical_options,
                    index=default_index
                )
                hierarchical_k = int(hierarchical_k_option.split('=')[1].split(' ')[0])  # Extract k value
            
            # Generate clusters for selected k values
            try:
                # K-means
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=42)
                kmeans.fit(features)
                kmeans_labels = kmeans.labels_
                
                # Hierarchical
                from sklearn.cluster import AgglomerativeClustering
                hierarchical = AgglomerativeClustering(n_clusters=hierarchical_k)
                hierarchical.fit(features)
                hierarchical_labels = hierarchical.labels_
                
                # Create the custom comparison figure
                custom_fig = utils.create_comparison_fig(data, kmeans_labels, hierarchical_labels, 
                                                        f"K-means={kmeans_k}, Hierarchical={hierarchical_k}")
                
                # Display the figure
                st.plotly_chart(custom_fig, use_container_width=True)
                
                # Show performance metrics for these K values as a visual plot
                st.markdown("### Performance Metrics Comparison")
                
                # Create and display performance comparison plot
                performance_fig = create_performance_comparison_plot(
                    results['kmeans_performances'], 
                    results['hierarchical_performances'],
                    kmeans_k,
                    hierarchical_k
                )
                st.plotly_chart(performance_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating clusters: {str(e)}")
                # Fall back to the default visualization
                st.plotly_chart(results['comparison_fig'], use_container_width=True)
        else:
            # Display the default comparison if data is not available
            st.plotly_chart(results['comparison_fig'], use_container_width=True)
        
        with st.expander("Visualization Explanation"):
            st.write("""
            - **Left**: Original class labels from the dataset
            - **Middle**: Clusters identified by K-means algorithm
            - **Right**: Clusters identified by Hierarchical clustering algorithm
            
            Interact with the plots by:
            - Rotating: Click and drag
            - Zooming: Scroll or use the zoom tools
            - Panning: Right-click and drag
            
            You can select different k values for each clustering method using the radio buttons above.
            The recommended values are marked as follows:
            - "Best from..." indicates values identified by a specific method
            - ⭐ (Ensemble Best) is the consensus value from multiple methods
            - 🏆 (Optimal) is the best value based on performance metrics
            
            The performance metrics plot shows how each metric changes with different k values for both algorithms.
            The red dashed lines mark your currently selected k values, with annotations showing the exact metric values at those points.
            """)
    
    # Summary & Recommendations tab
    with tabs[2]:
        display_summary(results, has_data, data)
    
    # Performance Metrics tab
    with tabs[3]:
        st.subheader("Performance Comparison")
        
        # Performance comparison at best K
        st.pyplot(results['performance_fig'])
        
        # Performance across all K values
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("K-means Performance Across K Values")
            st.pyplot(results['kmeans_metrics_fig'])
        
        with col2:
            st.subheader("Hierarchical Performance Across K Values")
            st.pyplot(results['hierarchical_metrics_fig'])
        
        with st.expander("Metrics Explanation"):
            st.write("""
            - **Precision (Pr)**: The ratio of true positive pairs to all pairs predicted to be in the same cluster
            - **Recall**: The ratio of true positive pairs to all pairs that should be in the same cluster
            - **Jaccard Index (J)**: The size of the intersection divided by the size of the union of the sample sets
            - **Rand Index**: The percentage of correct decisions (true positives and true negatives)
            - **Fowlkes-Mallows Score (FM)**: Geometric mean of precision and recall
            """)
    
    # Detailed Results tab
    with tabs[4]:
        st.subheader("Detailed Performance Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("K-means Performance for all K values:")
            st.dataframe(results['kmeans_performances'])
        
        with col2:
            st.write("Hierarchical Clustering Performance for all K values:")
            st.dataframe(results['hierarchical_performances'])