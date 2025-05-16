import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.cluster import pair_confusion_matrix, contingency_matrix
import math

def choose_k(train_data):
    """Function to determine optimal number of clusters"""
    inertias = []
    silhouettes = []
    ch_coeffs = []
    
    for k in range(1,11):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(train_data)
        inertias.append(kmeans.inertia_)
        if k > 1:
            try:
                silhouettes.append(silhouette_score(train_data, kmeans.labels_))
                ch_coeffs.append(calinski_harabasz_score(train_data, kmeans.labels_))
            except Exception as e:
                # Handle potential errors with scoring functions
                print(f"Error calculating metrics for k={k}: {str(e)}")
                silhouettes.append(0)
                ch_coeffs.append(0)
    
    fig = plt.figure(figsize=(10, 4))
    grid = plt.GridSpec(1, 3, wspace=0.3)
    
    plt.subplot(grid[0, 0])
    plt.plot(range(1, 11), inertias)
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Inertia (Within-Cluster Distances)")
    plt.title("Elbow Plot")
    
    plt.subplot(grid[0, 1])
    plt.plot(range(2, 11), silhouettes)
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Plot")
    
    plt.subplot(grid[0, 2])
    plt.plot(range(2, 11), ch_coeffs)
    plt.xlabel("K (number of clusters)")
    plt.ylabel("CH Score")
    plt.title("Calinski-Harabasz Plot")
    
    return fig, silhouettes, ch_coeffs, inertias

def external_performance_by_pair_confusion(true_label, pred_label, n_clusters):
    """Function to evaluate clustering performance against ground truth"""
    performance = pd.DataFrame(columns=['Pr', 'Recall', 'J', 'Rand', 'FM'])
    
    try:
        # Get confusion matrix
        matrix = pd.DataFrame(pair_confusion_matrix(true_label, pred_label))
        
        # Extract values
        TN = matrix.iloc[0, 0]
        FP = matrix.iloc[0, 1]
        FN = matrix.iloc[1, 0]
        TP = matrix.iloc[1, 1]
        
        # Calculate metrics
        performance.loc[0, 'Pr'] = TP/(TP+FP) if (TP+FP) > 0 else 0
        performance.loc[0, 'Recall'] = TP/(TP+FN) if (TP+FN) > 0 else 0
        performance.loc[0, 'J'] = TP/(TP+FP+FN) if (TP+FP+FN) > 0 else 0
        performance.loc[0, 'Rand'] = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0
        performance.loc[0, 'FM'] = ((TP/(TP+FP))*(TP/(TP+FN)))**(0.5) if (TP+FP)*(TP+FN) > 0 else 0
    except Exception as e:
        print(f"Error calculating performance metrics: {str(e)}")
        # Set default values in case of error
        performance.loc[0, 'Pr'] = 0
        performance.loc[0, 'Recall'] = 0
        performance.loc[0, 'J'] = 0
        performance.loc[0, 'Rand'] = 0
        performance.loc[0, 'FM'] = 0
    
    # Rename the index by n_clusters
    performance.index = pd.Series([f'k={n_clusters}'])
    
    return performance

def plot_clusters_3d(data, labels, title="3D Cluster Visualization"):
    """Function to create 3D scatter plot of clusters"""
    fig = px.scatter_3d(
        data, x=data.columns[0], y=data.columns[1], z=data.columns[2],
        color=labels, title=title, width=800, height=600
    )
    return fig

def create_comparison_fig(original_data, kmeans_results, hierarchical_results, k_value):
    """Create a side-by-side comparison of clustering methods"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Original Classes', f'K-Means (k={k_value})', f'Hierarchical (k={k_value})'),
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
    )
    
    # Original data plot
    fig.add_trace(
        go.Scatter3d(
            x=original_data.iloc[:, 0],
            y=original_data.iloc[:, 1],
            z=original_data.iloc[:, 2],
            mode='markers',
            marker=dict(size=4, color=original_data.iloc[:, 3], colorscale='Viridis'),
            name='Original Classes'
        ),
        row=1, col=1
    )
    
    # K-means plot
    fig.add_trace(
        go.Scatter3d(
            x=original_data.iloc[:, 0],
            y=original_data.iloc[:, 1],
            z=original_data.iloc[:, 2],
            mode='markers',
            marker=dict(size=4, color=kmeans_results, colorscale='Viridis'),
            name='K-Means Clusters'
        ),
        row=1, col=2
    )
    
    # Hierarchical plot
    fig.add_trace(
        go.Scatter3d(
            x=original_data.iloc[:, 0],
            y=original_data.iloc[:, 1],
            z=original_data.iloc[:, 2],
            mode='markers',
            marker=dict(size=4, color=hierarchical_results, colorscale='Viridis'),
            name='Hierarchical Clusters'
        ),
        row=1, col=3
    )
    
    fig.update_layout(height=600, width=1200)
    return fig

def plot_performance_comparison(df_kmeans, df_hierarchical, k_value):
    """Create bar chart comparing K-means and Hierarchical clustering performance"""
    metrics = ['Pr', 'Recall', 'J', 'Rand', 'FM']
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    
    try:
        # Filter for the specific k-value
        k_idx = f'k={k_value}'
        
        # Check if the k value exists in both DataFrames
        if k_idx in df_kmeans.index and k_idx in df_hierarchical.index:
            kmeans_data = df_kmeans.loc[k_idx, metrics].values
            hierarchical_data = df_hierarchical.loc[k_idx, metrics].values
            
            ax.bar(x - width/2, kmeans_data, width, label='K-Means')
            ax.bar(x + width/2, hierarchical_data, width, label='Hierarchical')
        else:
            # Use the first available index if the specific k doesn't exist
            kmeans_data = df_kmeans.iloc[0][metrics].values if not df_kmeans.empty else np.zeros(len(metrics))
            hierarchical_data = df_hierarchical.iloc[0][metrics].values if not df_hierarchical.empty else np.zeros(len(metrics))
            
            ax.bar(x - width/2, kmeans_data, width, label='K-Means')
            ax.bar(x + width/2, hierarchical_data, width, label='Hierarchical')
    except Exception as e:
        print(f"Error creating performance comparison: {str(e)}")
        # Create empty bars in case of error
        ax.bar(x - width/2, np.zeros(len(metrics)), width, label='K-Means')
        ax.bar(x + width/2, np.zeros(len(metrics)), width, label='Hierarchical')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_xlabel('Performance Metrics')
    ax.set_title(f'Clustering Performance Comparison (k={k_value})')
    ax.legend()
    
    return fig

def plot_performance_metrics(df_performances, method_name="Clustering"):
    """Plot the performance metrics across different k values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If the DataFrame is empty, return an empty plot
    if df_performances.empty:
        ax.set_title(f"No performance data available for {method_name}")
        return fig
    
    for column in df_performances.columns:
        ax.plot(range(len(df_performances)), df_performances[column], label=column)
    
    # Set x-axis labels based on actual index values
    ax.set_xticks(range(len(df_performances)))
    ax.set_xticklabels(df_performances.index)
    
    ax.set_ylabel('Coefficient')
    ax.set_xlabel('Number of clusters')
    ax.set_title(f'Performance Metrics for {method_name}')
    ax.legend()
    
    return fig

def generate_synthetic_data1():
    """Generate synthetic Data1 if the original file is not available"""
    np.random.seed(42)
    # Create 400 points for class 1 (centered around origin)
    class1 = np.random.normal(loc=0, scale=10, size=(400, 3))
    # Create 400 points for class 2 (centered away from origin)
    class2 = np.random.normal(loc=10, scale=10, size=(400, 3))
    
    all_data = np.vstack([class1, class2])
    labels = np.array([1] * 400 + [2] * 400)
    
    data1 = pd.DataFrame(all_data, columns=["X1", "X2", "X3"])
    data1["Class"] = labels
    
    return data1

def generate_synthetic_data5():
    """Generate synthetic Data5 if the original file is not available"""
    np.random.seed(42)
    num_clusters = 7
    points_per_cluster = 30
    
    all_data = []
    labels = []
    
    for i in range(num_clusters):
        # Generate cluster centers in a sphere
        angle = i * (2 * np.pi / num_clusters)
        center_x = 2 * np.cos(angle)
        center_y = 2 * np.sin(angle)
        center_z = i % 3 - 1  # Use modulo to place clusters at different z levels
        
        # Generate points around the center
        cluster_points = np.random.normal(
            loc=[center_x, center_y, center_z], 
            scale=0.3, 
            size=(points_per_cluster, 3)
        )
        
        all_data.append(cluster_points)
        labels.extend([i+1] * points_per_cluster)
    
    # Add extra points to first cluster to match the original dataset
    extra_points = np.random.normal(
        loc=[center_x, center_y, center_z], 
        scale=0.3, 
        size=(2, 3)
    )
    all_data.append(extra_points)
    labels.extend([1] * 2)
    
    all_data = np.vstack(all_data)
    
    data5 = pd.DataFrame(all_data, columns=["X1", "X2", "X3"])
    data5["Class"] = labels
    
    return data5

def run_clustering_analysis(data):
    """Run the complete clustering analysis on a dataset"""
    # Create copies to avoid modifying the original
    data_copy = data.copy()
    
    if 'Class' not in data_copy.columns:
        return None, "The dataset must contain a 'Class' column with ground truth labels."
    
    # Extract features and labels
    features = data_copy.drop(columns=['Class'])
    labels = data_copy['Class']
    
    # Determine optimal number of clusters
    k_plot, silhouettes, ch_scores, inertias = choose_k(features)
    
    # Run K-means and hierarchical clustering for k=2 to k=10
    kmeans_performances = pd.DataFrame(columns=['Pr', 'Recall', 'J', 'Rand', 'FM'])
    hierarchical_performances = pd.DataFrame(columns=['Pr', 'Recall', 'J', 'Rand', 'FM'])
    
    max_k = min(10, len(data_copy) - 1)  # Ensure k is less than the number of data points
    
    for k in range(2, max_k + 1):
        try:
            # K-means
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(features)
            kmeans_perf = external_performance_by_pair_confusion(labels, kmeans.labels_, k)
            kmeans_performances = pd.concat([kmeans_performances, kmeans_perf])
            
            # Hierarchical
            hierarchical = AgglomerativeClustering(n_clusters=k)
            hierarchical.fit(features)
            hierarchical_perf = external_performance_by_pair_confusion(labels, hierarchical.labels_, k)
            hierarchical_performances = pd.concat([hierarchical_performances, hierarchical_perf])
        except Exception as e:
            print(f"Error running clustering for k={k}: {str(e)}")
            # Add default performance values in case of error
            default_perf = pd.DataFrame({
                'Pr': [0], 'Recall': [0], 'J': [0], 'Rand': [0], 'FM': [0]
            }, index=[f'k={k}'])
            kmeans_performances = pd.concat([kmeans_performances, default_perf])
            hierarchical_performances = pd.concat([hierarchical_performances, default_perf])
    
    # Default best k value
    best_k = 2
    
    # Find best k-value based on Rand index if data is available
    if not kmeans_performances.empty and not hierarchical_performances.empty:
        try:
            best_k_kmeans = kmeans_performances['Rand'].idxmax()
            best_k_hierarchical = hierarchical_performances['Rand'].idxmax()
            best_k_kmeans_value = int(best_k_kmeans.split('=')[1])
            best_k_hierarchical_value = int(best_k_hierarchical.split('=')[1])
            
            # Determine overall best k (compromise between methods if they differ)
            if best_k_kmeans_value == best_k_hierarchical_value:
                best_k = best_k_kmeans_value
            else:
                # Use the mean of the two, rounded
                best_k = round((best_k_kmeans_value + best_k_hierarchical_value) / 2)
        except Exception as e:
            print(f"Error determining best k value: {str(e)}")
            # Default to k=2 if we encounter issues
            best_k = 2
            best_k_kmeans_value = 2
            best_k_hierarchical_value = 2
    else:
        # Default values if performances are empty
        best_k = 2
        best_k_kmeans_value = 2
        best_k_hierarchical_value = 2
    
    # Run K-means and hierarchical with the best k
    try:
        kmeans_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        kmeans_best.fit(features)
        hierarchical_best = AgglomerativeClustering(n_clusters=best_k)
        hierarchical_best.fit(features)
        
        # Create 3D plots
        comparison_fig = create_comparison_fig(
            data_copy, kmeans_best.labels_, hierarchical_best.labels_, best_k
        )
        
        # Performance comparison chart
        performance_fig = plot_performance_comparison(
            kmeans_performances, hierarchical_performances, best_k
        )
        
        # Performance metrics plots
        kmeans_metrics_fig = plot_performance_metrics(kmeans_performances, "K-means")
        hierarchical_metrics_fig = plot_performance_metrics(hierarchical_performances, "Hierarchical")
    except Exception as e:
        return None, f"Error generating visualizations: {str(e)}"
    
    results = {
        'k_plot': k_plot,
        'silhouettes': silhouettes,
        'ch_scores': ch_scores,
        'inertias': inertias,
        'kmeans_performances': kmeans_performances,
        'hierarchical_performances': hierarchical_performances,
        'comparison_fig': comparison_fig,
        'performance_fig': performance_fig,
        'kmeans_metrics_fig': kmeans_metrics_fig,
        'hierarchical_metrics_fig': hierarchical_metrics_fig,
        'best_k_kmeans': best_k_kmeans_value,
        'best_k_hierarchical': best_k_hierarchical_value,
        'best_k': best_k
    }
    
    return results, None  # None means no error