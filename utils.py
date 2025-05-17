import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.cluster import pair_confusion_matrix, contingency_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import argrelextrema
import math
from kneed import KneeLocator

def find_peak_or_elbow(values, method="peak", consider_rate_of_change=True):
    """
    Finds the peak or elbow point in an array of values, considering the rate of change.
    
    Parameters:
    - values: array of metric values for different k
    - method: "peak" to find peak value, "elbow" to find elbow point
    - consider_rate_of_change: If True, prioritize points with significant rate of change
    
    Returns:
    - index of peak/elbow (corresponding to k value)
    """
    if len(values) < 3:
        return 0  # Not enough values to find a peak/elbow
        
    if method == "peak":
        # Calculate the first derivative (rate of change)
        derivatives = np.diff(values)
        
        # If we need to consider rate of change
        if consider_rate_of_change and len(derivatives) >= 2:
            # Find the points where derivative changes sign (peaks in original function)
            sign_changes = np.where(np.diff(np.signbit(derivatives)))[0]
            
            # If we found sign changes (peaks)
            if len(sign_changes) > 0:
                # Get the actual indices in original values
                peak_indices = sign_changes + 1
                
                # Calculate score for each peak combining the height and derivative magnitude
                peak_scores = []
                max_value = max(values)
                max_deriv = max(abs(derivatives)) if max(abs(derivatives)) > 0 else 1
                
                for i in peak_indices:
                    if i > 0 and i < len(values) - 1:
                        # Normalize the values to 0-1 range
                        value_score = values[i] / max_value
                        
                        # Calculate average rate of change before this point
                        if i > 1:
                            before_deriv = abs(np.mean(derivatives[:i-1]))
                        else:
                            before_deriv = abs(derivatives[0])
                        
                        # Calculate rate of change after this point (if available)
                        if i < len(derivatives):
                            after_deriv = abs(derivatives[i])
                        else:
                            after_deriv = 0
                        
                        # The difference in derivatives indicates how "sharp" the peak is
                        deriv_diff = abs(before_deriv - after_deriv) / max_deriv
                        
                        # Combine metrics: value itself and change in rate
                        # We want high values with sharp changes in rate
                        combined_score = value_score * (0.5 + 0.5 * deriv_diff)
                        peak_scores.append((i, combined_score))
                
                # Get the peak with the highest combined score
                if peak_scores:
                    best_idx, _ = max(peak_scores, key=lambda x: x[1])
                    return best_idx
            
            # If no peaks found by derivative sign change, find the point with the biggest 
            # drop in derivative (where it starts slowing down the most)
            deriv_drops = np.diff(derivatives)
            if len(deriv_drops) > 0:
                # Find the biggest drop (most negative second derivative)
                biggest_drop_idx = np.argmin(deriv_drops) + 1
                # Only use this if it corresponds to a reasonably high value
                if values[biggest_drop_idx] > 0.7 * max(values):
                    return biggest_drop_idx
        
        # Fallback: find local maxima
        local_max_indices = list(argrelextrema(np.array(values), np.greater)[0])
        
        if not local_max_indices:
            # If no local maxima, just return the global maximum
            return np.argmax(values)
        else:
            # Return the index of the highest local maximum
            max_idx = max(local_max_indices, key=lambda i: values[i])
            return max_idx
    
    elif method == "elbow":
        try:
            # Use KneeLocator to find the elbow
            knee = KneeLocator(
                range(len(values)), values, curve='convex', 
                direction='increasing' if values[-1] > values[0] else 'decreasing'
            )
            if knee.knee is not None:
                return knee.knee
        except Exception as e:
            print(f"Error using KneeLocator: {str(e)}")
        
        # Fallback method: find where the rate of change significantly decreases
        if values[-1] > values[0]:  # Increasing values
            # Calculate derivatives
            derivatives = np.diff(values)
            for i in range(1, len(derivatives)):
                # If the derivative drops by more than 50% of the max derivative
                if derivatives[i] < 0.5 * np.max(derivatives[:i]):
                    return i
        else:  # Decreasing values
            derivatives = np.diff(values)
            for i in range(1, len(derivatives)):
                # If the derivative becomes less negative by more than 50% of the min derivative
                if derivatives[i] > 0.5 * np.min(derivatives[:i]):
                    return i
        
        # If no clear elbow is found, return the index of the max/min value
        return np.argmax(values) if values[-1] > values[0] else np.argmin(values)
    
    else:
        raise ValueError(f"Unknown method: {method}")

def choose_k(train_data):
    """Function to determine optimal number of clusters with automated detection"""
    inertias = []
    silhouettes = []
    ch_coeffs = []
    
    # Get maximum K value based on data size
    max_k = min(10, len(train_data) - 1)
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(train_data)
        inertias.append(kmeans.inertia_)
        if k > 1:
            try:
                silhouettes.append(silhouette_score(train_data, kmeans.labels_))
                ch_coeffs.append(calinski_harabasz_score(train_data, kmeans.labels_))
            except Exception as e:
                print(f"Error calculating metrics for k={k}: {str(e)}")
                silhouettes.append(0)
                ch_coeffs.append(0)
    
    # Automate best K detection
    best_k_values = {}
    
    # 1. Elbow Method - Using KneeLocator from kneed package
    try:
        knee = KneeLocator(
            k_range[:-1], inertias[:-1], curve='convex', direction='decreasing'
        )
        best_k_values['elbow'] = knee.elbow if knee.elbow else 2
    except Exception as e:
        print(f"Error in elbow method: {str(e)}")
        # Alternative: Calculate percentage decrease
        if len(inertias) > 2:
            percent_decreases = [(inertias[i-1] - inertias[i])/inertias[i-1] * 100 for i in range(1, len(inertias))]
            # Find where the percentage decrease falls below 20%
            for i, decrease in enumerate(percent_decreases, 2):
                if decrease < 20:
                    best_k_values['elbow'] = i
                    break
            else:
                best_k_values['elbow'] = 2
        else:
            best_k_values['elbow'] = 2
    
    # 2. Silhouette Method - Find peak or where increase rate slows
    if silhouettes:
        peak_idx = find_peak_or_elbow(silhouettes, method="peak")
        best_k_values['silhouette'] = peak_idx + 2  # +2 because silhouette starts at k=2
    else:
        best_k_values['silhouette'] = 2
    
    # 3. Calinski-Harabasz Method - Find peak or where increase rate slows
    if ch_coeffs:
        peak_idx = find_peak_or_elbow(ch_coeffs, method="peak")
        best_k_values['ch'] = peak_idx + 2  # +2 because CH starts at k=2
    else:
        best_k_values['ch'] = 2
    
    # Ensemble approach - Vote or take median
    best_k_counts = {}
    for method, k in best_k_values.items():
        if k in best_k_counts:
            best_k_counts[k] += 1
        else:
            best_k_counts[k] = 1
    
    # If there's a majority vote (at least 2 methods agree), use that k
    max_votes = max(best_k_counts.values())
    if max_votes >= 2:
        for k, votes in best_k_counts.items():
            if votes == max_votes:
                ensemble_best_k = k
                break
    else:
        # If all methods disagree, take the median
        ensemble_best_k = int(np.median(list(best_k_values.values())))
    
    # Create the visualization with improved layout using standard matplotlib
    fig = plt.figure(figsize=(15, 8))
    
    # Main title
    plt.suptitle("Determining the Optimal Number of Clusters", fontsize=16, y=0.98)
    
    # Create the subplots with fixed positions
    ax_kmeans_title = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=1)
    ax_hier_title = plt.subplot2grid((4, 4), (0, 3), colspan=1, rowspan=1)
    
    ax1 = plt.subplot2grid((4, 4), (1, 0), colspan=1, rowspan=3)  # Elbow plot
    ax2 = plt.subplot2grid((4, 4), (1, 1), colspan=1, rowspan=3)  # Silhouette plot
    ax3 = plt.subplot2grid((4, 4), (1, 2), colspan=1, rowspan=3)  # CH plot
    ax4 = plt.subplot2grid((4, 4), (1, 3), colspan=1, rowspan=3)  # Dendrogram
    
    # Add section titles
    ax_kmeans_title.text(0.5, 0.5, "K-means Clustering Evaluation Metrics", 
                     ha='center', va='center', fontsize=14, fontweight='bold')
    ax_kmeans_title.axis('off')
    
    ax_hier_title.text(0.5, 0.5, "Hierarchical Clustering", 
                   ha='center', va='center', fontsize=14, fontweight='bold')
    ax_hier_title.axis('off')
    
    # Elbow plot
    ax1.plot(k_range, inertias)
    ax1.axvline(x=best_k_values['elbow'], color='r', linestyle='--')
    ax1.set_xlabel("K (number of clusters)")
    ax1.set_ylabel("Inertia (Within-Cluster Distances)")
    ax1.set_title(f"Elbow Plot (Best K: {best_k_values['elbow']})")
    
    # Silhouette plot
    ax2.plot(range(2, max_k + 1), silhouettes)
    ax2.axvline(x=best_k_values['silhouette'], color='r', linestyle='--')
    
    # Add first derivative to the silhouette plot
    if len(silhouettes) > 2:
        # Plot the rate of change (first derivative)
        first_deriv = np.diff(silhouettes)
        # Normalize first derivative to fit on the same scale
        max_silhouette = max(abs(np.max(silhouettes)), 0.001)  # Avoid division by zero
        norm_deriv = first_deriv * (max_silhouette / max(abs(first_deriv) + 0.001)) * 0.5
        
        # Plot first derivative
        ax2_2 = ax2.twinx()
        ax2_2.plot(range(3, max_k + 1), norm_deriv, 'g--', alpha=0.5)
        ax2_2.set_ylabel("Rate of Change", color='g')
        ax2_2.tick_params(axis='y', labelcolor='g')
    
    ax2.set_xlabel("K (number of clusters)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title(f"Silhouette Plot (Best K: {best_k_values['silhouette']})")
    
    # CH plot
    ax3.plot(range(2, max_k + 1), ch_coeffs)
    ax3.axvline(x=best_k_values['ch'], color='r', linestyle='--')
    
    # Add first derivative to the CH plot
    if len(ch_coeffs) > 2:
        # Plot the rate of change (first derivative)
        first_deriv = np.diff(ch_coeffs)
        # Normalize first derivative to fit on the same scale
        max_ch = max(abs(np.max(ch_coeffs)), 0.001)  # Avoid division by zero
        norm_deriv = first_deriv * (max_ch / max(abs(first_deriv) + 0.001)) * 0.5
        
        # Plot first derivative
        ax3_2 = ax3.twinx()
        ax3_2.plot(range(3, max_k + 1), norm_deriv, 'g--', alpha=0.5)
        ax3_2.set_ylabel("Rate of Change", color='g')
        ax3_2.tick_params(axis='y', labelcolor='g')
    
    ax3.set_xlabel("K (number of clusters)")
    ax3.set_ylabel("CH Score")
    ax3.set_title(f"CH Plot (Best K: {best_k_values['ch']})")
    
    # Create hierarchical clustering dendrogram
    # Calculate the linkage matrix
    Z = linkage(train_data, method='ward')
    
    # Plot dendrogram
    dendrogram(Z, truncate_mode='level', p=3, ax=ax4)
    
    # If best_k is provided, add a horizontal line showing where to cut for that number of clusters
    if ensemble_best_k:
        # Find the height to cut the tree to get the desired number of clusters
        last_merge = Z[-(ensemble_best_k-1), 2]
        ax4.axhline(y=last_merge, color='r', linestyle='--')
        ax4.set_title(f"Hierarchical Clustering Dendrogram\n(Best K: {ensemble_best_k})")
    else:
        ax4.set_title("Hierarchical Clustering Dendrogram")
    
    ax4.set_xlabel("Node Points")
    ax4.set_ylabel("Distance")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    return fig, silhouettes, ch_coeffs, inertias, ensemble_best_k, best_k_values

def plot_dendrogram(data, best_k=None):
    """Create hierarchical clustering dendrogram with optimal cut line"""
    # Calculate the linkage matrix
    Z = linkage(data, method='ward')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot dendrogram
    dendrogram(Z, truncate_mode='level', p=3)
    
    # If best_k is provided, add a horizontal line showing where to cut for that number of clusters
    if best_k:
        # Find the height to cut the tree to get the desired number of clusters
        last_merge = Z[-(best_k-1), 2]
        plt.axhline(y=last_merge, color='r', linestyle='--')
        plt.title(f"Hierarchical Clustering Dendrogram (Best K: {best_k})")
    else:
        plt.title("Hierarchical Clustering Dendrogram")
    
    plt.xlabel("Number of points in node (or index of point if no parenthesis)")
    plt.ylabel("Distance")
    
    return plt.gcf()

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


def create_comparison_fig(original_data, kmeans_results, hierarchical_results, title_suffix=""):
    """Create a side-by-side comparison of clustering methods"""
    # Determine if title_suffix contains k values
    if isinstance(title_suffix, str) and title_suffix and not title_suffix.startswith("k="):
        subplot_titles = ('Original Classes', f'K-Means ({title_suffix})', f'Hierarchical ({title_suffix})')
    else:
        # Default title with k value
        k_value = title_suffix if isinstance(title_suffix, str) else f"k={title_suffix}" if title_suffix else ""
        subplot_titles = ('Original Classes', f'K-Means {k_value}', f'Hierarchical {k_value}')
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=subplot_titles,
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
    
    # Determine optimal number of clusters with automated detection
    combined_plot, silhouettes, ch_scores, inertias, ensemble_best_k, best_k_values = choose_k(features)
    
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
    
    # Default to the ensemble best_k value
    best_k = ensemble_best_k
    
    # Find best k-value based on Rand index if data is available (as a fallback)
    if not kmeans_performances.empty and not hierarchical_performances.empty:
        try:
            best_k_kmeans = kmeans_performances['Rand'].idxmax()
            best_k_hierarchical = hierarchical_performances['Rand'].idxmax()
            best_k_kmeans_value = int(best_k_kmeans.split('=')[1])
            best_k_hierarchical_value = int(best_k_hierarchical.split('=')[1])
        except Exception as e:
            print(f"Error determining best k value from performance metrics: {str(e)}")
            # Default to ensemble best_k
            best_k_kmeans_value = best_k
            best_k_hierarchical_value = best_k
    else:
        # Default values if performances are empty
        best_k_kmeans_value = best_k
        best_k_hierarchical_value = best_k
    
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
        'combined_plot': combined_plot,
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
        'best_k': best_k,
        'ensemble_best_k': ensemble_best_k,
        'best_k_values': best_k_values
    }
    
    return results, None  # None means no error
    
    results = {
        'k_plot': k_plot,
        'dendrogram_plot': dendrogram_plot,
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
        'best_k': best_k,
        'ensemble_best_k': ensemble_best_k,
        'best_k_values': best_k_values
    }
    
    return results, None  # None means no error