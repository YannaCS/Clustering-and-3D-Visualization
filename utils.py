import pandas as pd
import numpy as np
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
import io

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
    
    # Calculate hierarchical clustering linkage for dendrogram
    Z = linkage(train_data, method='ward')
    
    # Create Plotly visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Elbow Method",
            "Silhouette Score",
            "Calinski-Harabasz Score",
            "Hierarchical Dendrogram"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add title
    fig.update_layout(
        title_text="Determining the Optimal Number of Clusters",
        height=800,
        width=1200
    )
    
    # 1. Elbow plot
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add vertical line for best k from elbow method
    fig.add_shape(
        type="line",
        x0=best_k_values['elbow'], y0=0,
        x1=best_k_values['elbow'], y1=max(inertias),
        line=dict(color="red", width=2, dash="dash"),
        row=1, col=1
    )
    
    fig.add_annotation(
        x=best_k_values['elbow'],
        y=inertias[best_k_values['elbow']-1] if best_k_values['elbow'] <= len(inertias) else inertias[-1],
        text=f"Best k={best_k_values['elbow']}",
        showarrow=True,
        arrowhead=2,
        row=1, col=1
    )
    
    # 2. Silhouette plot
    fig.add_trace(
        go.Scatter(
            x=list(range(2, max_k + 1)),
            y=silhouettes,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # Add first derivative to silhouette plot if possible
    if len(silhouettes) > 1:
        first_deriv = np.diff(silhouettes)
        # Normalize to fit on same scale
        max_silhouette = max(abs(np.max(silhouettes)), 0.001)  # Avoid division by zero
        norm_deriv = first_deriv * (max_silhouette / max(abs(first_deriv) + 0.001)) * 0.5
        
        fig.add_trace(
            go.Scatter(
                x=list(range(3, max_k + 1)),
                y=norm_deriv,
                mode='lines',
                name='Rate of Change',
                line=dict(color='lightgreen', dash='dash')
            ),
            row=1, col=2
        )
    
    # Add vertical line for best k from silhouette method
    if best_k_values['silhouette'] >= 2 and best_k_values['silhouette'] <= max_k:
        silhouette_index = best_k_values['silhouette'] - 2  # convert to index (k=2 is index 0)
        if silhouette_index < len(silhouettes):
            fig.add_shape(
                type="line",
                x0=best_k_values['silhouette'], y0=0,
                x1=best_k_values['silhouette'], y1=max(silhouettes),
                line=dict(color="red", width=2, dash="dash"),
                row=1, col=2
            )
            
            fig.add_annotation(
                x=best_k_values['silhouette'],
                y=silhouettes[silhouette_index],
                text=f"Best k={best_k_values['silhouette']}",
                showarrow=True,
                arrowhead=2,
                row=1, col=2
            )
    
    # 3. CH plot
    fig.add_trace(
        go.Scatter(
            x=list(range(2, max_k + 1)),
            y=ch_coeffs,
            mode='lines+markers',
            name='CH Score',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # Add first derivative to CH plot if possible
    if len(ch_coeffs) > 1:
        first_deriv = np.diff(ch_coeffs)
        # Normalize to fit on same scale
        max_ch = max(abs(np.max(ch_coeffs)), 0.001)  # Avoid division by zero
        norm_deriv = first_deriv * (max_ch / max(abs(first_deriv) + 0.001)) * 0.5
        
        fig.add_trace(
            go.Scatter(
                x=list(range(3, max_k + 1)),
                y=norm_deriv,
                mode='lines',
                name='Rate of Change',
                line=dict(color='mediumpurple', dash='dash')
            ),
            row=2, col=1
        )
    
    # Add vertical line for best k from CH method
    if best_k_values['ch'] >= 2 and best_k_values['ch'] <= max_k:
        ch_index = best_k_values['ch'] - 2  # convert to index (k=2 is index 0)
        if ch_index < len(ch_coeffs):
            fig.add_shape(
                type="line",
                x0=best_k_values['ch'], y0=0,
                x1=best_k_values['ch'], y1=max(ch_coeffs),
                line=dict(color="red", width=2, dash="dash"),
                row=2, col=1
            )
            
            fig.add_annotation(
                x=best_k_values['ch'],
                y=ch_coeffs[ch_index],
                text=f"Best k={best_k_values['ch']}",
                showarrow=True,
                arrowhead=2,
                row=2, col=1
            )
    
    # 4. Dendrogram - This is trickier in Plotly, will create a simplified version
    # Extract the coordinates for the dendrogram lines
    icoord = []
    dcoord = []
    color_list = []
    
    # Convert scipy dendrogram to plotly format
    def _get_dendrogram_data(Z, show_labels=False):
        from scipy.cluster.hierarchy import dendrogram
        ddata = dendrogram(Z, no_plot=True)
        
        x = []
        y = []
        
        line_width = []
        
        for i, d in enumerate(ddata['dcoord']):
            x.extend(ddata['icoord'][i])
            x.append(None)
            y.extend(d)
            y.append(None)
            
            # Line width based on height
            line_width.extend([2] * 4)  # 4 points per linkage
            line_width.append(None)  # Break between linkages
        
        return x, y, line_width
    
    x, y, line_width = _get_dendrogram_data(Z)
    
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='black', width=1),
            hoverinfo='none',
            name='Dendrogram'
        ),
        row=2, col=2
    )
    
    # Add horizontal line for ensemble best k
    if ensemble_best_k:
        # Find the height to cut the tree to get ensemble_best_k clusters
        cut_height = Z[-(ensemble_best_k-1), 2] if ensemble_best_k > 1 and ensemble_best_k <= len(Z) + 1 else 0
        
        fig.add_shape(
            type="line",
            x0=0, y0=cut_height,
            x1=len(train_data) * 10, y1=cut_height,  # Making the line span the full width
            line=dict(color="red", width=2, dash="dash"),
            row=2, col=2
        )
        
        fig.add_annotation(
            x=len(train_data) * 5,  # Middle of the plot
            y=cut_height,
            text=f"Cut for k={ensemble_best_k}",
            showarrow=True,
            arrowhead=2,
            row=2, col=2
        )
    
    # Update titles and labels
    fig.update_xaxes(title_text="K (number of clusters)", row=1, col=1)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    
    fig.update_xaxes(title_text="K (number of clusters)", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_xaxes(title_text="K (number of clusters)", row=2, col=1)
    fig.update_yaxes(title_text="CH Score", row=2, col=1)
    
    fig.update_xaxes(title_text="Sample index", row=2, col=2)
    fig.update_yaxes(title_text="Distance", row=2, col=2)
    
    # Create a clean layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig, silhouettes, ch_coeffs, inertias, ensemble_best_k, best_k_values

def plot_dendrogram_plotly(data, best_k=None):
    """Create hierarchical clustering dendrogram with optimal cut line using Plotly"""
    # Calculate the linkage matrix
    Z = linkage(data, method='ward')
    
    # Convert scipy dendrogram to plotly format
    def _get_dendrogram_data(Z, show_labels=False):
        from scipy.cluster.hierarchy import dendrogram
        ddata = dendrogram(Z, no_plot=True)
        
        x = []
        y = []
        
        line_width = []
        
        for i, d in enumerate(ddata['dcoord']):
            x.extend(ddata['icoord'][i])
            x.append(None)
            y.extend(d)
            y.append(None)
            
            # Line width based on height
            line_width.extend([2] * 4)  # 4 points per linkage
            line_width.append(None)  # Break between linkages
        
        return x, y, line_width
    
    x, y, line_width = _get_dendrogram_data(Z)
    
    # Create figure
    fig = go.Figure()
    
    # Add dendrogram lines
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='black', width=1),
            hoverinfo='none',
            name='Dendrogram'
        )
    )
    
    # If best_k is provided, add a horizontal line showing where to cut for that number of clusters
    if best_k:
        # Find the height to cut the tree to get the desired number of clusters
        cut_height = Z[-(best_k-1), 2] if best_k > 1 and best_k <= len(Z) + 1 else 0
        
        fig.add_shape(
            type="line",
            x0=0, y0=cut_height,
            x1=len(data) * 10, y1=cut_height,  # Making the line span the full width
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.add_annotation(
            x=len(data) * 5,  # Middle of the plot
            y=cut_height,
            text=f"Cut for k={best_k}",
            showarrow=True,
            arrowhead=2
        )
    
    # Update layout
    fig.update_layout(
        title=f"Hierarchical Clustering Dendrogram{f' (Best K: {best_k})' if best_k else ''}",
        xaxis_title="Number of points in node (or index of point if no parenthesis)",
        yaxis_title="Distance",
        height=600,
        width=800
    )
    
    return fig

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

def plot_performance_comparison_plotly(df_kmeans, df_hierarchical, k_value):
    """Create bar chart comparing K-means and Hierarchical clustering performance using Plotly"""
    metrics = ['Pr', 'Recall', 'J', 'Rand', 'FM']
    
    fig = go.Figure()
    
    try:
        # Filter for the specific k-value
        k_idx = f'k={k_value}'
        
        # Check if the k value exists in both DataFrames
        if k_idx in df_kmeans.index and k_idx in df_hierarchical.index:
            kmeans_data = df_kmeans.loc[k_idx, metrics].values
            hierarchical_data = df_hierarchical.loc[k_idx, metrics].values
            
            # Add bars for each algorithm
            fig.add_trace(go.Bar(
                x=metrics,
                y=kmeans_data,
                name='K-Means',
                marker_color='#4285F4', # nice blue
                width=0.4,
                offset=-0.2
            ))
            
            fig.add_trace(go.Bar(
                x=metrics,
                y=hierarchical_data,
                name='Hierarchical',
                marker_color='#e74639', # nice red
                width=0.4,
                offset=0.2
            ))
        else:
            # Use the first available index if the specific k doesn't exist
            kmeans_data = df_kmeans.iloc[0][metrics].values if not df_kmeans.empty else np.zeros(len(metrics))
            hierarchical_data = df_hierarchical.iloc[0][metrics].values if not df_hierarchical.empty else np.zeros(len(metrics))
            
            # Add bars for each algorithm
            fig.add_trace(go.Bar(
                x=metrics,
                y=kmeans_data,
                name='K-Means',
                marker_color='#4285F4',
                width=0.4,
                offset=-0.2
            ))
            
            fig.add_trace(go.Bar(
                x=metrics,
                y=hierarchical_data,
                name='Hierarchical',
                marker_color='#e74639',
                width=0.4,
                offset=0.2
            ))
    except Exception as e:
        print(f"Error creating performance comparison: {str(e)}")
        # Create empty bars in case of error
        fig.add_trace(go.Bar(
            x=metrics,
            y=np.zeros(len(metrics)),
            name='K-Means',
            marker_color='#4285F4',
            width=0.4,
            offset=-0.2
        ))
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=np.zeros(len(metrics)),
            name='Hierarchical',
            marker_color='#e74639',
            width=0.4,
            offset=0.2
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Clustering Performance Comparison (k={k_value})',
        xaxis_title='Performance Metrics',
        yaxis_title='Value',
        barmode='group',
        height=500,
        width=800
    )
    
    return fig

def plot_performance_metrics_plotly(df_performances, method_name="Clustering"):
    """Plot the performance metrics across different k values using Plotly"""
    fig = go.Figure()
    
    # If the DataFrame is empty, return an empty plot
    if df_performances.empty:
        fig.update_layout(
            title=f"No performance data available for {method_name}",
            xaxis_title="Number of clusters",
            yaxis_title="Coefficient",
            height=500,
            width=700
        )
        return fig
    
    # Get x-values based on index (k values)
    x_values = [int(idx.split('=')[1]) for idx in df_performances.index]
    
    # Add a trace for each metric
    colors = px.colors.qualitative.Plotly
    for i, column in enumerate(df_performances.columns):
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df_performances[column].values,
            mode='lines+markers',
            name=column,
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Performance Metrics for {method_name}',
        xaxis_title='Number of clusters (k)',
        yaxis_title='Coefficient',
        height=500,
        width=700,
        hovermode="x unified"
    )
    
    # Configure axes
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_values,
        ticktext=[f'k={k}' for k in x_values]
    )
    
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
        loc=[2, 0, 0], 
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
    combined_plot_plotly, silhouettes, ch_scores, inertias, ensemble_best_k, best_k_values = choose_k(features)
    
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
        performance_fig_plotly = plot_performance_comparison_plotly(
            kmeans_performances, hierarchical_performances, best_k
        )
        
        # Performance metrics plots
        kmeans_metrics_fig_plotly = plot_performance_metrics_plotly(kmeans_performances, "K-means")
        hierarchical_metrics_fig_plotly = plot_performance_metrics_plotly(hierarchical_performances, "Hierarchical")
    except Exception as e:
        return None, f"Error generating visualizations: {str(e)}"
    
    # Store original data for derivative toggles
    optimal_clusters_data = {
        'inertias': inertias,
        'silhouettes': silhouettes,
        'ch_scores': ch_scores
    }
    
    results = {
        'combined_plot_plotly': combined_plot_plotly,
        'optimal_clusters_data': optimal_clusters_data, # Store raw data for derivative toggles
        'silhouettes': silhouettes,
        'ch_scores': ch_scores,
        'inertias': inertias,
        'kmeans_performances': kmeans_performances,
        'hierarchical_performances': hierarchical_performances,
        'comparison_fig': comparison_fig,
        'performance_fig_plotly': performance_fig_plotly,
        'kmeans_metrics_fig_plotly': kmeans_metrics_fig_plotly,
        'hierarchical_metrics_fig_plotly': hierarchical_metrics_fig_plotly,
        'best_k_kmeans': best_k_kmeans_value,
        'best_k_hierarchical': best_k_hierarchical_value,
        'best_k': best_k,
        'ensemble_best_k': ensemble_best_k,
        'best_k_values': best_k_values
    }
    
    return results, None  # None means no errordef create_optimal_clusters_plot(inertias, silhouettes, ch_scores, best_k_values, ensemble_best_k, 
                          # show_elbow_derivative=False, show_silhouette_derivative=False, show_ch_derivative=False):
    """Create a plotly visualization of optimal clusters with togglable derivatives"""
    # Calculate max k value
    max_k = len(inertias)
    k_range = range(1, max_k + 1)
    
    # Create the subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Elbow Method",
            "Silhouette Score",
            "Calinski-Harabasz Score",
            "Hierarchical Dendrogram"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add title
    fig.update_layout(
        title_text="Determining the Optimal Number of Clusters",
        height=800,
        width=1200
    )
    
    # 1. Elbow plot
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue'),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Add rate of change for Elbow method if toggled
    if show_elbow_derivative and len(inertias) > 1:
        # Calculate first derivative
        elbow_derivatives = np.diff(inertias)
        # Normalize to make it visible on the same scale
        max_inertia = max(abs(np.max(inertias)), 0.001)  # Avoid division by zero
        norm_elbow_deriv = elbow_derivatives * (max_inertia / max(abs(elbow_derivatives) + 0.001)) * 0.5
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, max_k)),  # One less point for derivative
                y=norm_elbow_deriv,
                mode='lines+markers',
                name='Elbow Rate of Change',
                line=dict(color='darkblue', dash='dash'),
                marker=dict(size=6),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Add vertical line for best k from elbow method
    fig.add_shape(
        type="line",
        x0=best_k_values['elbow'], y0=0,
        x1=best_k_values['elbow'], y1=max(inertias),
        line=dict(color="red", width=2, dash="dash"),
        row=1, col=1
    )
    
    fig.add_annotation(
        x=best_k_values['elbow'],
        y=inertias[best_k_values['elbow']-1] if best_k_values['elbow'] <= len(inertias) else inertias[-1],
        text=f"Best k={best_k_values['elbow']}",
        showarrow=True,
        arrowhead=2,
        row=1, col=1
    )
    
    # 2. Silhouette plot
    fig.add_trace(
        go.Scatter(
            x=list(range(2, max_k + 1)),
            y=silhouettes,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='green'),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Add rate of change for Silhouette method if toggled
    if show_silhouette_derivative and len(silhouettes) > 1:
        # Calculate first derivative
        silhouette_derivatives = np.diff(silhouettes)
        # Normalize to fit on same scale
        max_silhouette = max(abs(np.max(silhouettes)), 0.001)  # Avoid division by zero
        norm_silhouette_deriv = silhouette_derivatives * (max_silhouette / max(abs(silhouette_derivatives) + 0.001)) * 0.5
        
        fig.add_trace(
            go.Scatter(
                x=list(range(3, max_k + 1)),  # Starts at k=3 for silhouette derivative
                y=norm_silhouette_deriv,
                mode='lines+markers',
                name='Silhouette Rate of Change',
                line=dict(color='darkgreen', dash='dash'),
                marker=dict(size=6),
                opacity=0.7
            ),
            row=1, col=2
        )
    
    # Add vertical line for best k from silhouette method
    if best_k_values['silhouette'] >= 2 and best_k_values['silhouette'] <= max_k:
        silhouette_index = best_k_values['silhouette'] - 2  # convert to index (k=2 is index 0)
        if silhouette_index < len(silhouettes):
            fig.add_shape(
                type="line",
                x0=best_k_values['silhouette'], y0=0,
                x1=best_k_values['silhouette'], y1=max(silhouettes),
                line=dict(color="red", width=2, dash="dash"),
                row=1, col=2
            )
            
            fig.add_annotation(
                x=best_k_values['silhouette'],
                y=silhouettes[silhouette_index],
                text=f"Best k={best_k_values['silhouette']}",
                showarrow=True,
                arrowhead=2,
                row=1, col=2
            )
    
    # 3. CH plot
    fig.add_trace(
        go.Scatter(
            x=list(range(2, max_k + 1)),
            y=ch_scores,
            mode='lines+markers',
            name='CH Score',
            line=dict(color='purple'),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Add rate of change for CH method if toggled
    if show_ch_derivative and len(ch_scores) > 1:
        # Calculate first derivative
        ch_derivatives = np.diff(ch_scores)
        # Normalize to fit on same scale
        max_ch = max(abs(np.max(ch_scores)), 0.001)  # Avoid division by zero
        norm_ch_deriv = ch_derivatives * (max_ch / max(abs(ch_derivatives) + 0.001)) * 0.5
        
        fig.add_trace(
            go.Scatter(
                x=list(range(3, max_k + 1)),  # Starts at k=3 for CH derivative
                y=norm_ch_deriv,
                mode='lines+markers',
                name='CH Rate of Change',
                line=dict(color='darkviolet', dash='dash'),
                marker=dict(size=6),
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Add vertical line for best k from CH method
    if best_k_values['ch'] >= 2 and best_k_values['ch'] <= max_k:
        ch_index = best_k_values['ch'] - 2  # convert to index (k=2 is index 0)
        if ch_index < len(ch_scores):
            fig.add_shape(
                type="line",
                x0=best_k_values['ch'], y0=0,
                x1=best_k_values['ch'], y1=max(ch_scores),
                line=dict(color="red", width=2, dash="dash"),
                row=2, col=1
            )
            
            fig.add_annotation(
                x=best_k_values['ch'],
                y=ch_scores[ch_index],
                text=f"Best k={best_k_values['ch']}",
                showarrow=True,
                arrowhead=2,
                row=2, col=1
            )
    
    # 4. Ensemble decision and simulated dendrogram
    # For simplicity, create a bar chart to show the ensemble decision
    fig.add_trace(
        go.Bar(
            x=['Elbow', 'Silhouette', 'CH', 'Ensemble'],
            y=[best_k_values['elbow'], best_k_values['silhouette'], best_k_values['ch'], ensemble_best_k],
            marker_color=['blue', 'green', 'purple', 'red'],
            text=[f"k={best_k_values['elbow']}", f"k={best_k_values['silhouette']}", 
                 f"k={best_k_values['ch']}", f"k={ensemble_best_k}"],
            name='Best k Values'
        ),
        row=2, col=2
    )
    
    # Update titles and labels
    fig.update_xaxes(title_text="K (number of clusters)", row=1, col=1)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    
    fig.update_xaxes(title_text="K (number of clusters)", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_xaxes(title_text="K (number of clusters)", row=2, col=1)
    fig.update_yaxes(title_text="CH Score", row=2, col=1)
    
    fig.update_xaxes(title_text="Method", row=2, col=2)
    fig.update_yaxes(title_text="Best k Value", row=2, col=2)
    
    # Add annotation for ensemble decision
    fig.add_annotation(
        x='Ensemble', y=ensemble_best_k,
        text=f"Final k={ensemble_best_k}",
        showarrow=True,
        arrowhead=2,
        row=2, col=2
    )
    
    # Create a clean layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def validate_data(data):
    """Validate uploaded data for clustering analysis"""
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        return False, "Data must be a pandas DataFrame."
    
    # Check if data has at least 4 columns (3 features + class)
    if data.shape[1] < 4:
        return False, "Data must have at least 4 columns (3 features + class label)."
    
    # Check if data has enough rows for clustering
    if data.shape[0] < 10:
        return False, "Data must have at least 10 rows for meaningful clustering."
    
    return True, "Data is valid for clustering analysis."

def preprocess_data(data, feature_mapping=None, class_column='Class'):
    """
    Preprocess data for clustering analysis
    
    Parameters:
    - data: The input DataFrame
    - feature_mapping: Dictionary mapping X1, X2, X3 to original column names
    - class_column: The column containing class labels
    
    Returns:
    - Preprocessed DataFrame with columns X1, X2, X3, Class
    """
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # If feature_mapping is provided, use it to transform the data
    if feature_mapping:
        # Create a new DataFrame with the transformed columns
        transformed = pd.DataFrame({
            'X1': result[feature_mapping['X1']],
            'X2': result[feature_mapping['X2']],
            'X3': result[feature_mapping['X3']],
            'Class': result[class_column]
        })
        return transformed
    
    # If no mapping is provided, just ensure 'Class' column is present and named correctly
    if 'Class' not in result.columns and class_column in result.columns:
        result = result.rename(columns={class_column: 'Class'})
    
    return result