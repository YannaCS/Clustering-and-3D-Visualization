This repository contains a comprehensive clustering analysis implementation using K-means and Hierarchical Clustering algorithms on 3D datasets.

## Overview
The project analyzes two different 3D datasets (Data1 and Data5) using both K-means and Hierarchical Clustering approaches. It includes visualization tools and multiple clustering performance metrics to evaluate and compare the effectiveness of different clustering methods and configurations.

## Features
0. Data Preparation: Preprocessing of 3D data points
1. Multiple Cluster Selection Metrics used to choose best k of k-means clustering:
- Elbow method (inertia)
- Silhouette score
- Calinski-Harabasz index

2. External Performance Validation:
- Precision, Recall
- Jaccard Index
- Rand Index
- Fowlkes-Mallows Score

3. Interactive 3D Visualizations:
- Original data with ground truth labels
- K-means clustering results
- Hierarchical clustering results
- Comparative visualization of all results

4. Dendrogram Analysis: Visual representation of hierarchical clustering

## Results
- Dataset 1: Comparison showed both methods achieve good clustering with slightly better performance for hierarchical clustering at k=5.
- Dataset 5: Both methods achieved perfect clustering at k=7, matching the ground truth classes.

## Conclusion
This study demonstrates that both K-means and Hierarchical clustering algorithms can effectively identify the natural groupings in 3D datasets. Key findings include:
1. For Dataset 1 (with 2 classes), hierarchical clustering at k=5 provided slightly better results than K-means.
<img width="836" alt="image" src="https://github.com/user-attachments/assets/cd2a6d5c-958d-4dc1-91da-c2cbec5e0863" />
<img width="762" alt="image" src="https://github.com/user-attachments/assets/411cb4bc-f445-4594-8532-dc280f8226af" />

3. For Dataset 5 (with 7 classes), both algorithms achieved perfect clustering (precision=1.0, recall=1.0) when k=7, validating that both methods can correctly identify the actual class structure when properly configured.
4. The silhouette score and CH index proved reliable metrics for determining the optimal number of clusters, consistently pointing to the correct value in our experiments.
5. Visualizing clustering results in 3D space provides valuable insights into the strengths and weaknesses of different algorithms and parameter choices.

This work highlights the importance of using multiple validation metrics and visualization techniques when applying unsupervised learning methods to ensure reliable results.

## Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- plotly

## Usage
Run the Jupyter notebook to:
- Load and explore the datasets
- Determine optimal number of clusters
- Run K-means and Hierarchical clustering
- Evaluate clustering performance
- Visualize the results
