# Clustering Analysis Dashboard
A Streamlit application for analyzing datasets using K-means and Hierarchical clustering algorithms. This tool helps determine the optimal number of clusters and provides detailed visualizations and comparisons of clustering results.  

![cluster_0_10s](https://github.com/user-attachments/assets/693948d0-25f9-40a6-8c74-1107e5dbb740)



## Features
- **Multiple Data Sources**: Analyze built-in sample datasets or upload your own CSV files
- **Interactive 3D Visualizations**: Compare original data with K-means and Hierarchical clustering results
- **Optimal Cluster Detection**: Automated determination of the optimal number of clusters using:
  - Elbow Method
  - Silhouette Score
  - Calinski-Harabasz Method
  - Ensemble approach combining all methods

- **Comprehensive Performance Metrics**:
  - Precision, Recall, Jaccard Index
  - Rand Index
  - Fowlkes-Mallows Score

- **Custom Dataset Mapping**: For uploaded files, map columns to appropriate features

## Installation
1. Clone this repository:  
    git clone https://github.com/yourusername/clustering-analysis.git  
    cd clustering-analysis  

2. Install required dependencies:
pip install -r requirements.txt

3. Run the application:
streamlit run app_all_in_one.py


## Requirements

Python 3.10  
Streamlit 1.25.0+  
Pandas 2.0.1+  
NumPy 1.22.0+  
Plotly 5.10.0+  
Scikit-learn 1.0.2+  
SciPy 1.8.0+  
kneed 0.8.1+  

## Usage
### Analyzing Built-in Datasets

1. Select "Built-in Datasets" from the radio button options
2. Choose between "Data1" (800 points, 2 classes) or "Data5" (212 points, 7 classes)
3. The system will automatically perform clustering analysis and display results

### Analyzing Your Own Data

1. Select "Upload Your Data" from the radio button options
2. Upload a CSV file (must have at least 4 columns: 3 features + 1 class label)
3. Select which column contains the class labels
4. Map your data columns to X1, X2, and X3 features
5. The system will transform your data and perform clustering analysis

### Exploring Results
Results are organized into 4 tabs:

#### 1. Optimal Clusters:

- View plots of Elbow Method, Silhouette Score, and CH Score
- Toggle rate of change (first derivative) visibility using buttons
- See automated best K detection results


#### 2. 3D Visualization:

- Compare original class labels with K-means and Hierarchical clustering
- Select different k values using dropdown menus to see how clusters change
- Default selections show the recommended optimal values

#### 3. Performance Metrics:

- View detailed performance comparisons across all metrics
- Compare K-means and Hierarchical clustering side by side
- Examine how metrics change with different k values


#### 4. Summary & Recommendations:

- Get a data-driven recommendation for the best clustering method
- See which method (K-means or Hierarchical) performs better for your data
- Understand the optimal number of clusters for your dataset



### Understanding Evaluation Metrics

- Precision (Pr): Ratio of true positive pairs to all pairs predicted to be in the same cluster
- Recall: Ratio of true positive pairs to all pairs that should be in the same cluster
- Jaccard Index (J): Size of the intersection divided by the size of the union of the sample sets
- Rand Index: Percentage of correct decisions (true positives and true negatives)
- Fowlkes-Mallows Score (FM): Geometric mean of precision and recall

### How It Works
The application uses an ensemble approach to determine the optimal number of clusters:

- Elbow Method: Identifies where adding more clusters provides diminishing returns
- Silhouette Method: Measures how similar points are to their own cluster compared to other clusters
- Calinski-Harabasz Method: Measures the ratio of between-cluster to within-cluster variance
- Ensemble Decision:
  - If at least two methods agree on a value, that becomes the final k
  - If all methods disagree, the median value is used

Performance metrics are calculated by comparing clustering results with the ground truth labels.


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
