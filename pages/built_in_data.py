import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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
        display_results(results)

def display_results(results):
    """Display the results of clustering analysis"""
    # Create tabs for organizing results
    tabs = st.tabs(["Optimal Clusters", "3D Visualization", "Performance Metrics", "Detailed Results"])
    
    # Optimal Clusters tab
    with tabs[0]:
        st.subheader("Determining the Optimal Number of Clusters")
        st.pyplot(results['k_plot'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best K for K-means", results['best_k_kmeans'])
        with col2:
            st.metric("Best K for Hierarchical", results['best_k_hierarchical'])
        with col3:
            st.metric("Chosen K for Analysis", results['best_k'])
        
        with st.expander("Explanation of Metrics"):
            st.write("""
            - **Elbow Plot**: Shows the sum of squared distances from each point to its assigned center. 
              The optimal K value is where the curve begins to flatten (the "elbow").
            - **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters.
              Values range from -1 to 1, with higher values indicating better clustering.
            - **Calinski-Harabasz (CH) Score**: Ratio of between-cluster dispersion to within-cluster dispersion.
              Higher values indicate better-defined clusters.
            """)
    
    # 3D Visualization tab
    with tabs[1]:
        st.subheader("3D Visualization of Clusters")
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
            """)
    
    # Performance Metrics tab
    with tabs[2]:
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
    with tabs[3]:
        st.subheader("Detailed Performance Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("K-means Performance for all K values:")
            st.dataframe(results['kmeans_performances'])
        
        with col2:
            st.write("Hierarchical Clustering Performance for all K values:")
            st.dataframe(results['hierarchical_performances'])