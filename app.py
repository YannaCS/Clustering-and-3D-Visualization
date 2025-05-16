import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Clustering Analysis", 
    page_icon="📊",
    layout="wide"
)

# Add title and description
st.title("Clustering Analysis Dashboard")
st.write("""
This application demonstrates the use of K-means and Hierarchical clustering algorithms 
on 3D data. Select a page from the sidebar to explore different datasets and features.
""")

# Display navigation options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Built-in Datasets", "Upload Your Data"])

# Load appropriate page based on selection
if page == "Built-in Datasets":
    import pages.built_in_data as page_module
    page_module.show()
elif page == "Upload Your Data":
    import pages.upload_data as page_module
    page_module.show()

# Add footer
st.markdown("---")
st.write("Clustering Analysis Dashboard created with Streamlit")