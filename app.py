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
# Remove the app menu from the sidebar
# Hide the sidebar menu items
st.markdown("""
<style>
    /* Hide the default sidebar nav */
    .css-k8kh4d, #MainMenu, footer, header, [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Hide "App" and "built_in_data" from the sidebar */
    section[data-testid="stSidebar"] .css-1adrfps {
        display: none !important;
    }
    
    /* Hide the top right hamburger menu */
    .css-r698ls {
        display: none;
    }
    
    /* Hide the footer "Made with Streamlit" */
    .css-rncmk8 {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Add title and description
st.title("Clustering Analysis Dashboard")
st.write("""
This application demonstrates the use of K-means and Hierarchical clustering algorithms 
on 3D data. You can use built-in datasets or upload your own data for analysis.
""")

# Display navigation options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Dataset Analysis"])

# Import the built_in_data module (now handles both built-in and uploaded data)
try:
    from pages import built_in_data as page_module
    page_module.show()
except ImportError:
    # If the module doesn't exist in the pages package, try direct import
    try:
        import pages.built_in_data as page_module
        page_module.show()
    except ImportError:
        st.error("The built_in_data module could not be found. Please make sure the file exists.")

# Add footer
st.markdown("---")
st.write("Clustering Analysis Dashboard created by Yanna")