import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# Assume a function to load data; adapt from preprocessing.py
def load_dataset(dataset_name):
    if dataset_name == 'smartphones':
        return pd.read_csv('preprocessed_output_smartphones.csv', index_col=0)
    elif dataset_name == 'tractors':
        # Load from Database_Distance Analysis.xlsx, filter for tractors
        df = pd.read_excel('Database_Distance Analysis.xlsx')
        return df[df['category'] == 'tractors']  # Assume column; adjust
    # Add user upload: st.file_uploader
    else:
        return pd.DataFrame()  # Placeholder

st.title("Dataset Selector and Summary Stats")

dataset_options = ['smartphones', 'tractors', 'user_upload']
selected_dataset = st.selectbox("Select Dataset", dataset_options)

df = load_dataset(selected_dataset)
if df.empty:
    st.error("No data loaded.")
else:
    # Compute distance matrix if not preloaded
    features = df.select_dtypes(include=np.number)  # Numerical features
    dist_matrix = pairwise_distances(features)
    
    n = len(df)
    d = features.shape[1]
    min_dist = np.min(dist_matrix[np.triu_indices(n, k=1)])  # Upper triangle, exclude self
    mean_dist = np.mean(dist_matrix[np.triu_indices(n, k=1)])
    max_dist = np.max(dist_matrix[np.triu_indices(n, k=1)])
    
    st.subheader("Summary Stats")
    st.write(f"Number of products (n): {n}")
    st.write(f"Feature dimensions (d): {d}")
    st.write(f"Min pairwise distance: {min_dist:.2f}")
    st.write(f"Mean pairwise distance: {mean_dist:.2f}")
    st.write(f"Max pairwise distance: {max_dist:.2f}")
    
    # Cache df and dist_matrix in session state for other pages
    st.session_state['df'] = df
    st.session_state['dist_matrix'] = dist_matrix
  
