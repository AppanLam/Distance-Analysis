import streamlit as st
import numpy as np

st.title("Product Lookup: Top Nearest Neighbors")

if 'df' not in st.session_state or 'dist_matrix' not in st.session_state:
    st.error("Load a dataset first from Dataset Selector.")
else:
    df = st.session_state['df']
    dist_matrix = st.session_state['dist_matrix']
    
    product = st.selectbox("Select Product", df.index)
    k = st.slider("Top K Neighbors", 1, 10, 5)
    
    product_idx = list(df.index).index(product)
    dists = dist_matrix[product_idx]
    sorted_idxs = np.argsort(dists)[1:k+1]  # Exclude self
    neighbors = [df.index[i] for i in sorted_idxs]
    neighbor_dists = [dists[i] for i in sorted_idxs]
    
    st.subheader(f"Top {k} Neighbors for {product}")
    for neigh, dist in zip(neighbors, neighbor_dists):
        st.write(f"- {neigh}: Distance {dist:.2f}")
