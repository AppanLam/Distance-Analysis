import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dashboard.simulation import simulate_removal  # Import

st.title("Portfolio View")

if 'df' not in st.session_state or 'dist_matrix' not in st.session_state:
    st.error("Load a dataset first.")
else:
    df = st.session_state['df']
    dist_matrix = st.session_state['dist_matrix']
    
    threshold_overlap = st.number_input("Overlap Threshold (dist < this = overlap)", 0.0, 1.0, 0.1)
    threshold_gap = st.number_input("Gap Threshold (dist > this = gap)", 0.5, 5.0, 1.0)
    
    # Flag overlaps
    overlaps = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if dist_matrix[i,j] < threshold_overlap:
                overlaps.append((df.index[i], df.index[j], dist_matrix[i,j]))
    
    st.subheader("Flagged Overlaps")
    if overlaps:
        for a, b, d in overlaps:
            st.write(f"- {a} and {b}: Distance {d:.2f}")
    else:
        st.write("No overlaps below threshold.")
    
    # Simple gap view: Min dist per product
    min_dists = [np.min(dist_matrix[i, np.arange(len(df)) != i]) for i in range(len(df))]
    fig, ax = plt.subplots()
    sns.histplot(min_dists, ax=ax)
    ax.set_title("Distribution of Min Distances (Gaps)")
    st.pyplot(fig)
    
    # Optional: Flag large min_dists as gaps
    gaps = [df.index[i] for i, md in enumerate(min_dists) if md > threshold_gap]
    if gaps:
        st.write("Products with potential gaps (large min dist):", ", ".join(gaps))
