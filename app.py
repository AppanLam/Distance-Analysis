import streamlit as st
import pandas as pd
import os

from dashboard.viz import load_distance_matrix, load_preprocessed_numeric, compute_pca_embedding, plot_embedding, plot_distance_histogram, plot_neighbor_bar

st.set_page_config(page_title="Strategic Product Optimizer", layout="wide")
st.title("Product Portfolio Analysis")

# Sidebar for category selection
category = st.sidebar.selectbox(
    "Select Product Category",
    ["Smartphones", "Tractors", "User Upload"]
)

# Map the selection to Peter's folder names
folder_name = category.lower().replace(" ", "_")
base_path = os.path.join("outputs", folder_name)

# Paths for the specific files in Peter's folder structure
nn_path = os.path.join(base_path, "nearest_neighbors.csv")
summary_path = os.path.join(base_path, "distance_summary.txt")

# --- DISPLAY SUMMARY STATS ---
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        summary_text = f.read()
    st.sidebar.info(f"**Category Summary:**\n\n{summary_text}")

# --- MAIN ANALYSIS ---
if os.path.exists(nn_path):
    nn_df = pd.read_csv(nn_path)
    
    # Selection UI
    product_list = sorted(nn_df.iloc[:, 0].unique())
    selected_product = st.selectbox("Select a Product to find neighbors", product_list)
    
    # Filter Results
    results = nn_df[nn_df.iloc[:, 0] == selected_product].sort_values('distance')
    
    # Layout with Columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Top Nearest Neighbors for {selected_product}")
        display_results = results.iloc[:, [2, -1]].head(5)
        display_results.columns = ['Neighbor Name', 'Distance Score']
        st.table(display_results)

    with col2:
        st.subheader("Market Impact Analysis")
        top_neighbor_dist = results['distance'].iloc[0] # The closest distance
        
        if top_neighbor_dist < 0.05:
            st.error(f"Cannibalization Alert: Extreme overlap detected (Dist: {top_neighbor_dist:.4f})")
        elif top_neighbor_dist < 0.50:
            st.warning(f"Significant Overlap: High competition detected (Dist: {top_neighbor_dist:.4f})")
        else:
            st.success(f"Unique Positioning: Product is well-differentiated (Dist: {top_neighbor_dist:.4f})")
            
        st.info("Rule Logic: Cannibalization < 0.05 | Overlap < 0.50 | Unique > 0.50")

    # --- Task 2 Visualizations ---
    dm_path = os.path.join(base_path, "distance_matrix.csv")
    pre_path = os.path.join(base_path, "preprocessed_output.csv")

    if os.path.exists(dm_path) and os.path.exists(pre_path):
        st.subheader("Visualizations (Task 2)")

        # 1) Neighbor bar
        topk = results.sort_values("distance").head(10)
        nb_df = topk.iloc[:, [2, -1]].copy()
        nb_df.columns = ["neighbor", "distance"]
        st.plotly_chart(plot_neighbor_bar(nb_df, selected_product), use_container_width=True)

        # 2) Distance histogram
        dm = load_distance_matrix(dm_path)
        st.plotly_chart(plot_distance_histogram(dm), use_container_width=True)

        # 3) PCA embedding + highlight selected + neighbors
        pre = load_preprocessed_numeric(pre_path)
        emb_df = compute_pca_embedding(pre, list(dm.index))
        st.plotly_chart(
            plot_embedding(emb_df, selected=selected_product, neighbors=nb_df["neighbor"].tolist()),
            use_container_width=True
        )
    else:
        st.warning("Missing distance_matrix.csv or preprocessed_output.csv for visualizations.")

else:
    st.error(f"No analysis data found for {category}. Please ensure the distance matrix script has been run.")