import streamlit as st
from dashboard.simulation import simulate_removal

st.title("Simulation: Product Removal")

if 'df' not in st.session_state or 'dist_matrix' not in st.session_state:
    st.error("Load a dataset first.")
else:
    df = st.session_state['df']
    dist_matrix = st.session_state['dist_matrix']
    
    products_to_remove = st.multiselect("Select Products to Remove", df.index)
    threshold = st.number_input("Optional Gap Threshold", min_value=0.0, value=None)
    
    if st.button("Simulate Removal"):
        result = simulate_removal(df, dist_matrix, products_to_remove, threshold)
        if 'error' in result:
            st.error(result['error'])
        else:
            st.subheader("Simulation Results")
            st.write(f"Mean Substitute Distance: {result['mean_dist']:.2f}")
            st.write(f"Max Substitute Distance: {result['max_dist']:.2f}")
            
            st.subheader("Substitutes")
            for removed, info in result['substitutes'].items():
                st.write(f"- Removed {removed} -> Substitute {info['substitute']} (dist {info['distance']:.2f})")
            
            if threshold:
                st.write("Gaps (dist > threshold):", ", ".join(result.get('gaps', [])))
