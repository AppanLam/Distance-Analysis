import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Feature Analysis: Scatterplots")

if 'df' not in st.session_state:
    st.error("Load a dataset first.")
else:
    df = st.session_state['df']
    features = df.select_dtypes(include=np.number).columns.tolist()
    
    feat1 = st.selectbox("Select Feature 1", features)
    feat2 = st.selectbox("Select Feature 2", features)
    
    if feat1 and feat2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feat1, y=feat2, ax=ax)
        ax.set_title(f"Scatterplot: {feat1} vs {feat2}")
        st.pyplot(fig)
    
    # Optional: Multiple pairs
    st.subheader("Batch Plots")
    pairs = [('featA', 'featB'), ('featC', 'featD')]  # Hardcode key pairs or let user input
    for f1, f2 in pairs:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=f1, y=f2, ax=ax)
        st.pyplot(fig)
