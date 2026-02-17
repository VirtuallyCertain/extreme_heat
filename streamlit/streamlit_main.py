import streamlit as st
import vis_and_preprocessing as vp
import motivation

st.set_page_config(page_title="Extreme Temperatures Project", layout="wide")

st.sidebar.markdown("# 🌡️ Extreme Heat Events")
st.sidebar.title("Project Navigation")
page = st.sidebar.radio("Go to:", 
    ["0. Introduction & Data Exploration", "1. Preprocessing & Feature Engineering", "2. Model Training & Evaluation", "3. Model Insights & Predictions", "4. Conclusion & Next Steps"])

if page == "0. Introduction & Data Exploration":
    motivation.show_page()

elif page == "1. Preprocessing & Feature Engineering":
    vp.show_page()

elif page == "2. Model Training & Evaluation":
    st.title("Modeling")
    st.write("Content for Modeling goes here...")

elif page == "3. Model Insights & Predictions":
    st.title("Results")
    st.write("Content for Results goes here...")

elif page == "4. Conclusion & Next Steps":
    st.title("Explain Results")
    st.write("Content for Explain Results goes here...")
