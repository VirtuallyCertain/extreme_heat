import streamlit as st
import vis_and_preprocessing as vp
import motivation

st.set_page_config(page_title="Extreme Temperatures Project", layout="wide")

st.sidebar.markdown("# 🌡️ Extreme Heat Events")
st.sidebar.title("Project Navigation")
page = st.sidebar.radio("Go to:", 
    ["0. Introduction & Motivation", "1. Data Explanation & First Model", "2. Improvements & Final Model", "3. Conclusion & Next Steps"])

if page == "0. Introduction & Motivation":
    motivation.show_page()

elif page == "1. Data Explanation & First Model":
    #vp.show_page()
    st.title("Data Explanation & First Model")
    st.write("Content goes here...")

elif page == "2. Improvements & Final Model":
    st.title("Improvements & Final Model")
    st.write("Content goes here...")

elif page == "3. Conclusion & Next Steps":
    st.title("Conclusion & Next Steps")
    st.write("Content goes here...")
