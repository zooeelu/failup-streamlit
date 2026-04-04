import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FailUp", layout="wide")

with open("failup.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=1200, scrolling=True)

st.divider()
st.subheader("Streamlit test area")
if st.button("Click me"):
    st.success("Streamlit is working underneath the HTML prototype.")
