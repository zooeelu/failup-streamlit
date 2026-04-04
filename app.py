import streamlit as st

st.set_page_config(page_title="FailUp", layout="wide")

st.title("FailUp")
st.write("My first Streamlit app is running.")

if "ANTHROPIC_API_KEY" in st.secrets:
    st.success("Anthropic API key found in Streamlit secrets.")
else:
    st.error("Anthropic API key not found.")

