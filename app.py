import streamlit as st
import streamlit.components.v1 as components
from modules.claude_api import test_connection

st.set_page_config(page_title="FailUp", layout="wide")

with open("failup.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=1200, scrolling=True)

st.divider()
st.subheader("Streamlit test area")

if st.button("Test Claude helper"):
    try:
        result = test_connection()
        st.success("Claude helper worked.")
        st.write(result)
    except Exception as e:
        st.error("Claude helper failed.")
        st.code(str(e))
