import streamlit as st
import streamlit.components.v1 as components
from modules.claude_api import summarize_study

st.set_page_config(page_title="FailUp", layout="wide")

with open("failup.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=1200, scrolling=True)

st.divider()
st.subheader("Real summary form")

with st.form("study_summary_form"):
    title = st.text_input("Study title")
    study_type = st.selectbox(
        "Study type",
        ["RCT", "Observational", "Preclinical", "Other"]
    )
    result_type = st.selectbox(
        "Result type",
        ["Null", "Inconclusive", "Opposite"]
    )
    intervention = st.text_area("Intervention")
    population = st.text_area("Population")
    outcome = st.text_input("Primary outcome")
    p_value = st.text_input("P-value")
    limitations = st.text_area("Main limitations")

    submitted = st.form_submit_button("Generate summary")

if submitted:
    try:
        with st.spinner("Generating summary..."):
            summary = summarize_study(
                title=title,
                study_type=study_type,
                result_type=result_type,
                intervention=intervention,
                population=population,
                outcome=outcome,
                p_value=p_value,
                limitations=limitations,
            )

        st.success("Summary generated.")
        st.write(summary)

    except Exception as e:
        st.error("Summary generation failed.")
        st.code(str(e))
