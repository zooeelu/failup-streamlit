import json
import streamlit as st
import streamlit.components.v1 as components
from modules.claude_api import summarize_study

st.set_page_config(page_title="FailUp", layout="wide")

with open("failup.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=1200, scrolling=True)

st.divider()
st.subheader("Generate a structured FailUp summary")

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
        with st.spinner("Generating structured summary..."):
            summary_raw = summarize_study(
                title=title,
                study_type=study_type,
                result_type=result_type,
                intervention=intervention,
                population=population,
                outcome=outcome,
                p_value=p_value,
                limitations=limitations,
            )

        summary = json.loads(summary_raw)

        st.success("Summary generated.")

        st.markdown("### Background")
        st.write(summary.get("background", ""))

        st.markdown("### Findings")
        st.write(summary.get("findings", ""))

        st.markdown("### Main limitation")
        st.write(summary.get("main_limitation", ""))

        st.markdown("### Failure mode")
        failure_mode = summary.get("failure_mode", "")
        st.info(failure_mode)

        st.markdown("### Contradiction check")
        st.write(summary.get("contradiction_check", ""))

        st.markdown("### Graph tags")
        graph_tags = summary.get("graph_tags", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Mechanism:** {graph_tags.get('mechanism', '')}")
            st.markdown(f"**Target:** {graph_tags.get('target', '')}")
        with col2:
            st.markdown(f"**Population:** {graph_tags.get('population', '')}")
            st.markdown(f"**Therapeutic area:** {graph_tags.get('therapeutic_area', '')}")

    except Exception as e:
        st.error("Summary generation failed.")
        st.code(str(e))
