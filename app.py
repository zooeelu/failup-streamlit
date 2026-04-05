import json
import streamlit as st
import streamlit.components.v1 as components
from modules.claude_api import summarize_study

st.set_page_config(page_title="FailUp", layout="wide")

# -----------------------------
# Session state setup
# -----------------------------
if "submissions" not in st.session_state:
    st.session_state.submissions = []

# -----------------------------
# HTML prototype display
# -----------------------------
with open("failup.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=1200, scrolling=True)

st.divider()
st.subheader("Generate a structured FailUp summary")

# -----------------------------
# Submission form
# -----------------------------
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

# -----------------------------
# Generate + save submission
# -----------------------------
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

        record = {
            "study_title": title,
            "study_type": study_type,
            "result_type": result_type,
            "intervention": intervention,
            "population": population,
            "primary_outcome": outcome,
            "p_value": p_value,
            "limitations_input": limitations,
            "summary": summary,
        }

        st.session_state.submissions.insert(0, record)

        st.success("Summary generated and saved to this session.")

    except Exception as e:
        st.error("Summary generation failed.")
        st.code(str(e))

# -----------------------------
# Latest saved submission
# -----------------------------
if st.session_state.submissions:
    latest = st.session_state.submissions[0]
    summary = latest["summary"]

    st.divider()
    st.subheader("Latest saved submission")

    st.markdown(f"**Study title:** {latest.get('study_title', '')}")
    st.markdown(f"**Study type:** {latest.get('study_type', '')}")
    st.markdown(f"**Result type:** {latest.get('result_type', '')}")

    st.markdown("### Background")
    st.write(summary.get("background", ""))

    st.markdown("### Findings")
    st.write(summary.get("findings", ""))

    st.markdown("### Main limitation")
    st.write(summary.get("main_limitation", ""))

    st.markdown("### Failure mode")
    st.info(summary.get("failure_mode", ""))

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

    st.download_button(
        label="Download latest submission as JSON",
        data=json.dumps(latest, indent=2),
        file_name="failup_latest_submission.json",
        mime="application/json",
    )

# -----------------------------
# Recent submissions
# -----------------------------
if st.session_state.submissions:
    st.divider()
    st.subheader("Recent submissions")

    for i, submission in enumerate(st.session_state.submissions[:5], start=1):
        with st.expander(f"{i}. {submission['study_title']} ({submission['result_type']})"):
            st.markdown(f"**Study type:** {submission['study_type']}")
            st.markdown(f"**Population:** {submission['population']}")
            st.markdown(f"**Primary outcome:** {submission['primary_outcome']}")
            st.markdown(f"**P-value:** {submission['p_value']}")
            st.markdown(f"**Failure mode:** {submission['summary'].get('failure_mode', '')}")
            st.markdown(f"**Background:** {submission['summary'].get('background', '')}")
            st.markdown(f"**Findings:** {submission['summary'].get('findings', '')}")
