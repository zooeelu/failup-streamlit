import anthropic
import streamlit as st

def summarize_study(title, study_type, result_type, intervention, population, outcome, p_value, limitations):
    client = anthropic.Anthropic(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        system=(
            "You are a scientific summarization assistant for FailUp, a repository "
            "of negative biomedical research findings. Be precise, neutral, and concise. "
            "Do not speculate beyond what was provided. If information is missing, say so briefly."
        ),
        messages=[
            {
                "role": "user",
                "content": f"""
Summarize this negative, null, or inconclusive biomedical study.

Study title: {title}
Study type: {study_type}
Result type: {result_type}
Intervention: {intervention}
Population: {population}
Primary outcome: {outcome}
P-value: {p_value}
Limitations: {limitations}
"""
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "background": {"type": "string"},
                        "findings": {"type": "string"},
                        "main_limitation": {"type": "string"},
                        "failure_mode": {"type": "string"},
                        "contradiction_check": {"type": "string"},
                        "graph_tags": {
                            "type": "object",
                            "properties": {
                                "mechanism": {"type": "string"},
                                "target": {"type": "string"},
                                "population": {"type": "string"},
                                "therapeutic_area": {"type": "string"}
                            },
                            "required": ["mechanism", "target", "population", "therapeutic_area"],
                            "additionalProperties": False
                        }
                    },
                    "required": [
                        "background",
                        "findings",
                        "main_limitation",
                        "failure_mode",
                        "contradiction_check",
                        "graph_tags"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    return response.content[0].text
