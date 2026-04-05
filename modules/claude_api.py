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
            "of negative biomedical research findings. "
            "Be precise, neutral, concise, and scientifically cautious. "
            "Do not speculate beyond what was explicitly provided. "
            "Do not invent literature citations, prior studies, or claims of verification. "
            "Do not reframe negative results positively. "
            "If information is insufficient, say so plainly."
        ),
        messages=[
            {
                "role": "user",
                "content": f"""
Summarize this negative, null, or inconclusive biomedical study.

Return ONLY valid JSON that matches the schema.

Important rules for the fields:
- background: briefly state the study rationale using only the provided information.
- findings: summarize the result type and reported statistical information without exaggeration.
- main_limitation: use the provided limitation if available; otherwise say that no limitation was explicitly provided.
- failure_mode: choose from this exact list only:
  "target validity"
  "patient selection"
  "dosing/PK"
  "underpowered"
  "outcome measurement"
  "off-target effects"
  "unknown"
  If the provided information is not enough to classify confidently, use "unknown".
- contradiction_check: do NOT claim that literature was checked unless that was explicitly provided. If no such information was provided, say "Not assessed from provided information."
- graph_tags: create short, useful tags based only on provided information. If unknown, say "unspecified".

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
                        "failure_mode": {
                            "type": "string",
                            "enum": [
                                "target validity",
                                "patient selection",
                                "dosing/PK",
                                "underpowered",
                                "outcome measurement",
                                "off-target effects",
                                "unknown"
                            ]
                        },
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
