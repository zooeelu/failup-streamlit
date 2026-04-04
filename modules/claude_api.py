import anthropic
import streamlit as st

def test_connection():
    client = anthropic.Anthropic(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Say hello in one short sentence."
            }
        ]
    )

    return response.content[0].text


def summarize_study(title, study_type, result_type, intervention, population, outcome, p_value, limitations):
    client = anthropic.Anthropic(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

    prompt = f"""
You are helping summarize a negative or inconclusive biomedical study for a research repository.

Please write a clear short summary with these 3 labeled sections:
Background:
Findings:
Main limitation:

Study title: {title}
Study type: {study_type}
Result type: {result_type}
Intervention: {intervention}
Population: {population}
Primary outcome: {outcome}
P-value: {p_value}
Limitations: {limitations}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text
