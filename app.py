import streamlit as st
import anthropic

st.set_page_config(page_title="FailUp", layout="wide")

st.title("FailUp")
st.write("Testing Anthropic API connection.")

if "ANTHROPIC_API_KEY" in st.secrets:
    st.success("Anthropic API key found in Streamlit secrets.")
else:
    st.error("Anthropic API key not found.")

if st.button("Test Anthropic connection"):
    try:
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

        st.success("API call worked.")
        st.write(response.content[0].text)

    except Exception as e:
        st.error("API call failed.")
        st.code(str(e))


