"""Streamlit user interface for asking manual questions.

This file builds the simple browser UI. The user types a question and clicks
Ask. The UI sends that question to the FastAPI backend, waits for the generated
answer, and displays only the final answer to keep the screen clean. If the
backend is not running or returns an error, the UI shows an error message.
"""

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Askr's AI Assistant", layout="wide")
st.title("Askr's AI Assistant")

question = st.text_input("Question", placeholder="Ask a question from the manuals")

if st.button("Ask", type="primary") and question.strip():
    try:
        response = requests.get(
            f"{BACKEND_URL}/ask",
            params={"q": question},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        st.error(f"Backend request failed: {exc}")
        st.stop()

    st.subheader("Answer")
    st.write(data.get("answer", "Not found in manual"))
