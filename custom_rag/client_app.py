## Import Libraries

import streamlit as st
import requests


## Constants


API_PATH = "http://localhost:8000/custom_rag"


## Functions


def get_response(input_text: str) -> str:
    """
    Sends a POST request to an API endpoint with input_text as JSON data and retrieves the response.

    Parameters:
    input_text (str): The text to be sent as input to the API endpoint.

    Returns:
    output (str): The output retrieved from the API response.
    """
    response = requests.post(API_PATH, json={"input": input_text})
    output = response.json()["output"]
    return output


if __name__ == "__main__":
    st.title("Custom RAG based on LangChain and HuggingFace")
    query = st.text_input("Search a topic")
    if query:
        st.write(get_response(query))
