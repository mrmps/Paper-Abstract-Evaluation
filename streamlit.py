import streamlit as st
import requests

st.title('Paper Abstract Evaluation')


st.markdown("""
This application uses a combination of FastAPI, Streamlit, and OpenAI's language model (LLM) to evaluate the potential of academic papers based on their abstracts.

The process is as follows:

1. The user enters the abstract of a paper in the text area.
2. When the user clicks the 'Evaluate' button, the abstract is sent to a FastAPI application.
3. The FastAPI application uses an LLM to generate an evaluation of the paper's potential.
4. The evaluation is then analyzed using sentiment analysis to generate a numerical score.
5. This score is used to generate a summary of the paper's potential.
6. The evaluation, score, and summary are then returned to the Streamlit app and displayed to the user.

This is a novel approach because it uses the LLM's ability to understand and generate human-like text to provide a concrete score for the potential of a paper. This score is not based on simple keyword matching or other traditional NLP techniques, but on the LLM's understanding of the content of the abstract. This allows for a more nuanced and accurate evaluation of the paper's potential.
""")


abstract = st.text_area('Enter the abstract of the paper:', '')

if st.button('Evaluate'):
    if abstract:
        response = requests.post('http://localhost:8000/evaluate_paper_potential', params={'abstract': abstract})
        if response.status_code == 200:
            data = response.json()
            st.write('Paper Potential:', data['result']['potential'])
            st.write('Sentiment:', data['result']['sentiment'])
            st.write('Potential Summary:', data['result']['summary'])
        else:
            st.write('Error:', response.text)
    else:
        st.write('Please enter an abstract.')


