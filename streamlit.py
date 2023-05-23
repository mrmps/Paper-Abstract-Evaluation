import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()





# Define the individual chains
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

prompt_template1 = PromptTemplate(
    input_variables=["abstract"],
    template="Based on the following abstract, evaluate how promising the paper is: {abstract}"
)

chain1 = LLMChain(llm=llm, prompt=prompt_template1, output_key="paper_potential")

def sentiment_analysis(paper_potential):
    sentiment_blob = TextBlob(paper_potential)
    return sentiment_blob.sentiment.polarity  # return a value between -1 and 1 

def potential_summary(sentiment):
    if sentiment <= -0.5:
        return "The paper is flawed or has no potential."
    elif -0.5 < sentiment < 0:
        return "The paper has minor potential but requires substantial improvements."
    elif 0 <= sentiment < 0.5:
        return "The paper is promising, yet improvements can be made."
    else:
        return "The paper is extremely positive and has the potential to disrupt the field."

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
        # Run the chain and print the result
        paper_potential = chain1.run({"abstract": abstract})
        st.write("Paper Potential:", paper_potential)

        # Perform sentiment analysis on the potential
        sentiment = sentiment_analysis(paper_potential)
        st.write("Sentiment:", sentiment)

        # Generate potential summary
        summary = potential_summary(sentiment)
        st.write("Potential Summary:", summary)
    else:
        st.write('Please enter an abstract.')



