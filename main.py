from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

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

@app.post("/evaluate_paper_potential")
async def evaluate_paper_potential_endpoint(abstract: str):
    # Run the chain and print the result
    paper_potential = chain1.run({"abstract": abstract})
    print("Paper Potential:", paper_potential)

    # Perform sentiment analysis on the potential
    sentiment = sentiment_analysis(paper_potential)
    print("Sentiment:", sentiment)

    # Generate potential summary
    summary = potential_summary(sentiment)
    print("Potential Summary:", summary)

    # Update your response
    return {
        "message": "Paper potential evaluation complete", 
        "result": {
            "potential": paper_potential, 
            "sentiment": sentiment, 
            "summary": summary
        }
    }
