from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()
llm = ChatGroq(groq_api_key = os.getenv("GROQ_API_KEY"), model_name = "meta-llama/llama-4-maverick-17b-128e-instruct")


if __name__ == "__main__":
    response = llm.invoke("What are two main ingredients in samosa?")
    print(response.content)
