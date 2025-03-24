from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = os.getenv("GEMINI_API_KEY")
)

result = llm.invoke("What is the capital of India?")

print(result.content)