from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = os.getenv("GEMINI_API_KEY")
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are an expert in geography."), 
    HumanMessage("Give me the shortest path to reach from hyderabad, India to Dubai, UAE.")
]

result = llm.invoke(messages)

print(result.content)