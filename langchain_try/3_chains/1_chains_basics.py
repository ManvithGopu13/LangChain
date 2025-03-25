from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = os.getenv("GEMINI_API_KEY")
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who knows facts about {animal}."),
    ("human", "Tell me {fact_count} facts."),
])

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({
    "animal": "elephants",
    "fact_count": 2,
})

print(result)
