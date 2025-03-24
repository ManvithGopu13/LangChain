from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = os.getenv("GEMINI_API_KEY")
)

# template = "Write a {tone} email to {company} expessing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max."

# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({
#     "tone": "energetic",
#     "company": "Apple",
#     "position": "AI Engineer",
#     "skill": "AI",  
# })

#Example 2:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({
    "topic": "lawyers",
    "joke_count": 3,
})

result = llm.invoke(prompt)

print(result.content)