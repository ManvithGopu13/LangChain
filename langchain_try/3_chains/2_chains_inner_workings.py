from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = os.getenv("GEMINI_API_KEY")
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You love facts and you tell facts about {animal}"),
    ("human", "Tell me {count} facts."),
])

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first = format_prompt, middle = [invoke_model], last = parse_output)

response = chain.invoke({
    "animal": "cat",
    "count": 2,
})

print(response)