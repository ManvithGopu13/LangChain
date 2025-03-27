from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import datetime
from langchain.agents import tool

load_dotenv()


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current system time in the specified format."""

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

model = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = os.getenv("GOOGLE_API_KEY"),
)

# model = ChatMistralAI(
#     model="mistral-large-latest",
#     api_key=os.getenv("MISTRAL_API_KEY"),
# )


query = "What's the current time in London? (You are in India) Just show the current time and not the date. (London is 5 hrs and 30 mins behind India)"

prompt_template = hub.pull("hwchase17/react")

tools = [get_system_time]

agent = create_react_agent(
    llm = model,
    tools = tools,
    prompt = prompt_template,
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
)

result = agent_executor.invoke({"input": query})

print(result)