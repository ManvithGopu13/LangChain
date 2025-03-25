from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = os.getenv("GEMINI_API_KEY")
)

summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie_name}."),
    ]
)

def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

def analyze_characters(characters):
    characters_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?"),
        ]
    )
    return characters_template.format_prompt(characters=characters)

def combine_versicts(plot_analysis, characters_analysis):
    return f"Plot analysis: \n{plot_analysis}\n\nCharacters analysis: \n{characters_analysis}"

plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

chain  = (
    summary_template 
    | model 
    | StrOutputParser() 
    | RunnableParallel(branches = {"plot": plot_branch_chain, "characters": character_branch_chain}) 
    | RunnableLambda(lambda x: combine_versicts(x["branches"]["plot"], x["branches"]["characters"]))
)

result = chain.invoke({
    "movie_name": "The Matrix"
})

print(result)
