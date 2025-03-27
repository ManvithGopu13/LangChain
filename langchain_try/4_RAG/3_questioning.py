import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir , "chromadb_with_metadata")

embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
)

db = Chroma(
    embedding_function=embeddings,
    persist_directory=persistent_directory,
)

query = "What does dracula fear the most?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k": 5,
        "score_threshold": 0.3,
    }
)
relevant_docs = retriever.invoke(query)

combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\n Relevant documents:\n"
    + "\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

model = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
)

# model = ChatGoogleGenerativeAI(
#     model = "gemini-1.5-pro",
#     api_key = os.getenv("GEMINI_API_KEY")
# )

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n---- Generated response ----")
print(result.content)