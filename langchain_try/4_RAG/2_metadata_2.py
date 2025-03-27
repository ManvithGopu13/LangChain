import os
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

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

query =  "Where is Dracula's castle located?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k": 5,
        "score_threshold": 0.5,
    }
)
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n {doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")
