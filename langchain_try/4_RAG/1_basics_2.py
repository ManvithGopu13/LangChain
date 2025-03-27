import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chromadb")

embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
)

db = Chroma(
    embedding_function=embeddings,
    persist_directory=persistent_directory,
)

query =  "Where does Gandalf meet Frodo?"

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
    if doc.metadata:
        print(f"Metadata: {doc.metadata.get('source', 'Unknown')}\n")
