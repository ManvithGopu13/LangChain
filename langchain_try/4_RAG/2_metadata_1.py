import os 
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chromadb_with_metadata")

print(f"Books Directory: {books_dir}")
print(f"Persistent Directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )
    
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path=file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    print("---- Creating Embeddings ----")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    print("---- Embedding Creation Complete ----")

    print("---- Creating Vector Store ----")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory,
    )
    print("---- Vector Store Creation Complete ----")

else:
    print("\n Vector store already exists. No need to initialize.")