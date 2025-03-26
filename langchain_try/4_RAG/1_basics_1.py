import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chromadb")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Iniializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    loader = TextLoader(file_path = file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n---- Creating Embeddings ----")
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model = "models/embedding-001"
    # )
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
    # other params...
    )
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small"
    # ) 
    print("\n---- Embedding Creation Complete ----")

    print("\n---- Creating Vector Store ----")
    db = Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        persist_directory = persistent_directory,
    )
    print("\n---- Vector Store Creation Complete ----")

else:
    print("\n Vector store already exists. No need to initialize.")





