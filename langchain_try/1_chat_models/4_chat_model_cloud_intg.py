from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = os.getenv("GEMINI_API_KEY")
)

FIRESTORE_SESSION_ID = "user_session_new"
FIRESTORE_COLLECTION = "chat_history"

print("Initialzing FireStore Client....")
client = firestore.Client(project=os.getenv("FIRESTORE_PROJECT_ID"))
print("FireStore Client Initialized.")

print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id= FIRESTORE_SESSION_ID,
    collection= FIRESTORE_COLLECTION,
    client= client,
)

print("Chat History initialized.")
print("Current chat history:" , chat_history.messages)

print("Start chatting with the AI. Type 'exit' to stop.")

while True:
    human_input = input("You: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)

    ai_response = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")